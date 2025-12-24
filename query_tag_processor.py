#!/usr/bin/env python3
"""
Query Tag Processor: Generate tags for queries and split into separate CSV files.
Uses existing ID column to group similar queries and generates meaningful tags via OpenAI.
Then generates paraphrased questions and merges results.
"""

import csv
import os
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
from openai import OpenAI

# Import paraphrase generation and merge functions
from generate_paraphrases import run_paraphrase_generation
from merge_results import run_merge


class QueryTagProcessor:
    """
    Processes a CSV file containing queries, answers, and group IDs.
    Generates meaningful tags for each group using OpenAI and splits data into separate CSVs.
    """
    
    def __init__(self):
        """Initialize the processor."""
        self.data: List[Dict[str, str]] = []  # List of {query, answer, id} dicts
        self.id_to_queries: Dict[str, List[str]] = defaultdict(list)  # id -> list of queries
        self.id_to_answer: Dict[str, str] = {}  # id -> answer
        self.id_to_tag: Dict[str, str] = {}  # id -> generated tag
        self.query_tags: Dict[str, str] = {}  # query -> tag mapping
        self.tag_answer: Dict[str, str] = {}  # tag -> answer mapping
        
        # Initialize OpenAI client for tag generation
        try:
            self.client = OpenAI()
            if not os.environ.get("OPENAI_API_KEY"):
                print("ERROR: OPENAI_API_KEY environment variable is not set!")
                print("Please set your OpenAI API key:")
                print("  export OPENAI_API_KEY='your-api-key-here'")
                raise RuntimeError("OPENAI_API_KEY not set")
        except Exception as e:
            print(f"ERROR: Failed to initialize OpenAI client: {e}")
            raise
    
    def load_csv(self, csv_path: str, query_column: str = 'query', 
                 answer_column: str = 'answer', id_column: str = 'id') -> None:
        """
        Load queries, answers, and IDs from a CSV file.
        
        Args:
            csv_path: Path to the input CSV file.
            query_column: Name of the column containing queries.
            answer_column: Name of the column containing answers.
            id_column: Name of the column containing group IDs.
        """
        self.data = []
        self.id_to_queries = defaultdict(list)
        self.id_to_answer = {}
        
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            # Validate columns exist
            if query_column not in reader.fieldnames:
                raise ValueError(f"Column '{query_column}' not found in CSV. Available columns: {reader.fieldnames}")
            if answer_column not in reader.fieldnames:
                raise ValueError(f"Column '{answer_column}' not found in CSV. Available columns: {reader.fieldnames}")
            if id_column not in reader.fieldnames:
                raise ValueError(f"Column '{id_column}' not found in CSV. Available columns: {reader.fieldnames}")
            
            for row in reader:
                query = row[query_column].strip()
                answer = row[answer_column].strip()
                group_id = row[id_column].strip()
                
                if query and answer and group_id:
                    self.data.append({
                        'query': query,
                        'answer': answer,
                        'id': group_id
                    })
                    self.id_to_queries[group_id].append(query)
                    # Store the answer for this ID (all queries with same ID should have same answer)
                    if group_id not in self.id_to_answer:
                        self.id_to_answer[group_id] = answer
        
        unique_ids = len(self.id_to_queries)
        print(f"Loaded {len(self.data)} query-answer pairs from {csv_path}")
        print(f"Found {unique_ids} unique group IDs")
    
    def _generate_tag_for_group(self, group_id: str, queries: List[str], answer: str) -> str:
        """
        Generate a meaningful tag for a group of similar queries using OpenAI.
        
        Args:
            group_id: The group ID.
            queries: List of queries in this group.
            answer: The answer for this group.
            
        Returns:
            A meaningful tag string.
        """
        # Take sample queries (up to 5) for context
        sample_queries = queries[:5]
        queries_text = "\n".join([f"- {q}" for q in sample_queries])
        
        prompt = f"""Based on the following similar queries and their answer, generate a short, descriptive tag (2-4 words) that captures the main topic or intent.

Sample Queries:
{queries_text}

Answer:
{answer[:500]}{"..." if len(answer) > 500 else ""}

Requirements:
- The tag should be 2-4 words maximum
- Use lowercase with underscores between words (e.g., "voter_registration_process")
- The tag should be descriptive and capture the main topic
- Do not include special characters except underscores
- Output ONLY the tag, nothing else

Tag:"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that generates concise, descriptive tags for categorizing questions."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=50,
                temperature=0.3
            )
            
            tag = response.choices[0].message.content.strip()
            # Clean up the tag
            tag = tag.lower().replace(" ", "_").replace("-", "_")
            tag = "".join(c if c.isalnum() or c == '_' else '' for c in tag)
            tag = tag.strip('_')
            
            # Ensure tag is not empty
            if not tag:
                tag = f"tag_{group_id}"
            
            return tag
            
        except Exception as e:
            print(f"WARNING: Failed to generate tag for group {group_id}: {e}")
            return f"tag_{group_id}"
    
    def generate_tags(self) -> Dict[str, str]:
        """
        Generate meaningful tags for each group ID using OpenAI.
        Similar queries (same ID) receive the same tag.
        
        Returns:
            Dictionary mapping queries to their assigned tags.
        """
        if not self.data:
            raise RuntimeError("No data loaded. Call load_csv() first.")
        
        unique_ids = list(self.id_to_queries.keys())
        print(f"Generating tags for {len(unique_ids)} unique groups...")
        
        # Track generated tags to ensure uniqueness
        used_tags = set()
        
        for i, group_id in enumerate(unique_ids):
            queries = self.id_to_queries[group_id]
            answer = self.id_to_answer[group_id]
            
            # Generate tag using OpenAI
            tag = self._generate_tag_for_group(group_id, queries, answer)
            
            # Ensure uniqueness by appending number if needed
            original_tag = tag
            counter = 1
            while tag in used_tags:
                tag = f"{original_tag}_{counter}"
                counter += 1
            
            used_tags.add(tag)
            self.id_to_tag[group_id] = tag
            
            if (i + 1) % 10 == 0 or (i + 1) == len(unique_ids):
                print(f"  Processed {i + 1}/{len(unique_ids)} groups...")
        
        # Build query -> tag mapping
        self.query_tags = {}
        for item in self.data:
            query = item['query']
            group_id = item['id']
            self.query_tags[query] = self.id_to_tag[group_id]
        
        # Build tag -> answer mapping
        self.tag_answer = {}
        for group_id, tag in self.id_to_tag.items():
            self.tag_answer[tag] = self.id_to_answer[group_id]
        
        print(f"✓ Generated {len(self.id_to_tag)} unique tags")
        
        return self.query_tags
    
    def split_to_csv_files(self, output_dir: str = ".") -> Tuple[str, str]:
        """
        Split the data into two CSV files:
        - queries_tags.csv: Contains query and tag columns
        - tags_answers.csv: Contains tag and answer columns
        
        Args:
            output_dir: Directory to save the output files.
            
        Returns:
            Tuple of (queries_tags_path, tags_answers_path).
        """
        if not self.id_to_tag:
            raise RuntimeError("No tags generated. Call generate_tags() first.")
        
        os.makedirs(output_dir, exist_ok=True)
        
        queries_tags_path = os.path.join(output_dir, "queries_tags.csv")
        tags_answers_path = os.path.join(output_dir, "tags_answers.csv")
        
        # Write queries_tags.csv
        print(f"Writing {queries_tags_path}...")
        with open(queries_tags_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['query', 'tag'])
            
            for item in self.data:
                query = item['query']
                group_id = item['id']
                tag = self.id_to_tag[group_id]
                writer.writerow([query, tag])
        
        # Write tags_answers.csv
        print(f"Writing {tags_answers_path}...")
        with open(tags_answers_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['tag', 'answer'])
            
            for tag, answer in self.tag_answer.items():
                writer.writerow([tag, answer])
        
        print(f"✓ Created {queries_tags_path} with {len(self.data)} rows")
        print(f"✓ Created {tags_answers_path} with {len(self.tag_answer)} rows")
        
        return queries_tags_path, tags_answers_path
    
    def process(self, csv_path: str, output_dir: str = ".", 
                query_column: str = 'query', answer_column: str = 'answer',
                id_column: str = 'id', 
                run_paraphrase: bool = False,
                test_mode: bool = False,
                test_tag_count: int = 1) -> Tuple[str, str]:
        """
        Convenience method to run the full pipeline.
        
        Args:
            csv_path: Path to input CSV file.
            output_dir: Directory to save output files.
            query_column: Name of the query column in input CSV.
            answer_column: Name of the answer column in input CSV.
            id_column: Name of the ID column in input CSV (groups similar queries).
            run_paraphrase: If True, also run paraphrase generation and merge.
            test_mode: If True, only process first N tags for paraphrasing.
            test_tag_count: Number of tags to process in test mode.
            
        Returns:
            Tuple of (queries_tags_path, tags_answers_path).
        """
        self.load_csv(csv_path, query_column, answer_column, id_column)
        self.generate_tags()
        queries_tags_path, tags_answers_path = self.split_to_csv_files(output_dir)
        
        # Optionally run paraphrase generation and merge
        if run_paraphrase:
            self.run_full_pipeline(queries_tags_path, tags_answers_path, output_dir, 
                                   test_mode, test_tag_count)
        
        return queries_tags_path, tags_answers_path
    
    def run_full_pipeline(self, queries_tags_path: str, tags_answers_path: str, 
                          output_dir: str, test_mode: bool = False, 
                          test_tag_count: int = 1) -> Optional[str]:
        """
        Run paraphrase generation and merge after tag generation.
        
        Args:
            queries_tags_path: Path to queries_tags.csv (query, tag columns)
            tags_answers_path: Path to tags_answers.csv (tag, answer columns)
            output_dir: Base output directory
            test_mode: If True, only process first N tags
            test_tag_count: Number of tags to process in test mode
            
        Returns:
            Path to the final merged dataset, or None if failed.
        """
        print("\n" + "="*80)
        print("🚀 STARTING PARAPHRASE GENERATION PIPELINE")
        print("="*80)
        
        # Create paraphrase output directory
        paraphrase_output_dir = os.path.join(output_dir, "paraphrased_output")
        os.makedirs(paraphrase_output_dir, exist_ok=True)
        
        print(f"\n📂 Input files:")
        print(f"   - Questions/Tags: {queries_tags_path}")
        print(f"   - Tags/Answers:   {tags_answers_path}")
        print(f"   - Output Dir:     {paraphrase_output_dir}")
        
        # Run paraphrase generation
        print("\n" + "-"*80)
        print("📝 Step 1: Generating paraphrased questions...")
        print("-"*80)
        
        success, failed, skipped = run_paraphrase_generation(
            examples_file=queries_tags_path,
            answers_file=tags_answers_path,
            output_dir=paraphrase_output_dir,
            test_mode=test_mode,
            test_tag_count=test_tag_count
        )
        
        print(f"\n✓ Paraphrase generation complete:")
        print(f"   - Success: {success}")
        print(f"   - Failed:  {failed}")
        print(f"   - Skipped: {skipped}")
        
        # Run merge
        print("\n" + "-"*80)
        print("🔗 Step 2: Merging all generated files...")
        print("-"*80)
        
        individual_tags_dir = os.path.join(paraphrase_output_dir, "individual_tags")
        merged_file = run_merge(input_dir=individual_tags_dir)
        
        if merged_file:
            print(f"\n✓ Final merged dataset: {merged_file}")
        else:
            print("\n✗ Merge failed - no files to merge")
        
        print("\n" + "="*80)
        print("🎉 PIPELINE COMPLETE!")
        print("="*80)
        
        return merged_file


def print_info_box():
    """Print a beautiful info box about input CSV requirements."""
    box_width = 54
    
    print()
    print("╔" + "═" * box_width + "╗")
    print("║" + " " * box_width + "║")
    print("║" + "   📋 INPUT CSV REQUIREMENTS".center(box_width-1) + "║")
    print("║" + " " * box_width + "║")
    print("╠" + "═" * box_width + "╣")
    print("║" + " " * box_width + "║")
    print("║" + "   Input CSV must contain 3 columns:".ljust(box_width) + "║")
    print("║" + " " * box_width + "║")
    print("║" + "     • query   → The user query/question".ljust(box_width) + "║")
    print("║" + "     • answer  → The corresponding answer".ljust(box_width) + "║")
    print("║" + "     • id      → Group ID (similar queries".ljust(box_width) + "║")
    print("║" + "                  share the same ID)".ljust(box_width) + "║")
    print("║" + " " * box_width + "║")
    print("╠" + "═" * box_width + "╣")
    print("║" + " " * box_width + "║")
    print("║" + "   Example:".ljust(box_width) + "║")
    print("║" + "   ┌────────────────────┬───────────┬────┐".ljust(box_width) + "║")
    print("║" + "   │ query              │ answer    │ id │".ljust(box_width) + "║")
    print("║" + "   ├────────────────────┼───────────┼────┤".ljust(box_width) + "║")
    print("║" + "   │ How to vote?       │ You can...│ 1  │".ljust(box_width) + "║")
    print("║" + "   │ Voting process?    │ You can...│ 1  │".ljust(box_width) + "║")
    print("║" + "   │ Get NID card       │ Apply at..│ 2  │".ljust(box_width) + "║")
    print("║" + "   └────────────────────┴───────────┴────┘".ljust(box_width) + "║")
    print("║" + " " * box_width + "║")
    print("╚" + "═" * box_width + "╝")
    print()


def main():
    """Example usage of QueryTagProcessor."""
    import argparse
    
    # Print info box when program runs
    print_info_box()
    
    parser = argparse.ArgumentParser(description='Process queries and generate tags.')
    parser.add_argument('input_csv', help='Path to input CSV file with queries, answers, and IDs')
    parser.add_argument('--output-dir', '-o', default='.', help='Output directory for CSV files')
    parser.add_argument('--query-column', '-q', default='query', help='Name of query column')
    parser.add_argument('--answer-column', '-a', default='answer', help='Name of answer column')
    parser.add_argument('--id-column', '-i', default='id', help='Name of ID column (groups similar queries)')
    parser.add_argument('--generate-paraphrases', '-g', action='store_true', 
                        help='Also run paraphrase generation and merge after tagging')
    parser.add_argument('--test', '-t', action='store_true',
                        help='Test mode: only process first N tags for paraphrasing')
    parser.add_argument('--test-count', '-n', type=int, default=1,
                        help='Number of tags to process in test mode (default: 1)')
    
    args = parser.parse_args()
    
    processor = QueryTagProcessor()
    queries_tags_path, tags_answers_path = processor.process(
        csv_path=args.input_csv,
        output_dir=args.output_dir,
        query_column=args.query_column,
        answer_column=args.answer_column,
        id_column=args.id_column,
        run_paraphrase=args.generate_paraphrases,
        test_mode=args.test,
        test_tag_count=args.test_count
    )
    
    print(f"\nOutput files:")
    print(f"  - {queries_tags_path}")
    print(f"  - {tags_answers_path}")
    
    if args.generate_paraphrases:
        print(f"\n📁 Paraphrased output available in: {args.output_dir}/paraphrased_output/")


if __name__ == "__main__":
    main()
