import csv
import os
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional 
from openai import OpenAI


class QueryTagProcessor:
    """
    Processes a CSV file containing queries, answers, and group IDs.
    Generates meaningful tags for each group using OpenAI and splits data into separate CSVs.
    """

    def __init__(self):
        """Initialize the processor."""
        self.data: List[Dict[str, str]] = []  # List of {query, answer, id} dicts
        self.id_to_queries: Dict[str, List[str]] = defaultdict(list)  # id -> list of queries
        self.id_to_answer: Dict[str, str] = {}
        self.id_to_tag: Dict[str, str] = {}
        self.query_tags: Dict[str, str] = {}
        self.tag_answer: Dict[str, str] = {}


        # Initialize OpenAI client for tag generation
        try:
            self.client = OpenAI()
            if not os.environ.get("OPENAI_API_KEY"):
                print("ERROR: OPENAI_API_KEY environment variable is not set!")
                print("Please set your OpenAI API key:")
                raise RuntimeError("OPENAI_API_KEY not set")
        except Exception as e:
            print(f"ERROR: Failed to initialize OpenAI client: {e}")
            raise
    
    def load_csv(self, csv_path: str, query_column: str = 'query', answer_column: str = 'answer', id_column: str = 'id') -> None:
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

            print(f"Available columns: {reader.fieldnames}")

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
                    if group_id not in self.id_to_answer:
                        self.id_to_answer[group_id] = answer 
        unique_ids = len(self.id_to_queries)
        print(f"Loaded {len(self.data)} query-answer pairs from {csv_path}")
        print(f"Found {unique_ids} unique group IDs")


def main():
    parser = argparse.ArgumentParser(description='Automated Query Preprocessing for Bot')
    parser.add_argument('input_csv', help='Path to input CSV file with queries, answers, and IDs')
    parser.add_argument('--output-dir', '-o', default='.', help='Output directory for CSV files')
    parser.add_argument('--query-column', '-q', default='query', help='Name of query column')
    parser.add_argument('--answer-column', '--a', default='answer', help='Name of answer column')
    parser.add_argument('--id-column', '-i', default='id', help='Name of ID column (groups similar queries)')

    print("Available Arguments:")
    parser.print_help()

    args = parser.parse_args()

    print(f"Processing input CSV: {args.input_csv}")
    print(f"Output directory: {args.output_dir}")
    print(f"Query column: {args.query_column}")
    print(f"Answer column: {args.answer_column}")
    print(f"ID column: {args.id_column}")

    # Load input CSV
    input_csv = args.input_csv
    output_dir = args.output_dir
    query_column = args.query_column

if __name__ == "__main__":
    main()