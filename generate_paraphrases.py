#!/usr/bin/env python3
"""
Generate paraphrased answers for election commission tags using OpenAI GPT-5.
Each tag's result is saved immediately to prevent data loss.
"""

import csv
import os
import sys
import time
import random
from pathlib import Path
from collections import defaultdict
from openai import OpenAI

# Default Configuration (can be overridden by run_paraphrase_generation)
EXAMPLES_FILE = "ec_missing_tag_question.csv"  # Source for example questions and unique tags
ANSWERS_FILE = "ec_missing_tag_answer.csv"      # Lookup for tag->answer mapping
OUTPUT_DIR = "paraphrased_output_missing/individual_tags"
LOG_FILE = "paraphrased_output_missing/processing_log.txt"
PROGRESS_FILE = "paraphrased_output_missing/progress.txt"
# Test mode: Only process first N tags
TEST_MODE = len(sys.argv) > 1 and sys.argv[1] == "--test"
TEST_TAG_COUNT = 1

# Global OpenAI client (initialized lazily)
client = None

def init_openai_client():
    """Initialize OpenAI client if not already initialized."""
    global client
    if client is None:
        try:
            client = OpenAI()
            if not os.environ.get("OPENAI_API_KEY"):
                print("ERROR: OPENAI_API_KEY environment variable is not set!")
                print("Please set your OpenAI API key:")
                print("  export OPENAI_API_KEY='your-api-key-here'")
                return False
        except Exception as e:
            print(f"ERROR: Failed to initialize OpenAI client: {e}")
            return False
    return True

def get_unique_tags_from_questions(questions_file):
    """Get all unique tags from question_tag.csv, sorted alphabetically (case-insensitive)."""
    unique_tags = set()
    with open(questions_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            tag = row['tag'].strip()
            if tag and tag != 'fraction':
                unique_tags.add(tag)
    return sorted(list(unique_tags), key=str.lower)

def load_answers_lookup(answers_file):
    """Load tag->answer mapping from tag_answer.csv."""
    tag_to_answer = {}
    with open(answers_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            tag_to_answer[row['tag']] = row['answer']
    return tag_to_answer

def load_example_questions(examples_file):
    """
    Load example questions from question_tag.csv and build a tag->questions mapping.
    Filters out 'fraction' tag and invalid entries.
    """
    tag_questions = defaultdict(list)
    excluded_tags = {'fraction'}

    with open(examples_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Filter out invalid entries
            if row['tag'] in excluded_tags:
                continue
            if not row['question'].strip():
                continue

            # Store valid question
            tag_questions[row['tag']].append(row['question'].strip())

    return dict(tag_questions)

def get_random_examples(tag_questions, tag_name, num_examples=30):
    """
    Get 10 random example questions for a given tag.
    Returns empty list if tag has no examples.
    """
    if tag_name not in tag_questions:
        return []

    available = tag_questions[tag_name]

    # Return all if we have fewer than requested
    if len(available) <= num_examples:
        return available

    # Otherwise, randomly select num_examples
    return random.sample(available, num_examples)

def get_last_processed_index():
    """Get the index of the last successfully processed tag."""
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'r') as f:
            content = f.read().strip()
            if content:
                return int(content)
    return -1

def update_progress(index):
    """Update the progress file with the current index."""
    with open(PROGRESS_FILE, 'w') as f:
        f.write(str(index))

def log_message(message):
    """Log a message to the log file and print to console."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] {message}"
    print(log_entry)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(log_entry + '\n')

def generate_questions_with_gpt5(original_answer, tag_name, example_questions, max_retries=3):
    """
    Use OpenAI GPT-5 to generate 200 different questions in Bengali that would lead to this answer.
    Uses real example questions from the dataset to guide the style and tone.
    Includes retry logic for API failures.
    """
    # Build examples section - emphasize question patterns
    examples_text = ""
    if example_questions:
        examples_text = "Here are 10 real example questions from users asking about this topic:\n\n"
        for i, example in enumerate(example_questions, 1):
            examples_text += f"Example {i}: {example}\n"
        examples_text += "\n"

    # Build answer context section (de-emphasized)
    answer_context = f"\nNote: All these example questions lead to the following answer (provided for context only):\n{original_answer}\n"

    prompt = f"""You are creating training data for a Bengali question-answering system about Election Commission services of Bangladesh.

{examples_text}CRITICAL INSTRUCTIONS:

1. **PRIMARY FOCUS**: Analyze the question patterns, structure, phrasing, and lexical choices in the 5 example questions above. These examples are your PRIMARY reference for style and semantic meaning.

2. **SEMANTIC SIMILARITY**: Generate 200 NEW questions that have 80-99% semantic similarity to the example questions. This means:
   - Questions should ask about the same core topic/intent
   - Questions should be contextually equivalent (would all lead to the same answer)
   - But use different words, phrasing, sentence structures, and question formats

3. **DIVERSITY REQUIREMENTS**: Ensure high diversity across the 200 questions through:
   - **Lexical variation**: Use different synonyms, alternative Bengali terms, regional variations
   - **Structural variation**: Different question formats (direct questions, indirect questions, statements with question markers)
   - **Stylistic variation**: Vary sentence length, complexity, and phrasing patterns
   - **Register variation**: Use MIXED registers with EMPHASIS on formal Bengali (approximately 60-70% formal, 30-200% informal/conversational)
   - Avoid using the same words or phrases repeatedly across questions
   - Create human-like natural-sounding questions, avoiding robotic or formulaic patterns

4. **QUALITY STANDARDS**:
   - Questions should be standard Bengali (বাংলা), suitable for a wide audience in Bangladesh
   - All questions must be natural and authentic-sounding
   - All questions must be semantically equivalent (80-99% similarity range)
   - All questions must lead to the same answer
   - Avoid repetition - each question should be distinct

5. **OUTPUT FORMAT**: Output exactly 200 questions, one per line. Number them 1-200. Do not include any other text.

{answer_context}

Generate 200 diverse questions now:"""

    for attempt in range(max_retries):
        try:
            response = client.responses.create(
                model="gpt-5",
                input=prompt
            )

            questions_text = response.output_text.strip()

            # Basic validation: check if response is not empty
            if not questions_text:
                log_message(f"WARNING: Empty response for tag '{tag_name}', attempt {attempt + 1}")
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                else:
                    return None

            # Split into questions - handle numbered and unnumbered formats
            questions = []
            for line in questions_text.split('\n'):
                line = line.strip()
                if not line:
                    continue
                # Remove leading numbers and dots (e.g., "1. ", "1)", etc.)
                line = line.lstrip('0123456789').lstrip('. )-')
                line = line.strip()
                if line:
                    questions.append(line)

            # Ensure we have exactly 200 questions
            if len(questions) < 200:
                log_message(f"WARNING: Got {len(questions)} questions instead of 200 for tag '{tag_name}', attempt {attempt + 1}")
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                else:
                    return None

            # Return first 200 questions
            return questions[:200]

        except Exception as e:
            log_message(f"ERROR: API call failed for tag '{tag_name}', attempt {attempt + 1}: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(5)  # Wait before retry
            else:
                return None

    return None

def save_individual_csv(index, tag_name, generated_questions):
    """Save the result for a single tag to its own CSV file with 200 rows."""
    # Use tag name as filename instead of sequential numbering
    filename = f"{tag_name}.csv"
    filepath = os.path.join(OUTPUT_DIR, filename)

    with open(filepath, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        # Format: question,tag (no answer column)
        writer.writerow(['question', 'tag'])
        # Write 200 rows with generated questions and tag
        for question in generated_questions:
            writer.writerow([question, tag_name])

    log_message(f"Saved: {filename} (200 questions generated)")

def run_paraphrase_generation(examples_file: str, answers_file: str, output_dir: str, 
                               test_mode: bool = False, test_tag_count: int = 1):
    """
    Run the paraphrase generation pipeline with custom file paths.
    
    Args:
        examples_file: Path to CSV with queries and tags (query, tag columns)
        answers_file: Path to CSV with tags and answers (tag, answer columns)
        output_dir: Directory to save individual tag CSV files
        test_mode: If True, only process first N tags
        test_tag_count: Number of tags to process in test mode
        
    Returns:
        Tuple of (success_count, failure_count, skipped_count)
    """
    global EXAMPLES_FILE, ANSWERS_FILE, OUTPUT_DIR, LOG_FILE, PROGRESS_FILE
    
    # Update global configuration
    EXAMPLES_FILE = examples_file
    ANSWERS_FILE = answers_file
    OUTPUT_DIR = os.path.join(output_dir, "individual_tags")
    LOG_FILE = os.path.join(output_dir, "processing_log.txt")
    PROGRESS_FILE = os.path.join(output_dir, "progress.txt")
    
    # Initialize OpenAI client
    if not init_openai_client():
        return (0, 0, 0)
    
    # Ensure output directories exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

    log_message("="*80)
    if test_mode:
        log_message(f"Starting question generation process [TEST MODE - {test_tag_count} tags]")
    else:
        log_message("Starting question generation process [FULL MODE]")
    log_message("="*80)

    # Load example questions from question_tag.csv
    log_message(f"Loading example questions from {EXAMPLES_FILE}...")
    tag_questions = load_example_questions(EXAMPLES_FILE)
    log_message(f"Loaded examples for {len(tag_questions)} unique tags")

    # Get unique tags from question_tag.csv (source of truth for which tags to process)
    unique_tags = get_unique_tags_from_questions(EXAMPLES_FILE)
    log_message(f"Found {len(unique_tags)} unique tags to process")

    # Load answers lookup from tag_answer.csv
    log_message(f"Loading answers from {ANSWERS_FILE}...")
    tag_to_answer = load_answers_lookup(ANSWERS_FILE)
    log_message(f"Loaded answers for {len(tag_to_answer)} tags")

    # Limit tags in test mode
    if test_mode:
        unique_tags = unique_tags[:test_tag_count]
        log_message(f"Test mode: Processing only {len(unique_tags)} tags")
    else:
        log_message(f"Processing {len(unique_tags)} tags")

    total_tags = len(unique_tags)

    # Check for previous progress
    last_processed = get_last_processed_index()
    start_index = last_processed + 1

    # Reset progress if the saved index is beyond the available tags (e.g., when input CSV shrinks)
    if start_index >= total_tags:
        log_message(f"Progress file shows index {start_index}, but only {total_tags} tags exist. Resetting progress to 0.")
        start_index = 0
        update_progress(-1)

    if start_index > 0:
        log_message(f"Resuming from tag index {start_index} (already processed: {start_index})")

    # Process each tag
    success_count = 0
    failure_count = 0
    skipped_count = 0
    start_time = time.time()

    for i in range(start_index, total_tags):
        tag_name = unique_tags[i]
        
        # Calculate progress
        progress_pct = ((i + 1) / total_tags) * 100
        elapsed_time = time.time() - start_time
        if i > start_index:
            avg_time_per_tag = elapsed_time / (i - start_index)
            remaining_tags = total_tags - (i + 1)
            estimated_remaining = avg_time_per_tag * remaining_tags
            eta_minutes = estimated_remaining / 60
        else:
            eta_minutes = 0
        
        # Get answer for this tag
        if tag_name not in tag_to_answer:
            log_message(f"\n[{i+1}/{total_tags}] ({progress_pct:.1f}%) SKIPPED: No answer found for tag '{tag_name}'")
            skipped_count += 1
            update_progress(i)
            continue
            
        original_answer = tag_to_answer[tag_name]

        log_message(f"\n{'='*80}")
        log_message(f"[{i+1}/{total_tags}] ({progress_pct:.1f}%) Processing: {tag_name}")
        if eta_minutes > 0:
            log_message(f"Progress: ✓ {success_count} success | ✗ {failure_count} failed | ⊘ {skipped_count} skipped")
            log_message(f"Estimated time remaining: {eta_minutes:.1f} minutes")

        # Get random example questions for this tag
        examples = get_random_examples(tag_questions, tag_name, num_examples=30)

        if not examples:
            log_message(f"SKIPPED: No example questions found for tag '{tag_name}'")
            skipped_count += 1
            update_progress(i)
            continue

        log_message(f"Using {len(examples)} example question(s) as reference")
        log_message(f"Generating 200 questions...")

        # Generate 200 questions from the answer using GPT-5 with examples
        questions = generate_questions_with_gpt5(original_answer, tag_name, examples)

        if questions and len(questions) == 200:
            # Save immediately to prevent data loss
            save_individual_csv(i, tag_name, questions)
            update_progress(i)
            success_count += 1
            log_message(f"✓ SUCCESS: Generated and saved 200 questions")

            # Small delay to avoid rate limiting
            time.sleep(1)
        else:
            log_message(f"✗ FAILED: Could not generate questions for tag '{tag_name}' after retries")
            failure_count += 1
            update_progress(i)
            # Continue processing other tags even if one fails

    # Final summary
    log_message("\n" + "="*80)
    log_message("Processing complete!")
    log_message(f"Total tags attempted: {total_tags}")
    log_message(f"Successfully generated: {success_count}")
    log_message(f"Skipped (no examples): {skipped_count}")
    log_message(f"Failed (API errors): {failure_count}")
    log_message("="*80)
    
    return (success_count, failure_count, skipped_count)


def main():
    """Main processing function (for standalone execution)."""
    # Initialize client at startup for standalone mode
    if not init_openai_client():
        sys.exit(1)
    
    run_paraphrase_generation(
        examples_file=EXAMPLES_FILE,
        answers_file=ANSWERS_FILE,
        output_dir=os.path.dirname(OUTPUT_DIR),
        test_mode=TEST_MODE,
        test_tag_count=TEST_TAG_COUNT
    )


if __name__ == "__main__":
    main()
