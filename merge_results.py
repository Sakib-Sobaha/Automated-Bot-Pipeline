#!/usr/bin/env python3
"""
Merge all individual CSV files into a single merged result file.
"""

import csv
import os
from pathlib import Path
from datetime import datetime

# Default Configuration (can be overridden by run_merge)
INPUT_DIR = "paraphrased_output_additional/individual_tags"
# Output to root directory with today's date in filename
today = datetime.now().strftime("%Y-%m-%d")
OUTPUT_FILE = f"new_dataset_{today}.csv"

def get_sorted_csv_files(directory):
    """Get all CSV files sorted by filename (alphabetically by tag name)."""
    csv_files = []
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            filepath = os.path.join(directory, filename)
            csv_files.append(filepath)

    # Sort alphabetically by filename (tag name)
    csv_files.sort()
    return csv_files

def read_individual_csv(filepath):
    """Read a single CSV file and return its data rows (should be 10 rows)."""
    rows = []
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows

def merge_csv_files(csv_files, output_file):
    """Merge all CSV files into a single output file."""
    merged_data = []

    print(f"Merging {len(csv_files)} CSV files...")

    for i, filepath in enumerate(csv_files, 1):
        rows = read_individual_csv(filepath)
        if rows:
            # Each file contains 10 rows (10 questions for the same tag)
            merged_data.extend(rows)
            if i % 50 == 0:
                print(f"  Processed {i}/{len(csv_files)} files...")
        else:
            print(f"WARNING: No data found in {filepath}")

    # Write merged data to output file
    print(f"\nWriting merged data to {output_file}...")
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        if merged_data:
            # Format: question,tag (no answer column)
            fieldnames = ['question', 'tag']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(merged_data)

    print(f"✓ Successfully merged {len(merged_data)} rows into {output_file}")
    return len(merged_data)

def validate_merged_file(output_file, expected_count):
    """Validate the merged file."""
    print("\nValidating merged file...")

    # Count rows
    with open(output_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        actual_count = len(rows)

    print(f"  Total rows: {actual_count}")
    print(f"  Expected: {expected_count} rows (each file has 10 questions)")

    if actual_count == expected_count:
        print("  ✓ Row count matches!")
    else:
        print(f"  ✗ Row count mismatch! Difference: {expected_count - actual_count} rows")

    # Check tag counts (each tag should appear exactly 10 times)
    tags = [row['tag'] for row in rows]
    unique_tags = set(tags)
    from collections import Counter
    tag_counts = Counter(tags)

    print(f"  Unique tags: {len(unique_tags)}")

    # Check if each tag appears exactly 10 times
    tags_with_wrong_count = [tag for tag, count in tag_counts.items() if count != 10]
    if not tags_with_wrong_count:
        print("  ✓ Each tag appears exactly 10 times (10 questions per tag)!")
    else:
        print(f"  ✗ {len(tags_with_wrong_count)} tags don't have exactly 10 questions")
        for tag in tags_with_wrong_count[:5]:  # Show first 5
            print(f"      - {tag}: {tag_counts[tag]} questions")

    # Check for field correctness (question and tag should be populated)
    invalid_count = 0
    for row in rows:
        if not row['question'] or not row['tag']:
            invalid_count += 1

    if invalid_count == 0:
        print("  ✓ All rows have correct format (question and tag populated)!")
    else:
        print(f"  ✗ Found {invalid_count} rows with empty fields")

    print("\nValidation complete!")

def run_merge(input_dir: str, output_file: str = None) -> str:
    """
    Run the merge process with custom file paths.
    
    Args:
        input_dir: Directory containing individual tag CSV files
        output_file: Path for the merged output file. If None, uses default naming.
        
    Returns:
        Path to the merged output file
    """
    if output_file is None:
        today = datetime.now().strftime("%Y-%m-%d")
        output_file = os.path.join(os.path.dirname(input_dir), f"merged_dataset_{today}.csv")
    
    print("="*80)
    print("Starting merge process")
    print("="*80)

    # Get all CSV files
    csv_files = get_sorted_csv_files(input_dir)

    if not csv_files:
        print(f"ERROR: No CSV files found in {input_dir}")
        return None

    print(f"Found {len(csv_files)} CSV files to merge")

    # Merge files
    merged_count = merge_csv_files(csv_files, output_file)

    # Validate
    validate_merged_file(output_file, merged_count)

    print("\n" + "="*80)
    print("Merge complete!")
    print("="*80)
    
    return output_file


def main():
    """Main merge function (for standalone execution)."""
    run_merge(INPUT_DIR, OUTPUT_FILE)


if __name__ == "__main__":
    main()
