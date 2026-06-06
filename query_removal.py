#!/usr/bin/env python3
"""
Query Removal Script: Removes similar questions from a dataset based on mismatched predictions.

This script:
1. Reads a CSV with question, similar_question, expected_tag, predicted_tag columns
2. Filters rows where expected_tag != predicted_tag (mismatches)
3. Reads another CSV with question, tag columns
4. Removes rows from the second CSV where question matches any similar_question from mismatches
5. Outputs the cleaned CSV to Dataset_Cleanup_Output folder
"""

import csv
import os
import sys
from pathlib import Path
from typing import Set, List, Dict


def read_mismatch_csv(csv_path: str) -> Set[str]:
    """
    Read the first CSV and extract similar_question values from mismatch rows.
    
    Args:
        csv_path: Path to CSV with question, similar_question, expected_tag, predicted_tag columns
        
    Returns:
        Set of similar_question values from rows where expected_tag != predicted_tag
    """
    similar_questions = set()
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        # Verify required columns exist
        required_cols = ['question', 'similar_question', 'expected_tag', 'predicted_tag']
        if not all(col in reader.fieldnames for col in required_cols):
            raise ValueError(
                f"CSV must contain columns: {required_cols}. "
                f"Found: {reader.fieldnames}"
            )
        
        mismatch_count = 0
        for row in reader:
            expected_tag = row['expected_tag'].strip()
            predicted_tag = row['predicted_tag'].strip()
            
            # Check for mismatch
            if expected_tag != predicted_tag:
                similar_question = row['similar_question'].strip()
                if similar_question:  # Only add non-empty similar questions
                    similar_questions.add(similar_question)
                    mismatch_count += 1
        
        print(f"✓ Found {mismatch_count} mismatch rows")
        print(f"✓ Extracted {len(similar_questions)} unique similar_question values to remove")
    
    return similar_questions


def remove_queries_from_csv(csv_path: str, queries_to_remove: Set[str], output_path: str) -> None:
    """
    Remove rows from CSV where question matches any query in queries_to_remove.
    
    Args:
        csv_path: Path to CSV with question, tag columns
        queries_to_remove: Set of question strings to remove
        output_path: Path to save the cleaned CSV
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    rows_kept = []
    rows_removed = 0
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        # Verify required columns exist
        required_cols = ['question', 'tag']
        if not all(col in reader.fieldnames for col in required_cols):
            raise ValueError(
                f"CSV must contain columns: {required_cols}. "
                f"Found: {reader.fieldnames}"
            )
        
        fieldnames = reader.fieldnames
        
        for row in reader:
            question = row['question'].strip()
            
            # Check if this question should be removed
            if question in queries_to_remove:
                rows_removed += 1
            else:
                rows_kept.append(row)
    
    # Write the cleaned CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_kept)
    
    print(f"✓ Removed {rows_removed} rows from the dataset")
    print(f"✓ Kept {len(rows_kept)} rows")
    print(f"✓ Saved cleaned CSV to: {output_path}")


def main():
    """Main function to orchestrate the query removal process."""
    
    # Define input and output directories
    input_dir = Path("Dataset_Cleanup_Input")
    output_dir = Path("Dataset_Cleanup_Output")
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(exist_ok=True)
    
    # Check if input directory exists
    if not input_dir.exists():
        print(f"Error: Input directory '{input_dir}' does not exist.")
        print("Please create it and place your CSV files there.")
        sys.exit(1)
    
    # Get CSV file paths (you can modify these filenames as needed)
    # CSV 1: Contains question, similar_question, expected_tag, predicted_tag
    mismatch_csv = input_dir / "mismatch_evaluation.csv"
    
    # CSV 2: Contains question, tag
    dataset_csv = input_dir / "dataset.csv"
    
    # Allow command-line arguments for file names
    if len(sys.argv) >= 3:
        mismatch_csv = input_dir / sys.argv[1]
        dataset_csv = input_dir / sys.argv[2]
    elif len(sys.argv) == 2:
        print("Usage: python query_removal.py [mismatch_csv_filename] [dataset_csv_filename]")
        print("Or place files as 'mismatch_evaluation.csv' and 'dataset.csv' in Dataset_Cleanup_Input/")
        sys.exit(1)
    
    # Generate output filename
    dataset_name = Path(dataset_csv).stem
    output_csv = output_dir / f"{dataset_name}_cleaned.csv"
    
    print("=" * 60)
    print("Query Removal Script")
    print("=" * 60)
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"\nReading mismatch CSV: {mismatch_csv}")
    print(f"Reading dataset CSV: {dataset_csv}")
    print(f"Output CSV: {output_csv}")
    print("-" * 60)
    
    try:
        # Step 1: Read mismatch CSV and extract similar_questions
        similar_questions = read_mismatch_csv(str(mismatch_csv))
        
        if not similar_questions:
            print("\n⚠ No similar questions to remove. Output CSV will be identical to input.")
        
        # Step 2: Remove matching queries from dataset CSV
        remove_queries_from_csv(
            str(dataset_csv),
            similar_questions,
            str(output_csv)
        )
        
        print("-" * 60)
        print("✓ Process completed successfully!")
        
    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"\n✗ Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
