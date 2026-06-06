import csv
import os
import sys
from pathlib import Path 
from typing import Set, List, Dict 

def read_mismatch_csv(csv_path1: str, csv_path2: str) -> Set[str]:
    """
    Read the first CSV and extract similar_question values from mismatch rows.

    Args:
        csv_path: Path to CSV created after evaluation with question, similar_question, expected_tag, predicted_tag columns

    Returns:
        Set of similar_question values from rows where expected_tag != predicted_tag
    """
    similar_questions_to_be_removed = set()
    original_questions = []
    if not os.path.exists(csv_path1 or csv_path2):
        raise FileNotFoundError(f"CSV file not found: {csv_path1} or {csv_path2}")

    with open(csv_path1, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)

        required_cols = ['question', 'similar_question','expected_tag', 'predicted_tag']
        if not all(col in reader.fieldnames for col in required_cols):
            raise ValueError(
                f"CSV must contain columns: {required_cols}"
                f"Found: {reader.fieldnames}"
            )
        
        mismatch_count = 0
        for row in reader:
            expected_tag = row['expected_tag'].strip()
            predicted_tag = row['predicted_tag'].strip()

            if expected_tag != predicted_tag:
                similar_question = row['similar_question'].strip()
                if similar_question:
                    similar_questions_to_be_removed.add(similar_question)
                    mismatch_count += 1
        
        print(f"Found {mismatch_count} mismatch rows")
        print(f"Extracted {len(similar_questions_to_be_removed)} unique similar_question values to remove")

    with open(csv_path2, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)

        required_cols = ['question', 'tag']
        if not all(col in reader.fieldnames for col in required_cols):
            raise ValueError(
                f"CSV must contain columns: {required_cols}"
                f"Found: {reader.fieldnames}"
            )
        for row in reader:
            original_question = row['question'].strip()
            original_questions.append(original_question)
    
    for query in similar_questions_to_be_removed.copy():
        if query in original_questions:
            similar_questions_to_be_removed.remove(query)
    
    return similar_questions_to_be_removed

def remove_queries_from_csv(csv_path: str, queries_to_remove: Set[str], output_path: str) -> None:
    """
    Remove rows from CSV where question matches any query in queries_to_remove.
    When comparing, if either question has a question mark, both are compared without the question mark.

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
        reader = csv.DictReader(f);

        required_cols = ['question', 'tag']
        if not all(col in reader.fieldnames for col in required_cols):
            raise ValueError(
                f"CSV must contains columns: {required_cols}"
                f"Found: {reader.fieldnames}"
            )
        
        fieldnames = reader.fieldnames 

        for row in reader:
            question = row['question'].strip()
            
            # Check if question should be removed
            should_remove = False
            
            # Direct exact match first
            if question in queries_to_remove:
                should_remove = True
            else:
                # Check each query in the removal set
                for query in queries_to_remove:
                    # If either question has a question mark, compare without question marks
                    if '?' in question or '?' in query:
                        normalized_question = question.rstrip('?').strip()
                        normalized_query = query.rstrip('?').strip()
                        if normalized_question == normalized_query:
                            should_remove = True
                            break
            
            if should_remove:
                rows_removed += 1
            else:
                rows_kept.append(row)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_kept)

    print(f"Removed {rows_removed} rows from the dataset")
    print(f"Kept {len(rows_kept)} rows")
    print(f"Saved cleaned CSV to: {output_path}")

def main():
    """ Main function to orchestrate the query removal process"""

    input_dir = Path("Dataset_Cleanup_Input")
    output_dir = Path("Dataset_Cleanup_Output")

    output_dir.mkdir(exist_ok=True)

    if not input_dir.exists():
        print(f"Error: Input directory '{input_dir}' does not exist.")
        print("Please create it and place your CSV files there.")
        sys.exit(1)

    mismatch_csv = input_dir/"ec_original_queries_threshold_0.57.csv"
    original_queries_csv = input_dir/"original_queries.csv"

    dataset_csv = input_dir/"question_tag.csv"

    if len(sys.argv) >= 3:
        mismatch_csv = input_dir / sys.argv[1]
        dataset_csv = input_dir / sys.argv[2]
    elif len(sys.argv) == 2:
        print("Usage: python query_removal.py [mismatch_csv_filename] [dataset_csv_filename]")
        print("Or palce files as 'mismatch_evaluation.csv' and 'question_tag.csv' in Dataset_Cleanu_Input/" )
        sys.exit(1)
    
    dataset_name = Path(dataset_csv).stem 
    output_csv = output_dir/f"{dataset_name}_cleaned.csv"

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
        similar_questions_to_be_removed = read_mismatch_csv(mismatch_csv, original_queries_csv)

        if not similar_questions_to_be_removed:
            print("\n No similar question to remove. Output CSV wull be identical to input.")

        

        removed_questions_path = output_dir/"removed_questions.txt"
        with open(removed_questions_path, "w") as f:
            for query in similar_questions_to_be_removed:
                f.write(query + "\n")
        # print(similar_questions_to_be_removed)
        remove_queries_from_csv(
            str(dataset_csv),
            similar_questions_to_be_removed,
            str(output_csv)
        )

        print("-"*60)
        print("Process completed successfully!")
    except FileNotFoundError as e:
        print(f"\n Error: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"\n Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()