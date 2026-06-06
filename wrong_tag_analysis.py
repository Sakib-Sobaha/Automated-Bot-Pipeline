#!/usr/bin/env python3
"""
Wrong Tag Analysis: Analyze prediction accuracy per tag from evaluation results.
Includes functionality to remove mismatched queries from train/test datasets.
"""

import csv
import os
from collections import defaultdict
from typing import Dict, List, Tuple, Set
from datetime import datetime


class TagAnalyzer:
    """Analyzes tag prediction accuracy from evaluation CSV files."""
    
    def __init__(self):
        self.tag_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: {'right': 0, 'wrong': 0})
        # Mismatch mappings: (expected_tag, predicted_tag) -> count
        self.mismatch_mappings: Dict[Tuple[str, str], int] = defaultdict(int)
    
    def load_evaluation_csv(self, csv_path: str, 
                            expected_col: str = 'expected_tag',
                            predicted_col: str = 'predicted_tag') -> None:
        """
        Load evaluation results from CSV and compute per-tag statistics.
        
        Args:
            csv_path: Path to the evaluation CSV file.
            expected_col: Name of the expected tag column.
            predicted_col: Name of the predicted tag column.
        """
        self.tag_stats = defaultdict(lambda: {'right': 0, 'wrong': 0})
        self.mismatch_mappings = defaultdict(int)
        
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                expected_tag = row[expected_col].strip()
                predicted_tag = row[predicted_col].strip()
                
                if expected_tag == predicted_tag:
                    self.tag_stats[expected_tag]['right'] += 1
                else:
                    self.tag_stats[expected_tag]['wrong'] += 1
                    self.mismatch_mappings[(expected_tag, predicted_tag)] += 1
        
        print(f"✓ Loaded {sum(s['right'] + s['wrong'] for s in self.tag_stats.values())} predictions")
        print(f"  - Unique tags: {len(self.tag_stats)}")
    
    def print_tag_analysis(self, 
                           sort_by_count: int = 0,
                           sort_by_accuracy: int = 0,
                           sort_by_name: int = 0,
                           show_top_n: int = None,
                           descending: bool = True,
                           output_dir: str = ".") -> str:
        """
        Print analysis of right/wrong predictions per tag and save to log file.
        
        Args:
            sort_by_count: If 1, sort by total count (right + wrong)
            sort_by_accuracy: If 1, sort by accuracy percentage (highest to lowest or lowest to highest)
            sort_by_name: If 1, sort alphabetically by tag name
            show_top_n: If set, only show top N tags (None = show all)
            descending: If True, sort highest to lowest; if False, sort lowest to highest
            output_dir: Directory to save the log file
            
        Returns:
            Path to the generated log file
        """
        if not self.tag_stats:
            print("No data loaded. Call load_evaluation_csv() first.")
            return None
        
        # Build list of (tag, right, wrong, total, accuracy)
        analysis_data: List[Tuple[str, int, int, int, float]] = []
        
        for tag, stats in self.tag_stats.items():
            right = stats['right']
            wrong = stats['wrong']
            total = right + wrong
            accuracy = (right / total * 100) if total > 0 else 0.0
            analysis_data.append((tag, right, wrong, total, accuracy))
        
        # Determine sort key
        if sort_by_accuracy == 1:
            # Sort by accuracy: descending = highest to lowest, ascending = lowest to highest
            analysis_data.sort(key=lambda x: x[4], reverse=descending)
            sort_label = "Accuracy" + (" (highest to lowest)" if descending else " (lowest to highest)")
        elif sort_by_count == 1:
            analysis_data.sort(key=lambda x: x[3], reverse=descending)
            sort_label = "Count" + (" (highest to lowest)" if descending else " (lowest to highest)")
        elif sort_by_name == 1:
            analysis_data.sort(key=lambda x: x[0].lower(), reverse=not descending)
            sort_label = "Name" + (" (A-Z)" if not descending else " (Z-A)")
        else:
            # Default: sort by total count descending
            analysis_data.sort(key=lambda x: x[3], reverse=True)
            sort_label = "Count (highest to lowest) [default]"
        
        # Limit to top N if specified
        if show_top_n:
            analysis_data = analysis_data[:show_top_n]
        
        # Calculate totals
        total_right = sum(s['right'] for s in self.tag_stats.values())
        total_wrong = sum(s['wrong'] for s in self.tag_stats.values())
        total_all = total_right + total_wrong
        overall_accuracy = (total_right / total_all * 100) if total_all > 0 else 0.0
        
        # Create output directory if needed
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate log filename with date
        today = datetime.now().strftime("%Y-%m-%d")
        log_filename = f"report_log_{today}.log"
        log_path = os.path.join(output_dir, log_filename)
        
        # Collect output lines (for both console and file)
        output_lines = []
        
        output_lines.append("")
        output_lines.append("╔" + "═" * 96 + "╗")
        output_lines.append("║" + "  📊 TAG PREDICTION ANALYSIS".center(96) + "║")
        output_lines.append("╠" + "═" * 96 + "╣")
        output_lines.append(f"║  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}".ljust(97) + "║")
        output_lines.append(f"║  Sorted by: {sort_label}".ljust(97) + "║")
        output_lines.append(f"║  Total predictions: {total_all} | Right: {total_right} | Wrong: {total_wrong} | Overall Accuracy: {overall_accuracy:.2f}%".ljust(97) + "║")
        output_lines.append("╠" + "═" * 96 + "╣")
        
        # Column headers
        output_lines.append("║ {:^4} │ {:^45} │ {:^8} │ {:^8} │ {:^8} │ {:^9} ║".format(
            "#", "Tag Name", "Right", "Wrong", "Total", "Accuracy"))
        output_lines.append("╠" + "═" * 96 + "╣")
        
        # Each row
        for i, (tag, right, wrong, total, accuracy) in enumerate(analysis_data, 1):
            # Truncate long tag names
            display_tag = tag[:43] + ".." if len(tag) > 45 else tag
            
            # Indicator based on accuracy
            if accuracy >= 90:
                indicator = "✓"
            elif accuracy >= 70:
                indicator = "~"
            else:
                indicator = "✗"
            
            output_lines.append("║ {:>4} │ {:45} │ {:>8} │ {:>8} │ {:>8} │ {:>7.2f}% {} ║".format(
                i, display_tag, right, wrong, total, accuracy, indicator))
        
        output_lines.append("╚" + "═" * 96 + "╝")
        output_lines.append("")
        output_lines.append(f"Legend: ✓ = ≥90% | ~ = 70-89% | ✗ = <70%")
        if show_top_n:
            output_lines.append(f"Showing top {show_top_n} of {len(self.tag_stats)} tags")
        output_lines.append("")
        
        # Print to console
        for line in output_lines:
            print(line)
        
        # Write to log file
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(output_lines))
        
        print(f"📄 Log saved to: {log_path}")
        
        # Append mismatch mapping section to the same log file
        self.log_mismatch_mappings(log_path)
        
        return log_path
    
    def log_mismatch_mappings(self, log_path: str) -> None:
        """
        Append to the given log file a section showing wrong-tag mappings:
        for each mismatch, expected_tag -> predicted_tag and count.
        Shows how many times each expected tag was mapped to each wrong predicted tag.
        """
        if not self.mismatch_mappings:
            return
        
        # Build list of (expected_tag, predicted_tag, count), sorted by count descending then by expected_tag
        rows: List[Tuple[str, str, int]] = [
            (exp, pred, count) for (exp, pred), count in self.mismatch_mappings.items()
        ]
        rows.sort(key=lambda x: (-x[2], x[0], x[1]))
        
        lines = []
        lines.append("")
        lines.append("╔" + "═" * 96 + "╗")
        lines.append("║" + "  🔀 MISMATCH MAPPINGS (expected_tag → predicted_tag, count)".center(96) + "║")
        lines.append("╠" + "═" * 96 + "╣")
        lines.append("║ {:^4} │ {:^40} │ {:^40} │ {:^6} ║".format("#", "Expected Tag", "Predicted Tag (wrong)", "Count"))
        lines.append("╠" + "═" * 96 + "╣")
        
        for i, (exp, pred, count) in enumerate(rows, 1):
            display_exp = (exp[:38] + "..") if len(exp) > 40 else exp
            display_pred = (pred[:38] + "..") if len(pred) > 40 else pred
            lines.append("║ {:>4} │ {:40} │ {:40} │ {:>6} ║".format(i, display_exp, display_pred, count))
        
        lines.append("╚" + "═" * 96 + "╝")
        lines.append("")
        
        # Append to existing log file
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write("\n".join(lines))
        
        # Also print to console
        for line in lines:
            print(line)
    
    def get_worst_tags(self, n: int = 10) -> List[Tuple[str, int, int, float]]:
        """
        Get the N tags with lowest accuracy.
        
        Returns:
            List of (tag, right, wrong, accuracy) tuples
        """
        results = []
        for tag, stats in self.tag_stats.items():
            right = stats['right']
            wrong = stats['wrong']
            total = right + wrong
            accuracy = (right / total * 100) if total > 0 else 0.0
            results.append((tag, right, wrong, accuracy))
        
        results.sort(key=lambda x: x[3])
        return results[:n]
    
    def get_best_tags(self, n: int = 10) -> List[Tuple[str, int, int, float]]:
        """
        Get the N tags with highest accuracy.
        
        Returns:
            List of (tag, right, wrong, accuracy) tuples
        """
        results = []
        for tag, stats in self.tag_stats.items():
            right = stats['right']
            wrong = stats['wrong']
            total = right + wrong
            accuracy = (right / total * 100) if total > 0 else 0.0
            results.append((tag, right, wrong, accuracy))
        
        results.sort(key=lambda x: x[3], reverse=True)
        return results[:n]
    
    def remove_mismatched_queries(self,
                                   evaluation_csv: str,
                                   test_csv: str,
                                   train_csv: str,
                                   output_dir: str = ".",
                                   question_col: str = 'question',
                                   similar_question_col: str = 'similar question',
                                   expected_tag_col: str = 'expected tag',
                                   predicted_tag_col: str = 'predicted tag') -> Dict[str, Dict[str, int]]:
        """
        Remove mismatched queries from test and train datasets.
        Only removes queries where expected_tag != predicted_tag (actual mismatches).
        
        For each mismatch:
        - Remove the 'question' from test dataset (sts_eval_updated.csv)
        - Remove the 'similar question' from train dataset (question_tag_answer.csv)
        
        Args:
            evaluation_csv: Path to evaluation CSV (can be full or mismatches-only)
            test_csv: Path to test dataset CSV (question, tag)
            train_csv: Path to train dataset CSV (question, tag, answer)
            output_dir: Directory to save output files
            question_col: Column name for test question in evaluation
            similar_question_col: Column name for train question in evaluation
            expected_tag_col: Column name for expected tag in evaluation
            predicted_tag_col: Column name for predicted tag in evaluation
            
        Returns:
            Dict with removal counts per tag for both train and test
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Track removal counts per tag
        removal_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: {'test': 0, 'train': 0})
        
        # Collect queries to remove from evaluation CSV (only mismatches)
        test_queries_to_remove: Set[str] = set()
        train_queries_to_remove: Set[str] = set()
        mismatch_data: List[Dict] = []
        total_rows = 0
        matched_rows = 0
        
        print(f"\n📂 Loading evaluation data from: {evaluation_csv}")
        with open(evaluation_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                total_rows += 1
                expected_tag = row[expected_tag_col].strip()
                predicted_tag = row[predicted_tag_col].strip()
                
                # Only process mismatches (where expected != predicted)
                if expected_tag == predicted_tag:
                    matched_rows += 1
                    continue  # Skip matched predictions
                
                test_query = row[question_col].strip()
                train_query = row[similar_question_col].strip()
                
                test_queries_to_remove.add(test_query)
                train_queries_to_remove.add(train_query)
                mismatch_data.append({
                    'test_query': test_query,
                    'train_query': train_query,
                    'expected_tag': expected_tag,
                    'predicted_tag': predicted_tag
                })
        
        print(f"   Total rows in evaluation: {total_rows}")
        print(f"   Matched (kept): {matched_rows}")
        print(f"   Mismatched (to remove): {len(mismatch_data)}")
        print(f"   Unique test queries to remove: {len(test_queries_to_remove)}")
        print(f"   Unique train queries to remove: {len(train_queries_to_remove)}")
        
        # Build a mapping of query -> expected_tag for accurate stats tracking
        test_query_to_tag: Dict[str, str] = {}
        train_query_to_tag: Dict[str, str] = {}
        for item in mismatch_data:
            test_query_to_tag[item['test_query']] = item['expected_tag']
            train_query_to_tag[item['train_query']] = item['expected_tag']
        
        # Process TEST dataset
        print(f"\n📝 Processing test dataset: {test_csv}")
        test_kept = []
        test_removed = []
        
        with open(test_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            test_fieldnames = reader.fieldnames
            
            for row in reader:
                query = row['question'].strip()
                
                if query in test_queries_to_remove:
                    test_removed.append(row)
                    # Use the expected tag from mismatch data for accurate tracking
                    expected_tag = test_query_to_tag.get(query, row['tag'].strip())
                    removal_stats[expected_tag]['test'] += 1
                else:
                    test_kept.append(row)
        
        # Write updated test CSV
        updated_test_path = test_csv  # Overwrite original
        with open(updated_test_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=test_fieldnames)
            writer.writeheader()
            writer.writerows(test_kept)
        
        # Write removed test queries
        removed_test_path = os.path.join(output_dir, "removed_test.csv")
        with open(removed_test_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=test_fieldnames)
            writer.writeheader()
            writer.writerows(test_removed)
        
        print(f"   ✓ Kept: {len(test_kept)} | Removed: {len(test_removed)}")
        print(f"   ✓ Updated: {updated_test_path}")
        print(f"   ✓ Removed queries saved to: {removed_test_path}")
        
        # Process TRAIN dataset
        print(f"\n📝 Processing train dataset: {train_csv}")
        train_kept = []
        train_removed = []
        
        with open(train_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            train_fieldnames = reader.fieldnames
            
            for row in reader:
                query = row['question'].strip()
                
                if query in train_queries_to_remove:
                    train_removed.append(row)
                    # Use the expected tag from mismatch data for accurate tracking
                    expected_tag = train_query_to_tag.get(query, row['tag'].strip())
                    removal_stats[expected_tag]['train'] += 1
                else:
                    train_kept.append(row)
        
        # Write updated train CSV
        updated_train_path = train_csv  # Overwrite original
        with open(updated_train_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=train_fieldnames)
            writer.writeheader()
            writer.writerows(train_kept)
        
        # Write removed train queries
        removed_train_path = os.path.join(output_dir, "removed_train.csv")
        with open(removed_train_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=train_fieldnames)
            writer.writeheader()
            writer.writerows(train_removed)
        
        print(f"   ✓ Kept: {len(train_kept)} | Removed: {len(train_removed)}")
        print(f"   ✓ Updated: {updated_train_path}")
        print(f"   ✓ Removed queries saved to: {removed_train_path}")
        
        # Generate report.log
        report_path = os.path.join(output_dir, "report.log")
        self._generate_removal_report(removal_stats, report_path, 
                                       len(test_removed), len(train_removed),
                                       len(test_kept), len(train_kept))
        
        print(f"\n📊 Report generated: {report_path}")
        
        return dict(removal_stats)
    
    def _generate_removal_report(self, 
                                  removal_stats: Dict[str, Dict[str, int]],
                                  report_path: str,
                                  total_test_removed: int,
                                  total_train_removed: int,
                                  total_test_kept: int,
                                  total_train_kept: int) -> None:
        """Generate a detailed report of removed queries per tag."""
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Sort by total removed (test + train) descending
        sorted_stats = sorted(
            removal_stats.items(),
            key=lambda x: x[1]['test'] + x[1]['train'],
            reverse=True
        )
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("MISMATCH QUERY REMOVAL REPORT\n")
            f.write(f"Generated: {timestamp}\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("SUMMARY\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total test queries removed:  {total_test_removed}\n")
            f.write(f"Total train queries removed: {total_train_removed}\n")
            f.write(f"Test queries remaining:      {total_test_kept}\n")
            f.write(f"Train queries remaining:     {total_train_kept}\n")
            f.write(f"Unique tags affected:        {len(removal_stats)}\n")
            f.write("\n")
            
            f.write("REMOVAL COUNTS BY TAG\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'#':<5} {'Tag Name':<50} {'Test':<10} {'Train':<10} {'Total':<10}\n")
            f.write("-" * 80 + "\n")
            
            for i, (tag, counts) in enumerate(sorted_stats, 1):
                test_count = counts['test']
                train_count = counts['train']
                total = test_count + train_count
                
                # Truncate long tag names
                display_tag = tag[:48] + ".." if len(tag) > 50 else tag
                
                f.write(f"{i:<5} {display_tag:<50} {test_count:<10} {train_count:<10} {total:<10}\n")
            
            f.write("-" * 80 + "\n")
            f.write(f"{'TOTAL':<56} {total_test_removed:<10} {total_train_removed:<10} {total_test_removed + total_train_removed:<10}\n")
            f.write("=" * 80 + "\n")
        
        # Also print summary to console
        print("\n" + "=" * 70)
        print("📋 REMOVAL SUMMARY")
        print("=" * 70)
        print(f"  Test queries removed:  {total_test_removed}")
        print(f"  Train queries removed: {total_train_removed}")
        print(f"  Tags affected:         {len(removal_stats)}")
        print("=" * 70)


def main():
    """Example usage of TagAnalyzer."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Analyze tag prediction accuracy and remove mismatched queries.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze predictions
  python wrong_tag_analysis.py ec_full_evaluation_threshold_0.923.csv -a 1

  # Remove mismatched queries from datasets (uses full evaluation CSV)
  # Only removes queries where expected_tag != predicted_tag
  python wrong_tag_analysis.py ec_full_evaluation_threshold_0.923.csv \\
      --remove-mismatches \\
      --test-csv sts_eval_updated.csv \\
      --train-csv question_tag_answer.csv \\
      --output-dir cleanup_output
        """
    )
    parser.add_argument('csv_file', help='Path to evaluation CSV file (with expected/predicted tags)')
    
    # Analysis options
    parser.add_argument('--sort-count', '-c', type=int, default=0, choices=[0, 1],
                        help='Sort by total count (0=off, 1=on)')
    parser.add_argument('--sort-accuracy', '-a', type=int, default=0, choices=[0, 1],
                        help='Sort by accuracy (0=off, 1=on)')
    parser.add_argument('--sort-name', '-n', type=int, default=0, choices=[0, 1],
                        help='Sort by tag name (0=off, 1=on)')
    parser.add_argument('--top', '-t', type=int, default=None,
                        help='Show only top N tags')
    parser.add_argument('--ascending', action='store_true',
                        help='Sort in ascending order (default is descending)')
    
    # Removal options
    parser.add_argument('--remove-mismatches', '-r', action='store_true',
                        help='Remove mismatched queries from test and train datasets')
    parser.add_argument('--test-csv', type=str, default='sts_eval_updated.csv',
                        help='Path to test dataset CSV (default: sts_eval_updated.csv)')
    parser.add_argument('--train-csv', type=str, default='question_tag_answer.csv',
                        help='Path to train dataset CSV (default: question_tag_answer.csv)')
    parser.add_argument('--output-dir', '-o', type=str, default='.',
                        help='Output directory for removed queries and report (default: current dir)')
    
    args = parser.parse_args()
    
    analyzer = TagAnalyzer()
    
    if args.remove_mismatches:
        # Remove mismatched queries mode
        print("\n" + "=" * 70)
        print("🧹 MISMATCH QUERY REMOVAL MODE")
        print("=" * 70)
        
        analyzer.remove_mismatched_queries(
            evaluation_csv=args.csv_file,
            test_csv=args.test_csv,
            train_csv=args.train_csv,
            output_dir=args.output_dir
        )
    else:
        # Analysis mode
        analyzer.load_evaluation_csv(args.csv_file)
        analyzer.print_tag_analysis(
            sort_by_count=args.sort_count,
            sort_by_accuracy=args.sort_accuracy,
            sort_by_name=args.sort_name,
            show_top_n=args.top,
            descending=not args.ascending,
            output_dir=args.output_dir
        )


if __name__ == "__main__":
    main()

