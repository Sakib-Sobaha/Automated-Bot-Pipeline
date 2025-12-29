#!/usr/bin/env python3
"""
Wrong Tag Analysis: Analyze prediction accuracy per tag from evaluation results.
"""

import csv
from collections import defaultdict
from typing import Dict, List, Tuple


class TagAnalyzer:
    """Analyzes tag prediction accuracy from evaluation CSV files."""
    
    def __init__(self):
        self.tag_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: {'right': 0, 'wrong': 0})
    
    def load_evaluation_csv(self, csv_path: str, 
                            expected_col: str = 'expected tag',
                            predicted_col: str = 'predicted tag') -> None:
        """
        Load evaluation results from CSV and compute per-tag statistics.
        
        Args:
            csv_path: Path to the evaluation CSV file.
            expected_col: Name of the expected tag column.
            predicted_col: Name of the predicted tag column.
        """
        self.tag_stats = defaultdict(lambda: {'right': 0, 'wrong': 0})
        
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                expected_tag = row[expected_col].strip()
                predicted_tag = row[predicted_col].strip()
                
                if expected_tag == predicted_tag:
                    self.tag_stats[expected_tag]['right'] += 1
                else:
                    self.tag_stats[expected_tag]['wrong'] += 1
        
        print(f"✓ Loaded {sum(s['right'] + s['wrong'] for s in self.tag_stats.values())} predictions")
        print(f"  - Unique tags: {len(self.tag_stats)}")
    
    def print_tag_analysis(self, 
                           sort_by_count: int = 0,
                           sort_by_accuracy: int = 0,
                           sort_by_name: int = 0,
                           show_top_n: int = None,
                           descending: bool = True) -> None:
        """
        Print analysis of right/wrong predictions per tag.
        
        Args:
            sort_by_count: If 1, sort by total count (right + wrong)
            sort_by_accuracy: If 1, sort by accuracy percentage
            sort_by_name: If 1, sort alphabetically by tag name
            show_top_n: If set, only show top N tags (None = show all)
            descending: If True, sort in descending order (except for name)
        """
        if not self.tag_stats:
            print("No data loaded. Call load_evaluation_csv() first.")
            return
        
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
            analysis_data.sort(key=lambda x: x[4], reverse=descending)
            sort_label = "Accuracy" + (" (desc)" if descending else " (asc)")
        elif sort_by_count == 1:
            analysis_data.sort(key=lambda x: x[3], reverse=descending)
            sort_label = "Count" + (" (desc)" if descending else " (asc)")
        elif sort_by_name == 1:
            analysis_data.sort(key=lambda x: x[0].lower(), reverse=not descending)
            sort_label = "Name" + (" (A-Z)" if not descending else " (Z-A)")
        else:
            # Default: sort by total count descending
            analysis_data.sort(key=lambda x: x[3], reverse=True)
            sort_label = "Count (desc) [default]"
        
        # Limit to top N if specified
        if show_top_n:
            analysis_data = analysis_data[:show_top_n]
        
        # Calculate totals
        total_right = sum(s['right'] for s in self.tag_stats.values())
        total_wrong = sum(s['wrong'] for s in self.tag_stats.values())
        total_all = total_right + total_wrong
        overall_accuracy = (total_right / total_all * 100) if total_all > 0 else 0.0
        
        # Print header
        print()
        print("╔" + "═" * 96 + "╗")
        print("║" + "  📊 TAG PREDICTION ANALYSIS".center(96) + "║")
        print("╠" + "═" * 96 + "╣")
        print(f"║  Sorted by: {sort_label}".ljust(97) + "║")
        print(f"║  Total predictions: {total_all} | Right: {total_right} | Wrong: {total_wrong} | Overall Accuracy: {overall_accuracy:.2f}%".ljust(97) + "║")
        print("╠" + "═" * 96 + "╣")
        
        # Column headers
        print("║ {:^4} │ {:^45} │ {:^8} │ {:^8} │ {:^8} │ {:^9} ║".format(
            "#", "Tag Name", "Right", "Wrong", "Total", "Accuracy"))
        print("╠" + "═" * 96 + "╣")
        
        # Print each row
        for i, (tag, right, wrong, total, accuracy) in enumerate(analysis_data, 1):
            # Truncate long tag names
            display_tag = tag[:43] + ".." if len(tag) > 45 else tag
            
            # Color indicator based on accuracy
            if accuracy >= 90:
                indicator = "✓"
            elif accuracy >= 70:
                indicator = "~"
            else:
                indicator = "✗"
            
            print("║ {:>4} │ {:45} │ {:>8} │ {:>8} │ {:>8} │ {:>7.2f}% {} ║".format(
                i, display_tag, right, wrong, total, accuracy, indicator))
        
        print("╚" + "═" * 96 + "╝")
        print()
        print(f"Legend: ✓ = ≥90% | ~ = 70-89% | ✗ = <70%")
        if show_top_n:
            print(f"Showing top {show_top_n} of {len(self.tag_stats)} tags")
        print()
    
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


def main():
    """Example usage of TagAnalyzer."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze tag prediction accuracy.')
    parser.add_argument('csv_file', help='Path to evaluation CSV file')
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
    
    args = parser.parse_args()
    
    analyzer = TagAnalyzer()
    analyzer.load_evaluation_csv(args.csv_file)
    analyzer.print_tag_analysis(
        sort_by_count=args.sort_count,
        sort_by_accuracy=args.sort_accuracy,
        sort_by_name=args.sort_name,
        show_top_n=args.top,
        descending=not args.ascending
    )


if __name__ == "__main__":
    main()

