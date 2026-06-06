# Automated Bot Pipeline

A comprehensive pipeline for processing queries, generating tags, creating paraphrased training data, and analyzing prediction accuracy for a Bengali question-answering system.

---

## 📁 Project Structure

```
Automated-Bot-Pipeline/
├── query_tag_processor.py      # Main pipeline: tag generation + paraphrase orchestration
├── generate_paraphrases.py     # Generate 200 paraphrased questions per tag using GPT
├── merge_results.py            # Merge individual tag CSVs into single dataset
├── wrong_tag_analysis.py       # Analyze prediction accuracy per tag
└── README.md
```

---

## 🚀 Quick Start

### Prerequisites

```bash
# Set your OpenAI API key
export OPENAI_API_KEY='your-api-key-here'

# Verify it's set
echo $OPENAI_API_KEY
```

---

## 📋 1. Query Tag Processor (`query_tag_processor.py`)

Processes a CSV with queries, answers, and group IDs. Generates meaningful tags for each group using OpenAI.

### Input CSV Format

| query | answer | id |
|-------|--------|-----|
| How to vote? | You can vote by... | 1 |
| Voting process? | You can vote by... | 1 |
| Get NID card | Apply at... | 2 |

> **Note:** Rows with the same `id` are considered similar queries and will receive the same tag.

### Commands

**Generate tags only:**
```bash
python query_tag_processor.py input.csv --output-dir output
```

**Full pipeline (tags + paraphrases + merge):**
```bash
python query_tag_processor.py input.csv --output-dir output --generate-paraphrases
```

**Test mode (process only 2 tags for testing):**
```bash
python query_tag_processor.py input.csv -o output -g --test --test-count 2
```

### Options

| Option | Short | Description |
|--------|-------|-------------|
| `--output-dir` | `-o` | Output directory for generated files |
| `--query-column` | `-q` | Name of query column (default: `query`) |
| `--answer-column` | `-a` | Name of answer column (default: `answer`) |
| `--id-column` | `-i` | Name of ID column (default: `id`) |
| `--generate-paraphrases` | `-g` | Run full pipeline including paraphrase generation |
| `--test` | `-t` | Test mode: process limited tags |
| `--test-count` | `-n` | Number of tags in test mode (default: 1) |

### Output Files

- `queries_tags.csv` - Maps queries to generated tags
- `tags_answers.csv` - Maps tags to answers
- `paraphrased_output/` - Directory with generated paraphrases (if `-g` flag used)

---

## 📝 2. Generate Paraphrases (`generate_paraphrases.py`)

Generates 200 paraphrased questions per tag using OpenAI GPT. Supports resume on failure.

### Standalone Usage

```bash
# Full mode
python generate_paraphrases.py

# Test mode (1 tag only)
python generate_paraphrases.py --test
```

### Configuration

Edit the file to change default paths:
```python
EXAMPLES_FILE = "question_tag.csv"    # Input: query, tag columns
ANSWERS_FILE = "tag_answer.csv"       # Input: tag, answer columns
OUTPUT_DIR = "paraphrased_output/individual_tags"
```

---

## 🔗 3. Merge Results (`merge_results.py`)

Merges all individual tag CSV files into a single dataset.

### Standalone Usage

```bash
python merge_results.py
```

### Configuration

Edit the file to change default paths:
```python
INPUT_DIR = "paraphrased_output/individual_tags"
OUTPUT_FILE = "merged_dataset_YYYY-MM-DD.csv"
```

---

## 📊 4. Wrong Tag Analysis (`wrong_tag_analysis.py`)

Analyzes prediction accuracy per tag from evaluation results. Shows right/wrong counts and accuracy percentages. Also supports removing mismatched queries from train/test datasets.

### 4.1 Analysis Mode

Analyze prediction accuracy per tag.

**Basic analysis (sorted by count, descending):**
```bash
python wrong_tag_analysis.py ec_full_evaluation_threshold_0.923.csv
```

**Sort by accuracy (best performing tags first):**
```bash
python wrong_tag_analysis.py ec_full_evaluation_threshold_0.923.csv -a 1
```

**Sort by accuracy ascending (worst performing tags first):**
```bash
python wrong_tag_analysis.py ec_full_evaluation_threshold_0.923.csv -a 1 --ascending
```

**Sort by total count (most predictions first), show top 20:**
```bash
python wrong_tag_analysis.py ec_full_evaluation_threshold_0.923.csv -c 1 --top 20
```

**Sort alphabetically by tag name:**
```bash
python wrong_tag_analysis.py ec_full_evaluation_threshold_0.923.csv -n 1
```

#### Analysis Options

| Option | Short | Values | Description |
|--------|-------|--------|-------------|
| `--sort-count` | `-c` | 0/1 | Sort by total prediction count |
| `--sort-accuracy` | `-a` | 0/1 | Sort by accuracy percentage |
| `--sort-name` | `-n` | 0/1 | Sort alphabetically by tag name |
| `--top` | `-t` | int | Show only top N tags |
| `--ascending` | | flag | Sort in ascending order (default: descending) |

#### Analysis Output Example

```
╔════════════════════════════════════════════════════════════════════════════════════════════════╗
║                                    📊 TAG PREDICTION ANALYSIS                                   ║
╠════════════════════════════════════════════════════════════════════════════════════════════════╣
║  Sorted by: Accuracy (asc)                                                                     ║
║  Total predictions: 11458 | Right: 10191 | Wrong: 1267 | Overall Accuracy: 88.94%              ║
╠════════════════════════════════════════════════════════════════════════════════════════════════╣
║  #   │                    Tag Name                     │   Right  │  Wrong   │  Total   │ Accuracy  ║
╠════════════════════════════════════════════════════════════════════════════════════════════════╣
║    1 │ problematic_tag                                 │       10 │       40 │       50 │   20.00% ✗ ║
║    2 │ another_low_accuracy_tag                        │       30 │       20 │       50 │   60.00% ✗ ║
...
╚════════════════════════════════════════════════════════════════════════════════════════════════╝

Legend: ✓ = ≥90% | ~ = 70-89% | ✗ = <70%
```

---

### 4.2 Mismatch Removal Mode

Remove mismatched queries from both test and train datasets. This is useful for cleaning up datasets after evaluation to improve model accuracy.

#### What it does:
1. Reads the mismatches CSV (queries where predicted ≠ expected)
2. Removes the `question` from test dataset (`sts_eval_updated.csv`)
3. Removes the `similar question` from train dataset (`question_tag_answer.csv`)
4. Saves removed queries to `removed_test.csv` and `removed_train.csv`
5. Generates a detailed `report.log` with per-tag removal counts

#### Commands

**Basic mismatch removal:**
```bash
python wrong_tag_analysis.py ec_full_evaluation_mismatches_threshold_0.923.csv \
    --remove-mismatches \
    --test-csv sts_eval_updated.csv \
    --train-csv question_tag_answer.csv
```

**With custom output directory:**
```bash
python wrong_tag_analysis.py ec_full_evaluation_mismatches_threshold_0.923.csv \
    --remove-mismatches \
    --test-csv sts_eval_updated.csv \
    --train-csv question_tag_answer.csv \
    --output-dir cleanup_output
```

**Short form:**
```bash
python wrong_tag_analysis.py ec_full_evaluation_mismatches_threshold_0.923.csv -r \
    --test-csv sts_eval_updated.csv \
    --train-csv question_tag_answer.csv \
    -o cleanup_output
```

#### Removal Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--remove-mismatches` | `-r` | - | Enable removal mode |
| `--test-csv` | | `sts_eval_updated.csv` | Path to test dataset CSV |
| `--train-csv` | | `question_tag_answer.csv` | Path to train dataset CSV |
| `--output-dir` | `-o` | `.` | Output directory for removed files and report |

#### Output Files

| File | Description |
|------|-------------|
| `removed_test.csv` | Queries removed from test dataset (question, tag) |
| `removed_train.csv` | Queries removed from train dataset (question, tag, answer) |
| `report.log` | Detailed removal counts per tag |

#### Input CSV Formats

**Mismatches CSV** (e.g., `ec_full_evaluation_mismatches_threshold_0.923.csv`):

| question | similar question | expected tag | predicted tag | time taken |
|----------|-----------------|--------------|---------------|------------|
| Test query... | Train query... | expected_tag | wrong_predicted_tag | 0.65 |

**Test CSV** (e.g., `sts_eval_updated.csv`):

| question | tag |
|----------|-----|
| Test query... | tag_name |

**Train CSV** (e.g., `question_tag_answer.csv`):

| question | tag | answer |
|----------|-----|--------|
| Train query... | tag_name | Answer text... |

#### Sample report.log

```
================================================================================
MISMATCH QUERY REMOVAL REPORT
Generated: 2025-12-30 14:30:45
================================================================================

SUMMARY
----------------------------------------
Total test queries removed:  1267
Total train queries removed: 1267
Test queries remaining:      10191
Train queries remaining:     77355
Unique tags affected:        89

REMOVAL COUNTS BY TAG
--------------------------------------------------------------------------------
#     Tag Name                                           Test       Train      Total     
--------------------------------------------------------------------------------
1     how_to_complain_not_getting_help_from_election..   45         45         90        
2     online_new_voter_registration                      38         38         76        
3     card_information_correction                        32         32         64        
4     nid_correction_general                             28         28         56        
5     voter_registration_process                         25         25         50        
...
--------------------------------------------------------------------------------
TOTAL                                                    1267       1267       2534      
================================================================================
```

---

### 4.3 Python API

```python
from wrong_tag_analysis import TagAnalyzer

analyzer = TagAnalyzer()

# --- Analysis Mode ---
analyzer.load_evaluation_csv("evaluation_results.csv")

# Print analysis with different sorting
analyzer.print_tag_analysis(sort_by_accuracy=1, descending=False)  # Worst first
analyzer.print_tag_analysis(sort_by_count=1)                       # Most predictions first
analyzer.print_tag_analysis(sort_by_name=1)                        # Alphabetical

# Get worst/best performing tags
worst_tags = analyzer.get_worst_tags(n=10)
best_tags = analyzer.get_best_tags(n=10)

# --- Mismatch Removal Mode ---
removal_stats = analyzer.remove_mismatched_queries(
    mismatches_csv="ec_full_evaluation_mismatches_threshold_0.923.csv",
    test_csv="sts_eval_updated.csv",
    train_csv="question_tag_answer.csv",
    output_dir="cleanup_output"
)

# removal_stats is a dict: {tag_name: {'test': count, 'train': count}, ...}
for tag, counts in removal_stats.items():
    print(f"{tag}: removed {counts['test']} from test, {counts['train']} from train")
```

---

## 🔄 Full Pipeline Flow

```
┌─────────────────────────────────────────────────────────────┐
│                      INPUT CSV                               │
│              (query, answer, id columns)                     │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              1. QUERY TAG PROCESSOR                          │
│         Generate meaningful tags using OpenAI                │
└─────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
    ┌──────────────────┐           ┌──────────────────┐
    │ queries_tags.csv │           │ tags_answers.csv │
    │  (query, tag)    │           │  (tag, answer)   │
    └──────────────────┘           └──────────────────┘
              │                               │
              └───────────────┬───────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              2. PARAPHRASE GENERATOR                         │
│      Generate 200 questions per tag using GPT                │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
              ┌───────────────────────────────┐
              │   individual_tags/*.csv       │
              │   (200 questions per tag)     │
              └───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    3. MERGE RESULTS                          │
│           Combine all CSVs into single dataset               │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
              ┌───────────────────────────────┐
              │  merged_dataset_YYYY-MM-DD.csv │
              │     (Final training data)      │
              └───────────────────────────────┘
```

---

## 📄 License

See [LICENSE](LICENSE) file for details.

