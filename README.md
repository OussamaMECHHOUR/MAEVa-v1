# MAEVa: Matching Agroecological Experimental Variables

**Description:** This repository provides the full pipeline used for our initial approach, called MAEVa, to match experimental agroecological variables using hybrid similarity techniques that combine name embeddings and description vectors. With this code, you can reproduce the BERT experiments described in our article and easily adapt the approach to your own task if you're working on a similar problem.

**Abstract:**
Source variables or observable properties used to describe agroecological experiments are heterogeneous, nonstandardized, and multilingual, which makes them challenging to understand, explain, and use in cropping system modeling and multicriteria evaluations of agroecological system performance. Data annotation via a controlled vocabulary, known as candidate variables from the agroecological global information system (AEGIS), offers a solution. Text similarity measures play crucial roles in tasks such as word sense disambiguation, schema matching in databases, and data annotation. Commonly used measures include 1) string-based similarity, 2) corpus-based similarity, 3) knowledge-based similarity, and 4) hybrid-based similarity, which combine two or more of these measures. This work presents a hybrid approach called Matching Agroecological Experiment Variables (MAEVa), designed to align source and candidate variables through the following components: (1) our key innovation, which consists of extending pretrained language models (PLMs), such as BERT, with an external multi-head attention layer for matching variable names; (2) an analysis of the relevance and impact of various context enrichment techniques (e.g., snippet extraction, data generation, and scientific articles) on corpus-based similarity measures for matching variable descriptions; (3) a linear combination of components (1) and (2); and (4) a voting-based method for selecting the final matching results. Experimental results demonstrate that extending PLMs with an external multi-head attention layer improves the matching of variable names. Furthermore, corpus-based methods benefit consistently from the presence of an enriched corpus, regardless of the specific enrichment technique employed. While the combination strategy itself is not novel, it yields statistically significant improvements, with performance gains exceeding 5\%.

---

This repository contains two scenarios:

1. **Scenario 1: MAEVa Published in the NLP Journal.** The existing pipeline corresponds to the BERT experiments published in the *Natural Language Processing Journal*. It performs no task-specific training: BERT uses two pretrained hidden layers with fixed weights, and the additional attention layer is randomly initialized with the selected seed and used without learning. With only 84 manually annotated pairs available initially, training the additional parameters would risk overfitting and leave little independent data for evaluation. This motivated the original setting without supervised learning.
2. **Scenario 2: MAEVa with Learning.** Developed several months after the published version, this extension learns the attention weights while keeping BERT fixed and introduces new training and held-out test benchmarks for a broader evaluation. The scripts and required files are located in `extension/`. This later extension is separate from the experiments reported in the published article.

### ⚠️ Hardware Configuration for Reproducibility

**Important:** For reproducibility, execute all evaluation experiments on CPU (`--device cpu` for Scenario 2) and perform only Scenario 2 training on GPU (`--device cuda`). Scenario 1 runs on CPU by default and does not require a device argument.

## 🚀 Scenario 1 (MAEVa Published in the NLP Journal): Complete Pipeline Usage Guide

This script performs the full MAEVa pipeline for matching experimental agroecological variables (Scenario 1).

### 🔍 What it does

1. **Preprocesses variable names, descriptions and the corpus used**  
   - For **variable names**, three functions are applied in sequence:
     1. `clean_text`: removes digits, parentheses, and special characters
     2. `remove_stopwords`: filters out common English stopwords
     3. `lemmatize`: reduces words to their base form using WordNet
   - For **descriptions and context corpus**, five preprocessing steps are applied:
     1. `clean_text`: same as above
     2. `remove_stopwords`: same as above
     3. `lemmatize`: same as above
     4. `remove_punctuation`: strips punctuation marks
     5. `replace_synonyms`: replaces words with their first synonym from WordNet if available

   All input variable files and the reference matching file are located in the `datasets/benchmarks/` folder.

2. **Generates embeddings for variable names**  
   Variable names are often acronyms and lack sufficient context, making them challenging to interpret using PLMs. Our intuition is that PLMs are capable of effectively embedding such variable names. For this reason, our innovation consists of extending existing PLMs with an external multi-head attention layer applied to their frozen embeddings. We compare the performance of this extended architecture with that of the original PLMs to evaluate their effectiveness in representing variable names. We use BERT for this purpose:
   - [`BERT`](https://huggingface.co/bert-base-uncased): we used the "bert-base-uncased" checkpoint with 2 hidden layers

3. **Applies MultiHeadAttention (MHA)**  
   - Enabled by default for refining name embeddings  
   - Can be disabled using the `--no_attention` flag

4. **Generates TF-IDF vectors for descriptions**  
   The objective is to analyze the impact of various context enrichment techniques (e.g., snippet extraction, data generation, and scientific articles) on TF–IDF, for matching variable descriptions. All corpora used are stored in the `datasets/corpora/` directory.

5. **Combines similarities**  
   We combine the similarity scores obtained from name embeddings and TF-IDF vectors through a linear weighted sum:
   - `final_score = 0.75 * sim_name + 0.25 * sim_description`

6. **Evaluates matching accuracy**  
   - For each source variable, the code calculates similarity scores with all candidates
   - Ranks the top-10 candidate matches
   - Computes Precision@K for K ∈ {1, 3, 5, 10}, based on whether the ground-truth match appears in the top-K list
   - Saves:
     - Precision summaries (number of correct matches and percentages)
     - A list of top-10 candidate matches for each source variable
     - A global precision score that marks a source variable as correctly matched if **any of the three methods** (names, descriptions, or combination) matched it correctly, which is the final result of MAEVa

   All these results are stored in the `outputs` folder.
   
---

## ⚙️ Available Arguments

| Argument               | Description                                                                 | Default                                      |
|------------------------|------------------------------------------------------------------------------|----------------------------------------------|
| `--base_path`          | Directory where input `.txt` and `.xlsx` files are located                   | `datasets/benchmarks`                        |
| `--src_names`          | Filename for source variable names                                           | `source_variable(names).txt`                 |
| `--cand_names`         | Filename for candidate variable names                                        | `candidate_variable(names).txt`              |
| `--src_desc`           | Filename for source variable descriptions                                    | `source_variable(descriptions).txt`          |
| `--cand_desc`          | Filename for candidate variable descriptions                                 | `candidate_variable(descriptions).txt`       |
| `--context`            | Full path to the context corpus used for TF-IDF (independent of base_path)  | `datasets/corpora/Corpus (GPT-prompt 1).txt` |
| `--reference_file`     | Excel file mapping source variables to their correct candidates              | `Correspondances.xlsx`                       |
| `--model`              | Embedding model for names: `bert`                                            | `bert`                                       |
| `--seed`               | Random seed for reproducibility                                              | `1751`                                       |
| `--num_heads`          | Number of attention heads in MultiHeadAttention                              | `256`                                        |
| `--no_attention`       | Add this flag to disable MultiHeadAttention                                  | *(enabled by default)*                      |
| `--output_path`        | Folder where result files (.xlsx) will be saved                              | `outputs`                                    |

---

## 🧪 Command Line Usage
Our code uses default values that yielded the best results, but these can be modified as described below.
Before executing any command, make sure that all requirements are installed to ensure the reproducibility of our results by running the following command: 

```bash
pip install -r requirements.txt
```

MAEVa pipeline can be executed in two ways: with default parameters and with custom options.

### ▶️ With default parameters:

```bash
python "scripts/maeva_pipeline.py"
```

This will execute the full pipeline with:

| Parameter           | Default Value                                       | Description                                      |
|---------------------|-----------------------------------------------------|--------------------------------------------------|
| `--base_path`       | `datasets/benchmarks`                               | Folder where all input `.txt` and `.xlsx` files are located |
| `--src_names`       | `source_variable(names).txt`                        | File of source variable names                    |
| `--cand_names`      | `candidate_variable(names).txt`                     | File of candidate variable names                 |
| `--src_desc`        | `source_variable(descriptions).txt`                | File of source variable descriptions             |
| `--cand_desc`       | `candidate_variable(descriptions).txt`             | File of candidate variable descriptions          |
| `--context`         | `datasets/corpora/Corpus (GPT-prompt 1).txt`       | File path to external corpus for TF-IDF          |
| `--reference_file`  | `Correspondances.xlsx`                              | Excel file with ground-truth correspondences     |
| `--model`           | `bert`                                              | Name embedding model: `bert`                     |
| `--seed`            | `1751`                                              | Random seed for reproducibility                  |
| `--num_heads`       | `256`                                               | Number of heads in MultiHeadAttention            |
| `--output_path`     | `outputs`                                           | Output folder for result `.xlsx` files           |
| `MultiHeadAttention`| ✅ enabled (default behavior)                       | Can be disabled using `--no_attention` flag      |

---

### ⚙️ With custom options:

#### ✅ With Multihead attention: 
```bash
python "scripts/maeva_pipeline.py" \
  --base_path "datasets/benchmarks" \
  --src_names "source_variable(names).txt" \
  --cand_names "candidate_variable(names).txt" \
  --src_desc "source_variable(descriptions).txt" \
  --cand_desc "candidate_variable(descriptions).txt" \
  --context "datasets/corpora/Corpus (GPT-prompt 1).txt" \
  --reference_file "Correspondances.xlsx" \
  --model "bert" \
  --seed 1751 \
  --num_heads 256 \
  --output_path "outputs" 
```

#### 🚫 Without Multihead attention:
```bash
python "scripts/maeva_pipeline.py" \
  --base_path "datasets/benchmarks" \
  --src_names "source_variable(names).txt" \
  --cand_names "candidate_variable(names).txt" \
  --src_desc "source_variable(descriptions).txt" \
  --cand_desc "candidate_variable(descriptions).txt" \
  --context "datasets/corpora/Corpus (GPT-prompt 1).txt" \
  --reference_file "Correspondances.xlsx" \
  --model "bert" \
  --seed 1751 \
  --num_heads 256 \
  --output_path "outputs" \
  --no_attention
```

This custom commands allows full control over:
- Which files to use
- Which model to apply
- Whether to use attention or not
- Where to save the results

---

## 📦 Output Files

All evaluation files are saved as `.xlsx` inside the folder specified by `--output_path`. The following files are generated:


### ✅ With Multihead attention: 
| Filename                                     | Description                                       |
|----------------------------------------------|---------------------------------------------------|
| `matching_names_{model}({seed}).xlsx`        | Top-10 matches and precision using name embeddings |
| `matching_descriptions_{model}({seed}).xlsx` | Top-10 matches and precision using TF-IDF         |
| `combination_method_{model}({seed}).xlsx`    | Combined scores (TF-IDF + names)                  |
| `final_results_{model}({seed}).xlsx`         | Global Precision@K evaluation                     |

### 🚫 Without Multihead attention:

| Filename                                     | Description                                       |
|----------------------------------------------|---------------------------------------------------|
| `matching_names_{model}_noatt({seed}).xlsx`        | Top-10 matches and precision using name embeddings |
| `matching_descriptions_{model}_noatt({seed}).xlsx` | Top-10 matches and precision using TF-IDF         |
| `combination_method_{model}_noatt({seed}).xlsx`    | Combined scores (TF-IDF + names)                  |
| `final_results_{model}_noatt({seed}).xlsx`         | Global Precision@K evaluation                     |

---

## 🚀 Scenario 2 (MAEVa with Learning): Complete Pipeline Usage Guide
This script performs the evaluation of MAEVa after supervised training of its attention layer (Scenario 2).

### 🔍 What it does

1. **Loads the trained model**  
   - Uses the saved two-layer BERT encoder, tokenizer and learned 256-head attention from `extension/maeva_run/`
   - Loads the fitted TF-IDF vectorizer and precomputed candidate pools
   - Performs no additional training during testing

2. **Applies the name and description preprocessing**  
   - Retains the preprocessing operations used in Scenario 1
   - Computes name and description similarities and their combination: `0.75 * sim_name + 0.25 * sim_description`

3. **Evaluates matching accuracy**  
   - Compares each source with the complete candidate pool for its target domain: DSSAT, STICS or AEGIS
   - Uses separate pools for names, descriptions and exact name–description pairs
   - Computes Precision@K for K ∈ {1, 3, 5, 10}, using the number of annotated test pairs as the denominator
   - Saves per-benchmark precision, macro averages and top-10 candidates for each pair

   The training and test Excel files, scripts, dependencies and context corpus are located in `extension/`. The saved model, tokenizer, attention, TF-IDF and candidate pools are in `extension/maeva_run/`.

---

## ⚙️ Available Arguments (Scenario 2)

| Argument | Description | Default |
|----------|-------------|---------|
| `--checkpoint` | Directory containing the saved model, tokenizer, attention and TF-IDF | Required |
| `--pool-dir` | Directory containing the candidate pool JSON files and their manifest | Required |
| `--test` | One or more test Excel files, separated by spaces | Required |
| `--output-dir` | Folder where evaluation files will be saved | `maeva_test_results` |
| `--batch-size` | Batch size for encoding variable names | `32` |
| `--device` | Execution device: `auto`, `cpu` or `cuda` | `auto` |
| `--no-attention` | Disable the additional attention using the same two-layer BERT checkpoint | Attention enabled by default |

---

## 🧪 Command Line Usage (Scenario 2)

Run these commands from the repository root. Use a separate environment because the dependency versions differ from Scenario 1:

```bash
python3 -m venv .venv-extension
source .venv-extension/bin/activate
pip install -r extension/requirements.txt
python -m nltk.downloader stopwords wordnet omw-1.4
```

The supplied package list is retained in `extension/installed_environment.txt`.

### ▶️ With the supplied trained model:

```bash
PYTHONHASHSEED=1751 python "extension/test_maeva.py" \
  --checkpoint "extension/maeva_run" \
  --pool-dir "extension/maeva_run/pools" \
  --test \
    "extension/ACME-DSSAT (ground truth)_test.xlsx" \
    "extension/ACME-STICS (ground truth)_test.xlsx" \
    "extension/ICASA-DSSAT (ground truth)_test.xlsx" \
    "extension/IntercropValues-STICS (ground truth)_test.xlsx" \
    "extension/SourceAEGIS-CandidateAEGIS (ground truth)_test.xlsx" \
  --output-dir "outputs_trained" \
  --batch-size 32 \
  --device auto
```

To test one benchmark, pass only its Excel path to `--test`. Select `--device cpu` or `--device cuda` if needed. Add `--no-attention` to disable the additional attention.

### ⚙️ Training:

The supplied training script is also included. It keeps BERT fixed and optimizes the additional attention. Test correspondence labels are excluded from training and validation; only the target columns of the test files are used when constructing candidate pools.

```bash
PYTHONHASHSEED=1751 python "extension/train_maeva.py" \
  --data-dir "extension" \
  --train "*_train*.xlsx" \
  --pool-test "*_test*.xlsx" \
  --context "extension/Corpus (GPT-prompt 1).txt" \
  --output-dir "extension/maeva_run" \
  --seed 1751 \
  --device auto
```

Use `python "extension/train_maeva.py" --help` for the training arguments. To test a newly trained model, set `--checkpoint` to its output folder and `--pool-dir` to the `pools/` subfolder.

---

## 📦 Output Files (Scenario 2)

Evaluation files are generated inside the folder specified by `--output-dir`:

| Filename | Description |
|----------|-------------|
| `precision_by_benchmark.csv` | Precision@K for each benchmark and method |
| `macro_average.csv` | Unweighted mean of the benchmark scores for each method |
| `rankings_top10_each_pair.csv` | Top-10 candidates, scores and ground-truth rank for each annotated pair |
| `run_manifest.json` | Run configuration and input/model checksums |

Precision values in the CSV files range from 0 to 1. The macro average gives each benchmark equal weight and differs from the any-method global score of Scenario 1.

For a GitHub update, install Git LFS and run `git lfs install` before committing the bundled model; its tracking rule is included in `.gitattributes`.

---

## 📬 Contact

If you have any questions, encounter issues with the code, or would like to know more about our work, please contact the corresponding author:

- 📧 **Personal email (permanent)**: [oussama.mechhour.cirad@gmail.com](mailto:oussama.mechhour.cirad@gmail.com)  
- 📧 **Professional email (not sure if permanent)**: [oussama.mechhour@cirad.fr](mailto:oussama.mechhour@cirad.fr)

[![LinkedIn](https://img.shields.io/badge/LinkedIn-blue?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/oussama-mechhour-ph-d-student-31a94323a/)
[![ORCID](https://img.shields.io/badge/ORCID-0000--0002--1825--0097-a6ce39?logo=orcid&logoColor=white)](https://orcid.org/0009-0004-3007-1229)
[![HAL](https://img.shields.io/badge/HAL-Archive-orange?logo=data&logoColor=white)](https://hal.science/search/index/?q=oussama+mechhour)
