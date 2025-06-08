# MAEVa: Matching Agroecological Experimental Variables

**Description:** This repository provides the full pipeline used for our initial approach, called MAEVa, to match experimental agroecological variables using hybrid similarity techniques that combine name embeddings and description vectors. With this code, you can reproduce all the experiments described in our article and easily adapt the approach to your own task if you're working on a similar problem.

**Abstract:**
Source variables or observable properties used to describe agroecological experiments are heterogeneous, nonstandardized, and multilingual, which makes them challenging to understand, explain, and use in cropping system modeling and multicriteria evaluations of agroecological system performance. Data annotation via a controlled vocabulary, known as candidate variables from the agroecological global information system (AEGIS), offers a solution. Text similarity measures play crucial roles in tasks such as word sense disambiguation, schema matching in databases, and data annotation. Commonly used measures include 1) string-based similarity, 2) corpus-based similarity, 3) knowledge-based similarity, and 4) hybrid-based similarity, which combine two or more of these measures. This work presents a hybrid approach called Matching Agroecological Experiment Variables (MAEVa), designed to align source and candidate variables through the following components: (1) our key innovation, which consists of extending pretrained language models (PLMs), such as BERT, with an external multi-head attention layer for matching variable names; (2) an analysis of the impact of various context enrichment techniques (e.g., snippet extraction, data generation, and scientific articles) on corpus-based similarity measures for matching variable descriptions; (3) a linear combination of components (1) and (2); and (4) a voting-based method for selecting the final matching results. Experimental results demonstrate that extending PLMs with an external multi-head attention layer improves the matching of variable names. Furthermore, corpus-based methods benefit consistently from the presence of an enriched corpus, regardless of the specific enrichment technique employed. While the combination strategy itself is not novel, it yields statistically significant improvements, with performance gains exceeding 5\%.

---

## 🚀 MAEVa: Complete Pipeline Usage Guide

This script performs the full MAEVa pipeline for matching experimental agroecological variables.

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
   Variable names are often acronyms and lack sufficient context, making them challenging to interpret using PLMs. Our intuition is that PLMs are capable of effectively embedding such variable names. For this reason, our innovation consists of extending existing PLMs with an external multi-head attention layer applied to their frozen embeddings. We compare the performance of this extended architecture with that of the original PLMs to evaluate their effectiveness in representing variable names. We chose three known models for this purpose:
   - [`BERT`](https://huggingface.co/bert-base-uncased): we used the "bert-base-uncased" checkpoint with 2 hidden layers
   - [`SBERT`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2): we applied the "all-MiniLM-L6-v2" checkpoint
   - [`SimCSE`](https://huggingface.co/princeton-nlp/sup-simcse-bert-base-uncased): we applied the "sup-simcse-bert-base-uncased" checkpoint

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
| `--model`              | Embedding model for names: `bert`, `sbert`, or `simcse`                      | `bert`                                       |
| `--seed`               | Random seed for reproducibility                                              | `1751`                                       |
| `--num_heads`          | Number of attention heads in MultiHeadAttention                              | `256`                                        |
| `--no_attention`       | Add this flag to disable MultiHeadAttention                                  | *(disabled by default)*                      |
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
| `--model`           | `bert`                                              | Name embedding model: `bert`, `sbert`, or `simcse` |
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

## 📬 Contact

If you have any questions, encounter issues with the code, or would like to know more about our work, please contact the corresponding author:

- 📧 **Personal email (permanent)**: [oussama.mechhour.cirad@gmail.com](mailto:oussama.mechhour.cirad@gmail.com)  
- 📧 **Professional email (not sure if permanent)**: [oussama.mechhour@cirad.fr](mailto:oussama.mechhour@cirad.fr)

[![LinkedIn](https://img.shields.io/badge/LinkedIn-blue?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/oussama-mechhour-ph-d-student-31a94323a/)
[![ORCID](https://img.shields.io/badge/ORCID-0000--0002--1825--0097-a6ce39?logo=orcid&logoColor=white)](https://orcid.org/0009-0004-3007-1229)
[![HAL](https://img.shields.io/badge/HAL-Archive-orange?logo=data&logoColor=white)](https://hal.science/search/index/?q=oussama+mechhour)
