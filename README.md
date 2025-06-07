# MAEVa: Matching Agroecological Experimental Variables

**Description:** This repository provides the full pipeline used for our initial approach, called MAEVa, to match experimental agroecological variables using hybrid similarity techniques that combine name embeddings and description vectors. With this code, you can reproduce all the experiments described in our article and easily adapt the approach to your own task if you're working on a similar problem.

**Abstract:**
Source variables or observable properties used to describe agroecological experiments are heterogeneous, nonstandardized, and multilingual, which makes them challenging to understand, explain, and use in cropping system modeling and multicriteria evaluations of agroecological system performance. Data annotation via a controlled vocabulary, known as candidate variables from the agroecological global information system (AEGIS), offers a solution. Text similarity measures play crucial roles in tasks such as word sense disambiguation, schema matching in databases, and data annotation. Commonly used measures include 1) string-based similarity, 2) corpus-based similarity, 3) knowledge-based similarity, and 4) hybrid-based similarity, which combine two or more of these measures. This work presents a hybrid approach called Matching Agroecological Experiment Variables (MAEVa), designed to align source and candidate variables through the following components: (1) our key innovation, which consists of extending pretrained language models (PLMs), such as BERT, with an external multi-head attention layer for matching variable names; (2) an analysis of the impact of various context enrichment techniques (e.g., snippet extraction, data generation, and scientific articles) on corpus-based similarity measures for matching variable descriptions; (3) a linear combination of components (1) and (2); and (4) a voting-based method for selecting the final matching results. Experimental results demonstrate that extending PLMs with an external multi-head attention layer improves the matching of variable names. Furthermore, corpus-based methods benefit consistently from the presence of an enriched corpus, regardless of the specific enrichment technique employed. While the combination strategy itself is not novel, it yields statistically significant improvements, with performance gains exceeding 5\%.

---

## 🚀 MAEVa: Complete Pipeline Usage Guide

This script performs the full MAEVa pipeline for matching experimental agroecological variables.

### 🔍 What it does

1. **Preprocesses variable names and descriptions**  
   - Cleaning digits, parentheses, special characters  
   - Removing stopwords  
   - Lemmatization (WordNet)  
   - Punctuation and synonym normalization (for descriptions)

2. **Generates embeddings for variable names**  
   - Using `BERT`, `SBERT`, or `SimCSE`

3. **Applies MultiHeadAttention (MHA)**  
   - Enabled by default for refining name embeddings  
   - Can be disabled with a flag

4. **Generates TF-IDF vectors for descriptions**  
   - Using a corpus (e.g., GPT-generated) passed as input

5. **Combines similarities**  
   - Weighted linear combination:  
     `final_score = 0.75 * sim_name + 0.25 * sim_description`

6. **Evaluates matching accuracy**  
   - Precision@1, 3, 5, and 10  
   - Saves evaluation reports and top-10 candidate lists

---

## ✅ Default Command

```bash
python "scripts/maeva_pipeline.py"
```

This uses:
- Model: `bert`
- Attention: enabled with 256 heads
- Data files from: `datasets/benchmarks`
- TF-IDF corpus from: `datasets/corpora/Corpus (GPT-prompt 1).txt`
- Results saved to: `outputs/`

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

MAEVa pipeline can be executed in two ways:

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

```bash
python "scripts/maeva_pipeline.py"   --base_path "Folder path containing your variable names, descriptions, and the file with correct matchings used for evaluation."   --src_names "your_source_names.txt"   --cand_names "your_candidate_names.txt"   --src_desc "your_source_descriptions.txt"   --cand_desc "your_candidate_descriptions.txt"   --context "full path to your_corpus_file.txt"   --reference_file "your_ground_truth.xlsx"   --model "Name embedding model: `bert`, `sbert`, or `simcse`"   --seed 42   --num_heads 8   --no_attention   --output_path "custom/output/folder"
```

This custom command allows full control over:
- Which files to use
- Which model to apply
- Whether to use attention or not
- Where to save the results

---

## 📦 Output Files

All evaluation files are saved as `.xlsx` inside the folder specified by `--output_path`. The following files are generated:

| Filename                                     | Description                                       |
|----------------------------------------------|---------------------------------------------------|
| `matching_names_{model}({seed}).xlsx`        | Top-10 matches and precision using name embeddings |
| `matching_descriptions_{model}({seed}).xlsx` | Top-10 matches and precision using TF-IDF         |
| `combination_method_{model}({seed}).xlsx`    | Combined scores (TF-IDF + names)                  |
| `final_results_{model}({seed}).xlsx`         | Global Precision@K evaluation                     |

---

## 👨‍💻 Author

**Oussama Mechhour**  
Ph.D. Student – University of Montpellier / CIRAD / I2S Doctoral School  
📧 Email: [your-email@domain.com]  
🌍 Research: Matching heterogeneous agroecological variable metadata with hybrid NLP techniques

---

## 📄 License

This code is licensed under the [MIT License](LICENSE).
