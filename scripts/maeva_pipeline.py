"""
MAEVa Full Pipeline: Data Preparation + Evaluation (BERT, SBERT, SimCSE)
Author: Oussama Mechhour

This script preprocesses source/candidate variable names and descriptions,
then computes similarity scores using various embedding models.
By default, MultiHeadAttention is applied to name embeddings.
Evaluation results are saved as Excel files with precision summaries.
"""

import os
import re
import string
import argparse
import random
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer
from transformers import BertTokenizer, BertModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk

nltk.download('stopwords')
nltk.download('wordnet')

# ---------------------
# Text Preprocessing
# ---------------------
def read_txt(filepath):
    with open(filepath, "r", encoding="utf-8") as file:
        return file.read().splitlines()

def read_txt_as_list(filepath):
    with open(filepath, "r", encoding="utf-8") as file:
        return [file.read()]

def clean_text(text):
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\([^()]*\)', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text.lower().strip()

def remove_stopwords(text, lang='english'):
    stop_words = set(stopwords.words(lang))
    words = text.split()
    return ' '.join([word for word in words if word not in stop_words])

def lemmatize(text):
    lemmatizer = WordNetLemmatizer()
    return ' '.join([lemmatizer.lemmatize(word) for word in text.split()])

def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))

def replace_synonyms(text):
    result = []
    for word in text.split():
        syns = wordnet.synsets(word)
        result.append(syns[0].lemmas()[0].name() if syns else word)
    return ' '.join(result)

def preprocess_names(data):
    return [lemmatize(remove_stopwords(clean_text(t))) for t in data]

def preprocess_descriptions(data):
    return [
        replace_synonyms(remove_punctuation(lemmatize(remove_stopwords(clean_text(t)))))
        for t in data
    ]

# ---------------------
# Reproducibility
# ---------------------
def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ---------------------
# MultiHeadAttention
# ---------------------
class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.attention_head_size = hidden_size // num_heads
        self.all_head_size = self.num_heads * self.attention_head_size
        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        self.out = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(0.1)

    def init_weights_uniform(self, seed):
        torch.manual_seed(seed)
        for layer in [self.query, self.key, self.value, self.out]:
            nn.init.uniform_(layer.weight, a=-1.0, b=1.0)
            nn.init.zeros_(layer.bias)

    def transpose_for_scores(self, x):
        new_shape = x.size()[:-1] + (self.num_heads, self.attention_head_size)
        return x.view(*new_shape).permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        q, k, v = self.query(hidden_states), self.key(hidden_states), self.value(hidden_states)
        q, k, v = map(self.transpose_for_scores, [q, k, v])
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.attention_head_size)
        attn_probs = self.dropout(nn.Softmax(dim=-1)(scores))
        context = torch.matmul(attn_probs, v).permute(0, 2, 1, 3).contiguous()
        return self.out(context.view(context.size(0), context.size(1), -1))

# ---------------------
# Evaluation Functions
# ---------------------
def compute_results(sim_matrix, src_list, cand_list, gt_dict):
    precisions, top10 = [], []
    for i, src in enumerate(src_list):
        scores = sim_matrix[i]
        sorted_idx = np.argsort(scores)[::-1]
        top10_vars = [cand_list[j] for j in sorted_idx[:10]]
        top10.append({"Variable source": src, "Top 10 candidates": ", ".join(top10_vars)})
        for k in [1, 3, 5, 10]:
            top_k = [cand_list[j] for j in sorted_idx[:k]]
            correct = int(gt_dict[src] in top_k)
            precisions.append({"Variable source": src, "P@k": k, "Correct": correct})
    return pd.DataFrame(precisions), pd.DataFrame(top10)

def precision_summary(df):
    total = df["Variable source"].nunique()
    return pd.DataFrame([
        {
            "P@k": k,
            "Correctly Matched": df[df["P@k"] == k]["Correct"].sum(),
            "Precision (%)": round(100 * df[df["P@k"] == k]["Correct"].sum() / total, 2)
        }
        for k in [1, 3, 5, 10]
    ])

def aggregate_global(prec1, prec2, prec3):
    all_vars = prec1["Variable source"].unique()
    summary = []
    for k in [1, 3, 5, 10]:
        count = 0
        for var in all_vars:
            c1 = prec1[(prec1["Variable source"] == var) & (prec1["P@k"] == k)]["Correct"].values[0]
            c2 = prec2[(prec2["Variable source"] == var) & (prec2["P@k"] == k)]["Correct"].values[0]
            c3 = prec3[(prec3["Variable source"] == var) & (prec3["P@k"] == k)]["Correct"].values[0]
            if max(c1, c2, c3) == 1:
                count += 1
        summary.append({"P@k": k, "Correctly Matched": count, "Precision (%)": round(100 * count / len(all_vars), 2)})
    return pd.DataFrame(summary)

# ---------------------
# Main Function
# ---------------------
def main(args):
    set_seed(args.seed)

    # === Read files ===
    lines_var_src = read_txt(os.path.join(args.base_path, args.src_names))
    lines_var_cand = read_txt(os.path.join(args.base_path, args.cand_names))
    lines_des_src = read_txt(os.path.join(args.base_path, args.src_desc))
    lines_des_cand = read_txt(os.path.join(args.base_path, args.cand_desc))
    contexte = read_txt_as_list(args.context)
    correspondances = pd.read_excel(os.path.join(args.base_path, args.reference_file))

    # === Preprocessing ===
    lines_var_src1 = [re.sub('_', ' ', l) for l in lines_var_src]
    lines_var_cand1 = [re.sub('_', ' ', l) for l in lines_var_cand]
    lines_var_src_pre3 = preprocess_names(lines_var_src1)
    lines_var_cand_pre3 = preprocess_names(lines_var_cand1)
    lines_des_src_pre5 = preprocess_descriptions(lines_des_src)
    lines_des_cand_pre5 = preprocess_descriptions(lines_des_cand)
    contexte_pre5 = preprocess_descriptions(contexte)

    varSrcList = correspondances["Variable source"].tolist()
    varCandList = lines_var_cand1
    gt_dict = dict(zip(correspondances["Variable source"], correspondances["Variable correspondante"]))

    # === Embedding ===
    if args.model == "bert":
        config = BertConfig(num_hidden_layers=2)
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", config)
        model = BertModel.from_pretrained("bert-base-uncased", config)
        src_input = tokenizer(lines_var_src_pre3, return_tensors="pt", padding=True, truncation=True)
        cand_input = tokenizer(lines_var_cand_pre3, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            src_out = model(**src_input).last_hidden_state
            cand_out = model(**cand_input).last_hidden_state
    else:
        model_name = {
            "sbert": "all-MiniLM-L6-v2",
            "simcse": "princeton-nlp/sup-simcse-bert-base-uncased"
        }[args.model]
        model = SentenceTransformer(model_name)
        src_out = torch.tensor(model.encode(lines_var_src_pre3, convert_to_numpy=True)).unsqueeze(1)
        cand_out = torch.tensor(model.encode(lines_var_cand_pre3, convert_to_numpy=True)).unsqueeze(1)

    # === Attention (default: True) ===
    if not args.no_attention:
        attn = MultiHeadAttention(hidden_size=src_out.shape[-1], num_heads=args.num_heads)
        attn.init_weights_uniform(args.seed)
        with torch.no_grad():
            src_vec = attn(src_out).mean(dim=1)
            cand_vec = attn(cand_out).mean(dim=1)
    else:
        src_vec = src_out.mean(dim=1)
        cand_vec = cand_out.mean(dim=1)

    # === Similarity Calculation ===
    sim_name = cosine_similarity(src_vec.cpu().numpy(), cand_vec.cpu().numpy())

    vectorizer = TfidfVectorizer(stop_words='english', norm='l1')
    vectorizer.fit(sorted(contexte_pre5))
    des_src_vect = vectorizer.transform(lines_des_src_pre5)
    des_cand_vect = vectorizer.transform(lines_des_cand_pre5)
    sim_tfidf = cosine_similarity(des_src_vect.toarray(), des_cand_vect.toarray())
    sim_comb = 0.25 * sim_tfidf + 0.75 * sim_name

    # === Evaluation ===
    prec1, _ = compute_results(sim_name, varSrcList, varCandList, gt_dict)
    prec2, _ = compute_results(sim_tfidf, varSrcList, varCandList, gt_dict)
    prec3, _ = compute_results(sim_comb, varSrcList, varCandList, gt_dict)

    # === Output ===
    prefix = f"{args.model}({args.seed})"
    base = args.output_path
    precision_summary(prec1).to_excel(os.path.join(base, f"matching_names_{prefix}.xlsx"), index=False)
    precision_summary(prec2).to_excel(os.path.join(base, f"matching_descriptions_{prefix}.xlsx"), index=False)
    precision_summary(prec3).to_excel(os.path.join(base, f"combination_method_{prefix}.xlsx"), index=False)
    aggregate_global(prec1, prec2, prec3).to_excel(os.path.join(base, f"final_results_{prefix}.xlsx"), index=False)
    print(f"Matching complete with {args.model.upper()} {'+ Attention' if not args.no_attention else ''}.")

# ---------------------
# CLI
# ---------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_path", type=str, default="datasets/benchmarks", help="Data and output folder path")
    parser.add_argument("--src_names", type=str, default="source_variable(names).txt")
    parser.add_argument("--cand_names", type=str, default="candidate_variable(names).txt")
    parser.add_argument("--src_desc", type=str, default="source_variable(descriptions).txt")
    parser.add_argument("--cand_desc", type=str, default="candidate_variable(descriptions).txt")
    parser.add_argument("--context", type=str, default="datasets/corpora/Corpus (GPT-prompt 1).txt")
    parser.add_argument("--reference_file", type=str, default="Correspondances.xlsx")
    parser.add_argument("--model", type=str, default="bert", choices=["bert", "sbert", "simcse"])
    parser.add_argument("--seed", type=int, default=1751)
    parser.add_argument("--num_heads", type=int, default=256)
    parser.add_argument("--no_attention", action="store_true", help="Disable MultiHeadAttention (enabled by default)")
    parser.add_argument("--output_path", type=str, default="outputs", help="Directory to save the output result files")
    args = parser.parse_args()
    main(args)
