import os
import re
import math
import argparse
import random
import string
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertModel, BertConfig
from sentence_transformers import SentenceTransformer
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer

# Téléchargement des ressources
nltk.download('stopwords')
nltk.download('wordnet')

# --- Fix seed for reproducibility ---
def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# --- MultiHeadAttention ---
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
        x = x.view(*new_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        q, k, v = self.query(hidden_states), self.key(hidden_states), self.value(hidden_states)
        q, k, v = map(self.transpose_for_scores, [q, k, v])
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.attention_head_size)
        attn_probs = self.dropout(nn.Softmax(dim=-1)(scores))
        context = torch.matmul(attn_probs, v).permute(0, 2, 1, 3).contiguous()
        context = context.view(context.size(0), context.size(1), -1)
        return self.out(context)

# --- Text preprocessing ---
def read_txt(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read().splitlines()

def read_txt_as_list(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return [f.read()]

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

def preprocess_descriptions(data):
    step1 = [clean_text(t) for t in data]
    step2 = [remove_stopwords(t) for t in step1]
    step3 = [lemmatize(t) for t in step2]
    step4 = [remove_punctuation(t) for t in step3]
    step5 = [replace_synonyms(t) for t in step4]
    return step5

def preprocess_names(data):
    step1 = [clean_text(t) for t in data]
    step2 = [remove_stopwords(t) for t in step1]
    step3 = [lemmatize(t) for t in step2]
    return step3

# --- Evaluation functions ---
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
        {"P@k": k, "Correctly Matched": df[df["P@k"] == k]["Correct"].sum(),
         "Precision (%)": round(100 * df[df["P@k"] == k]["Correct"].sum() / total, 2)}
        for k in [1, 3, 5, 10]
    ])

def aggregate_global(prec_name, prec_tfidf, prec_comb):
    all_vars = prec_name["Variable source"].unique()
    global_results = []
    for k in [1, 3, 5, 10]:
        count = 0
        for var in all_vars:
            c1 = prec_name[(prec_name["Variable source"] == var) & (prec_name["P@k"] == k)]["Correct"].values[0]
            c2 = prec_tfidf[(prec_tfidf["Variable source"] == var) & (prec_tfidf["P@k"] == k)]["Correct"].values[0]
            c3 = prec_comb[(prec_comb["Variable source"] == var) & (prec_comb["P@k"] == k)]["Correct"].values[0]
            if max(c1, c2, c3) == 1:
                count += 1
        global_results.append({"P@k": k, "Correctly Matched": count, "Precision (%)": round(100 * count / len(all_vars), 2)})
    return pd.DataFrame(global_results)

# --- Main pipeline ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MAEVa Variable Matching Pipeline")
    parser.add_argument("--base_path", type=str, default="datasets/benchmarks", help="Path to the base folder containing data")
    parser.add_argument("--src_names", type=str, default="source_variable(names).txt", help="File with source variable names")
    parser.add_argument("--cand_names", type=str, default="candidate_variable(names).txt", help="File with candidate variable names")
    parser.add_argument("--src_desc", type=str, default="source_variable(descriptions).txt", help="File with source variable descriptions")
    parser.add_argument("--cand_desc", type=str, default="candidate_variable(descriptions).txt", help="File with candidate variable descriptions")
    parser.add_argument("--context", type=str, default="datasets/corpora/Corpus (GPT-prompt 1).txt", help="Context corpus file path")
    parser.add_argument("--reference_file", type=str, default="Correspondances.xlsx", help="Excel file with reference matchings")
    parser.add_argument("--output_path", type=str, default="outputs", help="Directory to save the result files")
    parser.add_argument("--model", type=str, choices=["bert", "sbert", "simcse"], default="bert", help="Embedding model to use")
    parser.add_argument("--num_heads", type=int, default=256, help="Number of heads in MultiHeadAttention")
    parser.add_argument("--seed", type=int, default=1751, help="Random seed for reproducibility")
    parser.add_argument("--no_attention", action="store_true", help="Disable MultiHeadAttention and use mean pooling instead")
    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)
    set_seed(args.seed)

    # Chargement et prétraitement
    src_names = read_txt(os.path.join(args.base_path, args.src_names))
    cand_names = read_txt(os.path.join(args.base_path, args.cand_names))
    src_desc = read_txt(os.path.join(args.base_path, args.src_desc))
    cand_desc = read_txt(os.path.join(args.base_path, args.cand_desc))
    context = read_txt_as_list(args.context)
    correspondances = pd.read_excel(os.path.join(args.base_path, args.reference_file))

    src_names_clean = preprocess_names(src_names)
    cand_names_clean = preprocess_names(cand_names)
    src_desc_clean = preprocess_descriptions(src_desc)
    cand_desc_clean = preprocess_descriptions(cand_desc)
    context_clean = preprocess_descriptions(context)

    varSrcList = correspondances["Variable source"].tolist()
    varCandList = [re.sub('_', ' ', line) for line in cand_names]
    gt_dict = dict(zip(correspondances["Variable source"], correspondances["Variable correspondante"]))

    # Embedding + attention
    if args.model == "bert":
        config = BertConfig(num_hidden_layers=2)
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", config=config)
        model = BertModel.from_pretrained("bert-base-uncased", config=config)
        src_inputs = tokenizer(src_names_clean, padding=True, truncation=True, return_tensors="pt")
        cand_inputs = tokenizer(cand_names_clean, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            src_out = model(**src_inputs).last_hidden_state
            cand_out = model(**cand_inputs).last_hidden_state
        if args.no_attention:
            src_attn = src_out.mean(dim=1)
            cand_attn = cand_out.mean(dim=1)
        else:
            attn = MultiHeadAttention(config.hidden_size, args.num_heads)
            attn.init_weights_uniform(args.seed)
            with torch.no_grad():
                src_attn = attn(src_out).mean(dim=1)
                cand_attn = attn(cand_out).mean(dim=1)
        sim_name = torch.cosine_similarity(src_attn.unsqueeze(1), cand_attn.unsqueeze(0), dim=2).cpu().numpy()

    elif args.model == "sbert":
        sbert = SentenceTransformer("all-MiniLM-L6-v2")
        src_emb = torch.tensor(sbert.encode(src_names_clean, convert_to_numpy=True))
        cand_emb = torch.tensor(sbert.encode(cand_names_clean, convert_to_numpy=True))
        if args.no_attention:
            sim_name = cosine_similarity(src_emb.cpu().numpy(), cand_emb.cpu().numpy())
        else:
            attn = MultiHeadAttention(src_emb.shape[1], 8)
            attn.init_weights_uniform(args.seed)
            with torch.no_grad():
                src_attn = attn(src_emb.unsqueeze(1)).squeeze(1)
                cand_attn = attn(cand_emb.unsqueeze(1)).squeeze(1)
                sim_name = cosine_similarity(src_attn.cpu().numpy(), cand_attn.cpu().numpy())

    elif args.model == "simcse":
        sim_model = SentenceTransformer("princeton-nlp/sup-simcse-bert-base-uncased")
        src_emb = torch.tensor(sim_model.encode(src_names_clean, convert_to_numpy=True))
        cand_emb = torch.tensor(sim_model.encode(cand_names_clean, convert_to_numpy=True))
        if args.no_attention:
            sim_name = cosine_similarity(src_emb.cpu().numpy(), cand_emb.cpu().numpy())
        else:
            attn = MultiHeadAttention(src_emb.shape[1], args.num_heads)
            attn.init_weights_uniform(args.seed)
            with torch.no_grad():
                src_attn = attn(src_emb.unsqueeze(1)).mean(dim=1)
                cand_attn = attn(cand_emb.unsqueeze(1)).mean(dim=1)
                sim_name = cosine_similarity(src_attn.detach().cpu().numpy(), cand_attn.detach().cpu().numpy())

    # TF-IDF sur descriptions
    vectorizer = TfidfVectorizer(stop_words='english', norm='l1')
    vectorizer.fit(sorted(context_clean))
    src_desc_vect = vectorizer.transform(src_desc_clean)
    cand_desc_vect = vectorizer.transform(cand_desc_clean)
    sim_tfidf = cosine_similarity(src_desc_vect.toarray(), cand_desc_vect.toarray())

    # Fusion des similarités
    sim_comb = 0.25 * sim_tfidf + 0.75 * sim_name

    # Évaluations
    prec_name, top10_name = compute_results(sim_name, varSrcList, varCandList, gt_dict)
    prec_tfidf, top10_tfidf = compute_results(sim_tfidf, varSrcList, varCandList, gt_dict)
    prec_comb, top10_comb = compute_results(sim_comb, varSrcList, varCandList, gt_dict)

    # Sauvegardes
    model_suffix = f"{args.model}{'_noatt' if args.no_attention else ''}({args.seed})"
    output_files = [
        f"matching_names_{model_suffix}.xlsx",
        f"matching_descriptions_{model_suffix}.xlsx",
        f"combination_method_{model_suffix}.xlsx",
        f"final_results_{model_suffix}.xlsx"
    ]

    with pd.ExcelWriter(os.path.join(args.output_path, output_files[0])) as writer:
        precision_summary(prec_name).to_excel(writer, sheet_name="precisions", index=False)
        top10_name.to_excel(writer, sheet_name="top10_candidates", index=False)

    with pd.ExcelWriter(os.path.join(args.output_path, output_files[1])) as writer:
        precision_summary(prec_tfidf).to_excel(writer, sheet_name="precisions", index=False)
        top10_tfidf.to_excel(writer, sheet_name="top10_candidates", index=False)

    with pd.ExcelWriter(os.path.join(args.output_path, output_files[2])) as writer:
        precision_summary(prec_comb).to_excel(writer, sheet_name="precisions", index=False)
        top10_comb.to_excel(writer, sheet_name="top10_candidates", index=False)

    aggregate_global(prec_name, prec_tfidf, prec_comb).to_excel(os.path.join(args.output_path, output_files[3]), index=False)

    print("\n✅ Matching terminé avec succès. Fichiers sauvegardés :")
    for file in output_files:
        print("  -", os.path.join(args.output_path, file))
