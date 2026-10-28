#!/usr/bin/env python3
"""Standalone MAEVa test script. No maeva_core.py or training script required.

Uses domain-specific name / description / (name, description) pools pre-built by train_maeva.py.
The same MAEVa-v1 preprocessing operations and their order are embedded in this script.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import joblib

"""Locally embedded MAEVa helpers (preprocessing mirrors maeva_pipeline.py).

Scientific invariants:
- BERT-base uncased, two pretrained encoder layers, frozen;
- external trainable 256-head attention (head dimension 3), 0.1 dropout;
- original name/description text preprocessing;
- TF-IDF fitted to one preprocessed GPT-prompt-1 document, English stopwords, L1;
- final score = 0.75 name cosine + 0.25 description cosine;
- exact (name, description) candidate identities, full domain-specific target pool;
- evaluation instances are annotated pairs, not unique source variables.
"""
import hashlib
import json
import os
import random
import re
import string
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import joblib
import nltk
import numpy as np
import pandas as pd
import torch
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertModel, BertTokenizer

ALPHA_NAME = 0.75
ALPHA_DESCRIPTION = 0.25
NUM_HEADS = 256
BERT_LAYERS = 2
DROPOUT = 0.1
KS = (1, 3, 5, 10)
DOMAINS = ('DSSAT', 'STICS', 'AEGIS')


def printlog(message):
    print(message, flush=True)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)  # also set in shell before starting Python
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def ensure_nltk():
    # Never silently download/change preprocessing resources during experiments.
    for resource, locator in [('stopwords', 'corpora/stopwords'), ('wordnet', 'corpora/wordnet')]:
        try:
            nltk.data.find(locator)
        except LookupError:
            try:
                nltk.data.find(locator + '.zip')
            except LookupError as exc:
                raise RuntimeError(
                    f'Missing NLTK resource {resource}. Run: '
                    "python -m nltk.downloader stopwords wordnet omw-1.4"
                ) from exc


STOPWORDS = None
LEMMATIZER = WordNetLemmatizer()


def clean_text(text):
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\([^()]*\)', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text.lower().strip()


def remove_stopwords(text, lang='english'):
    global STOPWORDS
    if lang != 'english':
        return ' '.join(w for w in text.split() if w not in set(stopwords.words(lang)))
    if STOPWORDS is None:
        STOPWORDS = set(stopwords.words('english'))
    return ' '.join(w for w in text.split() if w not in STOPWORDS)


@lru_cache(maxsize=100000)
def _lemma(word):
    return LEMMATIZER.lemmatize(word)


def lemmatize(text):
    return ' '.join(_lemma(w) for w in text.split())


def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))


@lru_cache(maxsize=100000)
def _first_synonym(word):
    syns = wordnet.synsets(word)
    return syns[0].lemmas()[0].name() if syns else word


def replace_synonyms(text):
    return ' '.join(_first_synonym(w) for w in text.split())


def preprocess_names(data):
    step1 = [clean_text(t) for t in data]
    step2 = [remove_stopwords(t) for t in step1]
    step3 = [lemmatize(t) for t in step2]
    return step3


def preprocess_descriptions(data):
    step1 = [clean_text(t) for t in data]
    step2 = [remove_stopwords(t) for t in step1]
    step3 = [lemmatize(t) for t in step2]
    step4 = [remove_punctuation(t) for t in step3]
    step5 = [replace_synonyms(t) for t in step4]
    return step5


def identity(name, desc):
    return (str(name).strip(), str(desc).strip())


def item_id(domain, item):
    payload = json.dumps([domain, *item], ensure_ascii=False, separators=(',', ':'))
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()[:20]


def domain_columns(headers):
    """Select names/descriptions using column names, not benchmark filenames."""
    headers = set(headers)
    if {'CandidateAEGIS_Name', 'CandidateAEGIS_Description'} <= headers:
        return 'AEGIS', 'SourceAEGIS', 'CandidateAEGIS'
    for target in ('DSSAT', 'STICS'):
        if {target + '_Name', target + '_Description'} <= headers:
            sources = [p for p in ('ACME', 'ICASA', 'IntercropVALUES')
                       if {p + '_Name', p + '_Description'} <= headers]
            if len(sources) != 1:
                raise ValueError(f'Expected exactly one source prefix, got {sources}')
            return target, sources[0], target
    raise ValueError(f'Unsupported benchmark columns: {sorted(headers)}')


def read_pairs(path):
    df = pd.read_excel(path, dtype=str).fillna('')
    domain, source_prefix, target_prefix = domain_columns(df.columns)
    required = [p + s for p in (source_prefix, target_prefix) for s in ('_Name', '_Description')]
    if any(r not in df for r in required):
        raise ValueError(f'Missing required columns in {path}')
    pairs = []
    for row_idx, row in df.iterrows():
        source = identity(row[source_prefix + '_Name'], row[source_prefix + '_Description'])
        target = identity(row[target_prefix + '_Name'], row[target_prefix + '_Description'])
        if not source[0] or not target[0]:
            raise ValueError(f'Empty source/target name at row {row_idx + 2} of {path}')
        pairs.append({'domain': domain, 'source': source, 'target': target,
                      'file': Path(path).name, 'row': int(row_idx) + 2})
    return pairs


def pool_item_id(domain, method, value):
    payload = json.dumps([domain, method, value], ensure_ascii=False, separators=(',', ':'))
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()[:20]


def load_pools(directory):
    directory = Path(directory)
    manifest = json.loads((directory / 'pool_manifest.json').read_text(encoding='utf-8'))
    result = {}
    for domain, methods in manifest['counts'].items():
        result[domain] = {}
        for method, expected in methods.items():
            filename = f'pool_{domain}_{method}.json'
            if sha256_file(directory / filename) != manifest['sha256'][filename]:
                raise ValueError(f'Pool checksum mismatch: {filename}')
            rows = json.loads((directory / filename).read_text(encoding='utf-8'))
            values = []
            for row in rows:
                value = row['name'] if method == 'names' else (
                    row['description'] if method == 'descriptions' else
                    (row['name'], row['description']))
                values.append(value)
                if row['id'] != pool_item_id(domain, method, value):
                    raise ValueError(f'Pool item ID mismatch: {filename}')
            if len(rows) != expected or len(set(values)) != expected:
                raise ValueError(f'Pool duplicates or incorrect count: {filename}')
            result[domain][method] = rows
    return result


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, hidden_size=768, num_heads=NUM_HEADS):
        super().__init__()
        if hidden_size % num_heads:
            raise ValueError('hidden_size must be divisible by num_heads')
        self.num_heads = num_heads
        self.attention_head_size = hidden_size // num_heads
        self.all_head_size = num_heads * self.attention_head_size
        self.query = torch.nn.Linear(hidden_size, self.all_head_size)
        self.key = torch.nn.Linear(hidden_size, self.all_head_size)
        self.value = torch.nn.Linear(hidden_size, self.all_head_size)
        self.out = torch.nn.Linear(hidden_size, hidden_size)
        self.dropout = torch.nn.Dropout(DROPOUT)
        self.init_weights_corrected()

    def init_weights_corrected(self):
        # PyTorch's Xavier is appropriate to the fan-in/fan-out of 768-d projections.
        # Replaces MAEVa-v1 uniform(-1,+1), which saturates 3-d attention logits.
        for layer in [self.query, self.key, self.value, self.out]:
            torch.nn.init.xavier_uniform_(layer.weight)
            torch.nn.init.zeros_(layer.bias)

    def _transpose(self, x):
        return x.reshape(*x.size()[:-1], self.num_heads, self.attention_head_size).permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):
        q, k, v = (self._transpose(layer(hidden_states))
                   for layer in (self.query, self.key, self.value))
        scores = (q @ k.transpose(-1, -2)) / (self.attention_head_size ** 0.5)
        # Keys corresponding to pad tokens cannot be attended to.
        scores = scores.masked_fill(~attention_mask[:, None, None, :].bool(), -1e4)
        probs = self.dropout(scores.softmax(dim=-1))
        context = (probs @ v).permute(0, 2, 1, 3).contiguous()
        context = context.reshape(context.size(0), context.size(1), self.all_head_size)
        return self.out(context)


def masked_mean(hidden, mask):
    m = mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)


def encode_names(attn, hidden, masks, no_attention=False):
    if no_attention:
        return torch.nn.functional.normalize(masked_mean(hidden, masks), dim=-1)
    return torch.nn.functional.normalize(masked_mean(attn(hidden, masks), masks), dim=-1)


@torch.no_grad()
def bert_name_features(bert, tokenizer, names, device, batch_size=32):
    """Cache frozen BERT token states and their masks; attention remains trainable."""
    if not names:
        raise ValueError('No names to encode')
    bert.eval()
    processed = preprocess_names(names)
    # Mimic tokenizer(original, padding=True, truncation=True): one global pad length.
    max_len = max(len(tokenizer(x, truncation=True, max_length=512)['input_ids']) for x in processed)
    states = []
    masks = []
    for i in range(0, len(processed), batch_size):
        batch = tokenizer(processed[i:i + batch_size], padding='max_length',
                          truncation=True, max_length=max_len, return_tensors='pt')
        inputs = {k: v.to(device) for k, v in batch.items()}
        out = bert(**inputs).last_hidden_state
        states.append(out.detach().cpu())
        masks.append(batch['attention_mask'].cpu())
    return torch.cat(states, dim=0), torch.cat(masks, dim=0)


def pair_level_precision(scores, gold_indices):
    """Gold is one candidate index per annotated pair; rows may repeat per source."""
    if len(scores) != len(gold_indices):
        raise ValueError('Must have one score row per annotated pair')
    ranked = np.argsort(-scores, axis=1, kind='stable')
    correct = {k: int((ranked[:, :k] == np.asarray(gold_indices)[:, None]).any(axis=1).sum())
               for k in KS}
    n = len(gold_indices)
    return {f'P@{k}': (correct[k] / n if n else float('nan')) for k in KS}, correct, ranked


def emb_batches(attn, features, masks, positions, device, batch_size, no_attention):
    results = []
    with torch.no_grad():
        for start in range(0, len(positions), batch_size):
            ix = torch.tensor(positions[start:start + batch_size], dtype=torch.long)
            emb = encode_names(attn, features[ix].to(device), masks[ix].to(device),
                               no_attention=no_attention)
            results.append(emb.detach().cpu().numpy())
    return np.concatenate(results, axis=0)


def evaluation(method, scores, annotated, lookup, candidates, domain):
    """Score annotated PAIRS (not unique queries), each against its method-specific pool."""
    rows = np.asarray([lookup[p['source']] for p in annotated], dtype=np.int64)
    if method == 'names':
        index = {r['name']: i for i, r in enumerate(candidates)}
        get_gold = lambda pair: pair['target'][0]
    elif method == 'descriptions':
        index = {r['description']: i for i, r in enumerate(candidates)}
        get_gold = lambda pair: pair['target'][1]
    else:
        index = {(r['name'], r['description']): i for i, r in enumerate(candidates)}
        get_gold = lambda pair: pair['target']
    missing = [p for p in annotated if get_gold(p) not in index]
    if missing:
        raise ValueError(f'{len(missing)} gold targets missing from {domain}/{method} pool')
    gold = np.asarray([index[get_gold(p)] for p in annotated], dtype=np.int64)
    selected = scores[rows]
    precision, correct, ranked = pair_level_precision(selected, gold)
    detail = []
    for i, pair in enumerate(annotated):
        position = int(np.flatnonzero(ranked[i] == gold[i])[0]) + 1
        top10 = []
        for j in ranked[i, :10]:
            candidate = candidates[int(j)]
            top10.append({'id': candidate['id'],
                          **{key: candidate[key] for key in candidate if key not in ('id',)},
                          'score': round(float(selected[i, j]), 8)})
        detail.append({
            'benchmark': pair['file'], 'excel_row': pair['row'],
            'method': method, 'source_name': pair['source'][0],
            'source_description': pair['source'][1], 'gold_name': pair['target'][0],
            'gold_description': pair['target'][1], 'gold_rank': position,
            **{f'correct_P@{k}': int(position <= k) for k in KS},
            'top10_candidates_json': json.dumps(top10, ensure_ascii=False),
        })
    return precision, correct, detail


def main():
    parser = argparse.ArgumentParser(description='MAEVa reproducible pair-level P@k over precomputed domain-specific target pools')
    parser.add_argument('--checkpoint', required=True, help='Folder created by train_maeva.py')
    parser.add_argument('--pool-dir', required=True, help='Folder with 3 domain x 3 method precomputed pool JSONs + pool_manifest.json')
    parser.add_argument('--test', required=True, nargs='+', help='One or more test Excel files (not the train Excel)')
    parser.add_argument('--output-dir', default='maeva_test_results')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--no-attention', action='store_true', help='Optional original frozen BERT baseline without added attention')
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error('--batch-size must be positive')
    checkpoint = Path(args.checkpoint)
    pools = load_pools(args.pool_dir)
    config = json.loads((checkpoint / 'config.json').read_text(encoding='utf-8'))
    if (config['num_heads'], config['num_hidden_layers'], config['name_weight'],
            config['description_weight']) != (256, 2, 0.75, 0.25):
        raise ValueError('Checkpoint does not match the fixed MAEVa architecture')
    ensure_nltk()
    set_seed(config['seed'])
    device = torch.device('cuda' if args.device == 'auto' and torch.cuda.is_available() else ('cpu' if args.device == 'auto' else args.device))
    printlog('\n=== MAEVa: REPRODUCIBLE TEST (NO TRAINING) ===')
    printlog(f'Checkpoint: {checkpoint.resolve()} | Pools: {Path(args.pool_dir).resolve()} | Device: {device}')
    printlog(f'BERT layers=2; MHA heads=256; combination=0.75 names + 0.25 descriptions')
    printlog('Evaluation: full target pool; one annotated test pair = one evaluation instance.')
    test_files = [Path(p) for p in args.test]
    for p in test_files:
        if not p.is_file():
            parser.error(f'No such test Excel file: {p}')
        if 'train' in p.name.casefold():
            parser.error(f'This looks like a train file; --test expects held-out tests only: {p}')
    printlog('Precomputed pool sizes: ' + str({d: {m: len(r) for m, r in x.items()} for d, x in pools.items()}))
    test_pairs = {p.name: read_pairs(p) for p in test_files}
    for filename, pairs in test_pairs.items():
        d = pairs[0]['domain'] if pairs else ''
        printlog(f'  {filename}: {len(pairs)} annotated pairs, target domain {d}')
        if not pairs or d not in pools:
            raise ValueError(f'Empty benchmark or absent pool: {filename}: {d}')
    bert = BertModel.from_pretrained(str(checkpoint / 'bert'), local_files_only=True).to(device).eval()
    if bert.config.num_hidden_layers != 2:
        raise ValueError('Checkpoint BERT does not have two layers')
    for p in bert.parameters():
        p.requires_grad_(False)
    tokenizer = BertTokenizer.from_pretrained(str(checkpoint / 'tokenizer'), local_files_only=True)
    attn = MultiHeadAttention(hidden_size=768, num_heads=NUM_HEADS).to(device)
    attn.load_state_dict(torch.load(checkpoint / 'attention.pt', map_location='cpu', weights_only=True))
    attn.eval()  # disables dropout; no change to learned parameters
    vectorizer = joblib.load(checkpoint / 'tfidf.joblib')
    names = sorted(set([p['source'][0] for ps in test_pairs.values() for p in ps] +
                       [c['name'] for domain_pools in pools.values() for c in domain_pools['names']]))
    name_idx = {name: i for i, name in enumerate(names)}
    printlog(f'Computing BERT token states for {len(names)} distinct names...')
    features, masks = bert_name_features(bert, tokenizer, names, device, args.batch_size)
    del bert
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    test_out = Path(args.output_dir)
    test_out.mkdir(parents=True, exist_ok=True)
    summary = []
    full_rankings = []
    for filename, pairs in test_pairs.items():
        domain = pairs[0]['domain']
        candidates_by_method = pools[domain]
        sources = sorted({p['source'] for p in pairs})
        source_lookup = {source: i for i, source in enumerate(sources)}
        # Source names are encoded once. Each method uses its own target set.
        src_emb = emb_batches(attn, features, masks,
                              [name_idx[source[0]] for source in sources], device,
                              args.batch_size, args.no_attention)
        source_descriptions = vectorizer.transform(preprocess_descriptions([s[1] for s in sources]))
        for method in ('names', 'descriptions', 'combination'):
            candidates = candidates_by_method[method]
            if method == 'names':
                tgt_emb = emb_batches(attn, features, masks,
                    [name_idx[r['name']] for r in candidates], device, args.batch_size, args.no_attention)
                scores = src_emb @ tgt_emb.T
            elif method == 'descriptions':
                tgt_desc = vectorizer.transform(preprocess_descriptions([r['description'] for r in candidates]))
                scores = cosine_similarity(source_descriptions, tgt_desc)
            else:
                tgt_emb = emb_batches(attn, features, masks,
                    [name_idx[r['name']] for r in candidates], device, args.batch_size, args.no_attention)
                tgt_desc = vectorizer.transform(preprocess_descriptions([r['description'] for r in candidates]))
                sim_name = src_emb @ tgt_emb.T
                sim_desc = cosine_similarity(source_descriptions, tgt_desc)
                scores = ALPHA_NAME * sim_name + ALPHA_DESCRIPTION * sim_desc
            prec, correct, details = evaluation(method, scores, pairs, source_lookup,
                                                candidates, domain)
            full_rankings.extend(details)
            summary.append({'benchmark': filename, 'pool': domain, 'method': method,
                            'n_annotated_pairs': len(pairs), 'n_unique_sources': len(sources),
                            'n_pool_candidates': len(candidates), **prec,
                            **{f'correct@{k}': correct[k] for k in KS}})
            printlog(f' {filename} [{method}] pairs={len(pairs)}, pool={len(candidates)} | '
                     + ' '.join(f'P@{k}={prec[f"P@{k}"]*100:.2f}%' for k in KS))
    df = pd.DataFrame(summary)
    df.to_csv(test_out / 'precision_by_benchmark.csv', index=False)
    pd.DataFrame(full_rankings).to_csv(test_out / 'rankings_top10_each_pair.csv', index=False)
    macro = df.groupby('method', sort=False)[['P@1', 'P@3', 'P@5', 'P@10']].mean().reset_index()
    macro.to_csv(test_out / 'macro_average.csv', index=False)
    printlog('\n=== MACRO MEAN ACROSS TEST BENCHMARKS ===')
    for _, row in macro.iterrows():
        printlog(f"{row['method']:13s} " + ' '.join(f'P@{k}={row[f"P@{k}"]*100:.2f}%' for k in (1,3,5,10)))
    bert_model_path = checkpoint / 'bert' / 'model.safetensors'
    if not bert_model_path.is_file():
        bert_model_path = checkpoint / 'bert' / 'pytorch_model.bin'
    report = {'checkpoint_config': config,
              'seed': config['seed'], 'no_attention': args.no_attention,
              'checkpoint_weights_sha256': sha256_file(checkpoint / 'attention.pt'),
              'saved_bert_sha256': sha256_file(bert_model_path),
              'tfidf_sha256': sha256_file(checkpoint / 'tfidf.joblib'),
              'pool_manifest_sha256': sha256_file(Path(args.pool_dir) / 'pool_manifest.json'),
              'test_input_sha256': {str(p): sha256_file(p) for p in test_files},
              'benchmark_count': len(test_files), 'macro': macro.to_dict(orient='records')}
    (test_out / 'run_manifest.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    printlog(f'\nOUTPUTS SAVED: {test_out.resolve()}')


if __name__ == '__main__':
    main()
