#!/usr/bin/env python3
"""Standalone MAEVa training + preparation of THREE distinct candidate pools per target domain.

The same original MAEVa preprocessing functions are included directly below.
There is NO maeva_core.py dependency.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import math
import sys
from collections import Counter
import torch.nn.functional as F
from sklearn.model_selection import GroupShuffleSplit

"""Locally embedded MAEVa helpers (preprocessing mirrors maeva_pipeline.py).

Scientific invariants:
- BERT-base uncased, two pretrained encoder layers, frozen;
- external trainable 256-head attention (head dimension 3), 0.1 dropout;
- attention training and checkpoint selection use name cosine scores only;
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


def read_pool_candidates_only(path):
    """From test files read only target columns; NEVER load source/target gold pairs for training."""
    header = list(pd.read_excel(path, nrows=0).columns)
    domain, _, target_prefix = domain_columns(header)
    cols = [target_prefix + '_Name', target_prefix + '_Description']
    df = pd.read_excel(path, usecols=cols, dtype=str).fillna('')
    candidates = []
    for _, row in df.iterrows():
        candidate = identity(row[cols[0]], row[cols[1]])
        if candidate[0]:
            candidates.append(candidate)
    return domain, candidates


def build_pools(train_paths, pool_test_paths, directory):
    """Prepare three DIFFERENT candidate spaces for each domain.

    names:       unique exact name
    descriptions:unique exact description
    combination: unique exact (name, description) pair
    Only candidate columns are ever read from test files at this stage.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    catalog = defaultdict(set)
    pairs = []
    provenance = []
    for path in train_paths:
        rows = read_pairs(path)
        pairs.extend(rows)
        for row in rows:
            catalog[row['domain']].add(row['target'])
        provenance.append({'file': Path(path).name, 'sha256': sha256_file(path),
                           'used_for': 'training pairs and target catalog'})
    for path in pool_test_paths:
        domain, candidates = read_pool_candidates_only(path)
        catalog[domain].update(candidates)
        provenance.append({'file': Path(path).name, 'sha256': sha256_file(path),
                           'used_for': 'target columns ONLY, not source labels'})
    pools = {}
    counts = {}
    checksums = {}
    for domain in DOMAINS:
        targets = sorted(catalog[domain])
        if not targets:
            continue
        names = sorted({name for name, _ in targets})
        descriptions = sorted({desc for _, desc in targets})
        pools[domain] = {
            'names': [{'id': pool_item_id(domain, 'names', name), 'name': name}
                      for name in names],
            'descriptions': [{'id': pool_item_id(domain, 'descriptions', desc),
                              'description': desc} for desc in descriptions],
            'combination': [{'id': pool_item_id(domain, 'combination', (name, desc)),
                             'name': name, 'description': desc} for name, desc in targets],
        }
        counts[domain] = {}
        for method, records in pools[domain].items():
            filename = f'pool_{domain}_{method}.json'
            (directory / filename).write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding='utf-8')
            pd.DataFrame(records).to_csv(directory / f'pool_{domain}_{method}.csv', index=False, encoding='utf-8')
            counts[domain][method] = len(records)
            checksums[filename] = sha256_file(directory / filename)
        printlog(f'  {domain}: names={len(names)}, descriptions={len(descriptions)}, '
                 f'combination={len(targets)}')
    manifest = {
        'identity_rules': {'names': ['name'], 'descriptions': ['description'],
                           'combination': ['name', 'description']},
        'duplicate_rule': 'exact text after trim; never merge a (name,description) pair for combination unless both equal',
        'source_files': provenance, 'counts': counts, 'sha256': checksums,
    }
    (directory / 'pool_manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    return pairs, pools


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


def init_tfidf(context_path, output_dir):
    text = Path(context_path).read_text(encoding='utf-8')
    clean = preprocess_descriptions([text])  # EXACT original corpus preprocessing
    vectorizer = TfidfVectorizer(stop_words='english', norm='l1')
    vectorizer.fit(sorted(clean))  # original: sorted(context_clean) -- one document
    joblib.dump(vectorizer, Path(output_dir) / 'tfidf.joblib')
    return vectorizer


def pair_level_precision(scores, gold_indices):
    """Gold is one candidate index per annotated pair; rows may repeat per source."""
    if len(scores) != len(gold_indices):
        raise ValueError('Must have one score row per annotated pair')
    ranked = np.argsort(-scores, axis=1, kind='stable')
    correct = {k: int((ranked[:, :k] == np.asarray(gold_indices)[:, None]).any(axis=1).sum())
               for k in KS}
    n = len(gold_indices)
    return {f'P@{k}': (correct[k] / n if n else float('nan')) for k in KS}, correct, ranked


def fingerprint_config(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def file_paths(data_dir, patterns):
    files = []
    for pattern in patterns:
        path = Path(pattern)
        if path.is_file():
            files.append(path.resolve())
        elif any(c in pattern for c in '*?['):
            files.extend(data_dir.glob(pattern))
        else:
            files.extend(data_dir.glob('*' + pattern + '*'))
    return sorted(set(p.resolve() for p in files), key=str)


def group_split(pairs, seed, fraction):
    labels = np.arange(len(pairs))
    # Name-only learning: the same source name cannot occur on both sides.
    groups = [p['source'][0] for p in pairs]
    benchmarks = {p['file'] for p in pairs}
    for offset in range(200):
        splitter = GroupShuffleSplit(n_splits=1, test_size=fraction, random_state=seed + offset)
        tr, va = next(splitter.split(labels, groups=groups))
        if all(any(pairs[i]['file'] == dataset for i in va) for dataset in benchmarks):
            return tr.tolist(), va.tolist(), seed + offset
    raise ValueError('Cannot form validation split with at least one pair from each train benchmark')


def calc_val_score(attn, pairs, val_ids, pools, cand_idx, features, masks, name_idx,
                   device, batch_size):
    attn.eval()
    by_dataset = defaultdict(list)
    with torch.no_grad():
        # Select the checkpoint using names only, against each domain's full name pool.
        rankings = {}
        for domain in DOMAINS:
            rows = pools.get(domain, [])
            selected = [i for i in val_ids if pairs[i]['domain'] == domain]
            if not selected:
                continue
            uniq = sorted({pairs[i]['source'][0] for i in selected})
            candidates = [r['name'] for r in rows]
            unique_candidate_indices = [name_idx[name] for name in candidates]
            source_indices = [name_idx[name] for name in uniq]
            source_embeddings = batched_embeddings(attn, features, masks, source_indices, device, batch_size)
            target_embeddings = batched_embeddings(attn, features, masks, unique_candidate_indices, device, batch_size)
            name_scores = source_embeddings @ target_embeddings.T
            source_lookup = {s: idx for idx, s in enumerate(uniq)}
            for i in selected:
                p = pairs[i]
                gold = cand_idx[domain][p['target'][0]]
                ranks = np.argsort(-name_scores[source_lookup[p['source'][0]]], kind='stable')
                by_dataset[p['file']].append({k: float(gold in ranks[:k]) for k in (1, 3, 5, 10)})
        stats = {}
        for dataset, results in by_dataset.items():
            stats[dataset] = {f'P@{k}': float(np.mean([r[k] for r in results])) for k in (1, 3, 5, 10)}
        if not stats:
            raise RuntimeError('Empty validation benchmark(s)')
        macro_p1 = float(np.mean([v['P@1'] for v in stats.values()]))
        macro_p10 = float(np.mean([v['P@10'] for v in stats.values()]))
        return 0.5 * (macro_p1 + macro_p10), stats


def batched_embeddings(attn, features, masks, indices, device, batch_size):
    values = []
    with torch.no_grad():
        for i in range(0, len(indices), batch_size):
            sub = torch.tensor(indices[i:i + batch_size], dtype=torch.long)
            emb = encode_names(attn, features[sub].to(device), masks[sub].to(device))
            values.append(emb.detach().cpu().numpy())
    return np.concatenate(values, axis=0)


def listwise_ranking_loss(scores, temperature):
    """OLGAM-style listwise softmax objective for [positive, 9 hard negatives].

    scores: [batch, 1 + negatives], with the annotated positive in column 0.
    For each source, optimize -log(exp(score_positive / temperature) /
    sum_j exp(score_j / temperature)). Stable logsumexp implementation.
    Mathematically equivalent to F.cross_entropy(scores / temperature,
    torch.zeros(batch_size, dtype=torch.long, device=scores.device)).
    """
    logits = scores / temperature
    return (torch.logsumexp(logits, dim=1) - logits[:, 0]).mean()


def main():
    parser = argparse.ArgumentParser(description='Train fixed-BERT + trainable 256-head MAEVa attention and precompute target pools')
    parser.add_argument('--data-dir', default='.', help='Folder containing the input Excel files')
    parser.add_argument('--train', nargs='+', default=['*_train*.xlsx'], help='Training Excel filenames or glob patterns')
    parser.add_argument('--pool-test', nargs='+', default=['*_test*.xlsx'], help='Test Excel used ONLY for target columns when building pools')
    parser.add_argument('--context', required=True, help='Path to GPT-prompt 1 TXT corpus')
    parser.add_argument('--output-dir', default='maeva_run', help='Trained checkpoint, pools and run artifacts')
    parser.add_argument('--seed', type=int, default=1751)
    parser.add_argument('--validation-ratio', type=float, default=0.20)
    parser.add_argument('--num-heads', type=int, choices=[256], default=256, help='MAEVa-v1 value, fixed at 256')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--encode-batch-size', type=int, default=32)
    parser.add_argument('--negatives', type=int, default=9, help='Hard negatives per training pair')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--temperature', type=float, default=0.05, help='Temperature of supervised listwise objective')
    parser.add_argument('--device', default='auto', help='auto | cpu | cuda')
    args = parser.parse_args()
    if not 0 < args.validation_ratio < 1:
        parser.error('--validation-ratio must lie between 0 and 1')
    if args.epochs <= 0 or args.batch_size <= 0 or args.negatives <= 0 or args.temperature <= 0:
        parser.error('epochs, batch-size, negatives, temperature must be positive')
    data_dir = Path(args.data_dir)
    train_paths = file_paths(data_dir, args.train)
    test_pool_paths = file_paths(data_dir, args.pool_test)
    if not train_paths or not test_pool_paths:
        parser.error(f'No train/pool-test files found: train={train_paths}, pool-test={test_pool_paths}')
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    printlog('\n=== MAEVa: TRAINING AND POOL CONSTRUCTION ===')
    printlog(f'Train: {len(train_paths)} file(s); pool-only test inputs: {len(test_pool_paths)} file(s)')
    printlog('IMPORTANT: test source/target correspondence labels NEVER enter training or validation.')
    ensure_nltk()
    set_seed(args.seed)
    device = torch.device('cuda' if args.device == 'auto' and torch.cuda.is_available() else ('cpu' if args.device == 'auto' else args.device))
    printlog(f'Device: {device}, seed: {args.seed}, train/validation: {1-args.validation_ratio:.0%}/{args.validation_ratio:.0%}')
    pairs, method_pools = build_pools(train_paths, test_pool_paths, out_dir / 'pools')
    # Optimize names only; the other pools remain available for the unchanged test script.
    pools = {domain: p['names'] for domain, p in method_pools.items()}
    counts = Counter(p['file'] for p in pairs)
    printlog('Training annotated pairs: ' + str(dict(counts)))
    tr_ids, va_ids, split_seed = group_split(pairs, args.seed, args.validation_ratio)
    printlog(f'Group-disjoint split: {len(tr_ids)} train pairs / {len(va_ids)} validation pairs (random_state={split_seed})')
    (out_dir / 'split.json').write_text(json.dumps({
        'train': [{'file': pairs[i]['file'], 'row': pairs[i]['row']} for i in tr_ids],
        'validation': [{'file': pairs[i]['file'], 'row': pairs[i]['row']} for i in va_ids],
        'seed_used': split_seed}, indent=2), encoding='utf-8')
    printlog('Fitting original MAEVa TF-IDF on provided GPT-prompt-1 corpus (one document)...')
    init_tfidf(args.context, out_dir)  # Saved for testing; not used by the training loss or validation.
    # Per-domain full pool, including targets from all datasets pointing to the same domain.
    cand_idx = {domain: {r['name']: j for j, r in enumerate(rows)}
                for domain, rows in pools.items()}
    for p in pairs:
        if p['target'][0] not in cand_idx[p['domain']]:
            raise RuntimeError(f'Gold training target missing from pool {p}')
    printlog('Loading pretrained bert-base-uncased with exactly 2 encoder layers...')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert = BertModel.from_pretrained('bert-base-uncased', num_hidden_layers=BERT_LAYERS)
    bert.eval().to(device)
    for param in bert.parameters():
        param.requires_grad_(False)
    resolved_bert_revision = getattr(bert.config, '_commit_hash', None)
    bert.save_pretrained(out_dir / 'bert')
    tokenizer.save_pretrained(out_dir / 'tokenizer')
    all_names = sorted(set([p['source'][0] for p in pairs] +
                           [r['name'] for pool in pools.values() for r in pool]))
    name_idx = {name: i for i, name in enumerate(all_names)}
    printlog(f'Caching fixed BERT token embeddings for {len(all_names)} distinct names...')
    features, masks = bert_name_features(bert, tokenizer, all_names, device, args.encode_batch_size)
    del bert
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    attn = MultiHeadAttention(hidden_size=768, num_heads=NUM_HEADS).to(device)
    optimizer = torch.optim.AdamW(attn.parameters(), lr=args.lr)
    # Hard negatives use frozen BERT cosine, not test labels.
    base_embeddings = []
    for i in range(0, len(all_names), 256):
        base_embeddings.append(F.normalize(masked_mean(features[i:i+256], masks[i:i+256]), dim=-1).numpy())
    base_embeddings = np.vstack(base_embeddings)
    known_positive = defaultdict(set)
    for i in tr_ids:
        p = pairs[i]
        known_positive[(p['domain'], p['source'][0])].add(cand_idx[p['domain']][p['target'][0]])
    indexed = {}
    for domain, rows in pools.items():
        indexes = [name_idx[r['name']] for r in rows]
        srcs = sorted({pairs[i]['source'][0] for i in tr_ids if pairs[i]['domain'] == domain})
        for src in srcs:
            name_scores = base_embeddings[name_idx[src]] @ base_embeddings[indexes].T
            ranked = np.argsort(-name_scores, kind='stable')
            banned = known_positive[(domain, src)]
            hard = [int(j) for j in ranked if j not in banned]
            indexed[(domain, src)] = hard
    printlog(f'Trainable parameters: {sum(p.numel() for p in attn.parameters()):,} (external attention only)')
    config = {'seed': args.seed, 'split_seed': split_seed, 'bert': 'bert-base-uncased',
              'resolved_bert_revision': resolved_bert_revision,
              'num_hidden_layers': 2, 'hidden_size': 768, 'num_heads': 256,
              'dropout': DROPOUT, 'initialization': 'xavier_uniform',
              'name_weight': ALPHA_NAME, 'description_weight': ALPHA_DESCRIPTION,
              'epochs': args.epochs, 'batch_size': args.batch_size, 'lr': args.lr,
              'temperature': args.temperature, 'negatives': args.negatives,
              'validation_ratio': args.validation_ratio,
              'context_sha256': sha256_file(args.context),
              'training_objective': 'names_only', 'training_pool': 'names',
              'validation_method': 'names',
              'evaluation_unit': 'one annotated pair; exact target-name identity for training/validation'}
    (out_dir / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    history = []
    best = (-float('inf'), None)
    rng = np.random.default_rng(args.seed)
    for epoch in range(1, args.epochs + 1):
        attn.train()
        order = rng.permutation(tr_ids).tolist()
        losses = []
        for start in range(0, len(order), args.batch_size):
            batch_ids = order[start:start + args.batch_size]
            src_name_indices = []
            cand_names_indices = []
            for pair_idx in batch_ids:
                p = pairs[pair_idx]
                positive = cand_idx[p['domain']][p['target'][0]]
                negatives = indexed[(p['domain'], p['source'][0])]
                if len(negatives) < args.negatives:
                    raise ValueError(f'Only {len(negatives)} negatives for {p["domain"]}; set --negatives lower')
                picks = [positive] + negatives[:args.negatives]
                src_name_indices.append(name_idx[p['source'][0]])
                cand_names_indices.append([name_idx[pools[p['domain']][j]['name']] for j in picks])
            batch_tensor = torch.tensor(cand_names_indices, dtype=torch.long)
            src_tensor = torch.tensor(src_name_indices, dtype=torch.long)
            src_e = encode_names(attn, features[src_tensor].to(device), masks[src_tensor].to(device))
            flat_cands = batch_tensor.flatten()
            tgt_e = encode_names(attn, features[flat_cands].to(device), masks[flat_cands].to(device))
            tgt_e = tgt_e.view(len(batch_ids), len(cand_names_indices[0]), -1)
            name_sim = torch.einsum('bd,bkd->bk', src_e, tgt_e)
            loss = listwise_ranking_loss(name_sim, args.temperature)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().item()))
        val_score, val_metrics = calc_val_score(attn, pairs, va_ids, pools, cand_idx,
                                               features, masks, name_idx,
                                               device, args.encode_batch_size)
        entry = {'epoch': epoch, 'train_loss': float(np.mean(losses)),
                 'validation_balance_P1_P10': val_score, 'validation': val_metrics}
        history.append(entry)
        printlog(f'Epoch {epoch:02d}/{args.epochs}: loss={entry["train_loss"]:.5f}, '
                 f'validation names 0.5*(macroP@1+macroP@10)={val_score:.4f}')
        if val_score > best[0]:
            best = (val_score, epoch)
            torch.save({k: v.detach().cpu() for k, v in attn.state_dict().items()}, out_dir / 'attention.pt')
    (out_dir / 'history.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
    (out_dir / 'best_checkpoint.json').write_text(json.dumps({
        'selected_epoch': best[1], 'validation_balance_P1_P10': best[0],
        'model': 'bert/ + attention.pt', 'pools': 'pools/pool_DOMAIN_METHOD.json',
        'tfidf': 'tfidf.joblib'}, indent=2), encoding='utf-8')
    printlog(f'\nBEST validation epoch {best[1]}: score={best[0]:.4f}')
    printlog(f'MODEL SAVED: {out_dir.resolve()}')
    printlog(f'POOLS PRECOMPUTED: {(out_dir / "pools").resolve()}')


if __name__ == '__main__':
    main()
