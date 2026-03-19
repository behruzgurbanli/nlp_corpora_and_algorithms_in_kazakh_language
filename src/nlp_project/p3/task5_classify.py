#!/usr/bin/env python3
"""
Project 3 - Task 5:
- Compare RNN, BiRNN, and LSTM
- Use Count, TF-IDF, PMI, Word2Vec, and GloVe document features
"""

from __future__ import annotations

import json
import math
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from nlp_project.p3.common import doc_word_tokens, load_docs


@dataclass(frozen=True)
class P3Task5ClassifyConfig:
    inp_jsonl: Path
    text_field: str = "clean_text"
    label_field: str = "category"
    lowercase: bool = True
    remove_timestamps: bool = True

    train_ratio: float = 0.8
    dev_ratio: float = 0.1
    seed: int = 42

    count_max_features: int = 1000
    tfidf_max_features: int = 1000
    pmi_vocab_size: int = 5000
    pmi_window: int = 4
    pmi_vector_size: int = 100

    word2vec_json: Path = Path("data/reports/p3_task2_word2vec_report.json")
    glove_json: Path = Path("data/reports/p3_task3_glove_report.json")

    hidden_size: int = 64
    num_layers: int = 1
    dropout: float = 0.2
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    batch_size: int = 32
    epochs: int = 8

    baseline_c: float = 1.0
    baseline_max_iter: int = 2000

    prepare_only: bool = False

    out_dir: Path = Path("data/reports/p3_task5_classify")
    out_json: Path = Path("data/reports/p3_task5_classify_report.json")


def _require_torch():
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, Dataset
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Task 5 requires PyTorch. Run it on your machine in the conda env where torch is installed."
        ) from e
    return torch, nn, DataLoader, Dataset


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _load_labeled_docs(cfg: P3Task5ClassifyConfig) -> List[dict]:
    docs = load_docs(
        cfg.inp_jsonl,
        text_field=cfg.text_field,
        remove_timestamps=cfg.remove_timestamps,
    )
    out = []
    for doc in docs:
        label = doc.get(cfg.label_field)
        if label is None:
            continue
        tokens = doc_word_tokens(doc["text"], lowercase=cfg.lowercase)
        if not tokens:
            continue
        out.append(
            {
                "doc_id": doc["doc_id"],
                "label": str(label),
                "text": " ".join(tokens),
                "tokens": tokens,
            }
        )
    if not out:
        raise ValueError(f"No labeled documents found for label_field='{cfg.label_field}'.")
    return out


def _split_docs(docs: List[dict], *, cfg: P3Task5ClassifyConfig):
    labels = [d["label"] for d in docs]
    train_docs, rest_docs = train_test_split(
        docs,
        train_size=cfg.train_ratio,
        random_state=cfg.seed,
        stratify=labels,
    )
    rest_ratio = 1.0 - cfg.train_ratio
    dev_share = cfg.dev_ratio / rest_ratio
    rest_labels = [d["label"] for d in rest_docs]
    dev_docs, test_docs = train_test_split(
        rest_docs,
        train_size=dev_share,
        random_state=cfg.seed,
        stratify=rest_labels,
    )
    return train_docs, dev_docs, test_docs


def _texts(docs: Sequence[dict]) -> List[str]:
    return [d["text"] for d in docs]


def _tokens(docs: Sequence[dict]) -> List[List[str]]:
    return [d["tokens"] for d in docs]


def _labels(docs: Sequence[dict]) -> List[str]:
    return [d["label"] for d in docs]


def _load_vectors_from_report(report_json: Path) -> Dict[str, np.ndarray]:
    report = json.loads(report_json.read_text(encoding="utf-8"))
    vec_path = Path(report["artifacts"]["vectors_path"])
    if not vec_path.exists():
        raise FileNotFoundError(f"Missing vectors file referenced by {report_json}: {vec_path}")

    vectors: Dict[str, np.ndarray] = {}
    with vec_path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip().split()
            if len(parts) < 3:
                continue
            # Word2Vec text format may include a header row: "vocab dim"
            if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                continue
            word = parts[0]
            try:
                vec = np.array(parts[1:], dtype=np.float32)
            except ValueError:
                continue
            vectors[word] = vec
    if not vectors:
        raise ValueError(f"No vectors could be loaded from {vec_path}")
    return vectors


def _average_embedding_features(token_docs: Sequence[List[str]], vectors: Dict[str, np.ndarray]) -> np.ndarray:
    dim = len(next(iter(vectors.values())))
    X = np.zeros((len(token_docs), dim), dtype=np.float32)
    for i, toks in enumerate(token_docs):
        arrs = [vectors[t] for t in toks if t in vectors]
        if arrs:
            X[i] = np.mean(arrs, axis=0, dtype=np.float32)
    return X


def _build_ppmi_embeddings(train_token_docs: Sequence[List[str]], *, vocab_size: int, window: int, vector_size: int):
    word_counts = Counter()
    for toks in train_token_docs:
        word_counts.update(toks)

    vocab = [w for w, _ in word_counts.most_common(vocab_size)]
    vocab_index = {w: i for i, w in enumerate(vocab)}
    pair_counts = Counter()
    total_pairs = 0
    total_tokens = 0

    for toks in train_token_docs:
        toks = [t for t in toks if t in vocab_index]
        total_tokens += len(toks)
        for i, wi in enumerate(toks):
            left = max(0, i - window)
            right = min(len(toks), i + window + 1)
            for j in range(left, right):
                if i == j:
                    continue
                wj = toks[j]
                pair_counts[(wi, wj)] += 1
                total_pairs += 1

    if not vocab or total_pairs == 0:
        return {}, np.zeros((0, 0), dtype=np.float32)

    rows = []
    cols = []
    data = []
    for (wi, wj), c in pair_counts.items():
        p_ij = c / total_pairs
        p_i = word_counts[wi] / total_tokens
        p_j = word_counts[wj] / total_tokens
        if p_i <= 0 or p_j <= 0 or p_ij <= 0:
            continue
        ppmi = max(math.log2(p_ij / (p_i * p_j)), 0.0)
        if ppmi > 0:
            rows.append(vocab_index[wi])
            cols.append(vocab_index[wj])
            data.append(ppmi)

    if not data:
        return {}, np.zeros((0, 0), dtype=np.float32)

    from scipy.sparse import csr_matrix

    mat = csr_matrix((data, (rows, cols)), shape=(len(vocab), len(vocab)), dtype=np.float32)
    n_components = max(2, min(vector_size, len(vocab) - 1))
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    emb = svd.fit_transform(mat).astype(np.float32)
    vectors = {w: emb[idx] for w, idx in vocab_index.items()}
    return vectors, emb


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(p_macro),
        "recall_macro": float(r_macro),
        "f1_macro": float(f1_macro),
        "precision_weighted": float(p_weighted),
        "recall_weighted": float(r_weighted),
        "f1_weighted": float(f1_weighted),
    }


def _build_feature_sets(cfg: P3Task5ClassifyConfig, train_docs: List[dict], dev_docs: List[dict], test_docs: List[dict]) -> dict:
    processors = _build_feature_processors(cfg, train_docs)

    train_texts = _texts(train_docs)
    dev_texts = _texts(dev_docs)
    test_texts = _texts(test_docs)

    train_tokens = _tokens(train_docs)
    dev_tokens = _tokens(dev_docs)
    test_tokens = _tokens(test_docs)

    feature_sets: Dict[str, dict] = {}

    X_train_count = _transform_docs_with_processor(processors["count"], texts=train_texts, tokens=train_tokens)
    X_dev_count = _transform_docs_with_processor(processors["count"], texts=dev_texts, tokens=dev_tokens)
    X_test_count = _transform_docs_with_processor(processors["count"], texts=test_texts, tokens=test_tokens)
    feature_sets["count"] = {
        "train": X_train_count,
        "dev": X_dev_count,
        "test": X_test_count,
        "dim": X_train_count.shape[1],
    }

    X_train_tfidf = _transform_docs_with_processor(processors["tfidf"], texts=train_texts, tokens=train_tokens)
    X_dev_tfidf = _transform_docs_with_processor(processors["tfidf"], texts=dev_texts, tokens=dev_tokens)
    X_test_tfidf = _transform_docs_with_processor(processors["tfidf"], texts=test_texts, tokens=test_tokens)
    feature_sets["tfidf"] = {
        "train": X_train_tfidf,
        "dev": X_dev_tfidf,
        "test": X_test_tfidf,
        "dim": X_train_tfidf.shape[1],
    }

    X_train_pmi = _transform_docs_with_processor(processors["pmi"], texts=train_texts, tokens=train_tokens)
    X_dev_pmi = _transform_docs_with_processor(processors["pmi"], texts=dev_texts, tokens=dev_tokens)
    X_test_pmi = _transform_docs_with_processor(processors["pmi"], texts=test_texts, tokens=test_tokens)
    feature_sets["pmi"] = {
        "train": X_train_pmi,
        "dev": X_dev_pmi,
        "test": X_test_pmi,
        "dim": X_train_pmi.shape[1] if len(X_train_pmi.shape) == 2 else 0,
        "pmi_vocab": len(processors["pmi"]["vectors"]),
        "pmi_matrix_shape": list(processors["pmi"]["matrix_shape"]),
    }

    X_train_w2v = _transform_docs_with_processor(processors["word2vec"], texts=train_texts, tokens=train_tokens)
    X_dev_w2v = _transform_docs_with_processor(processors["word2vec"], texts=dev_texts, tokens=dev_tokens)
    X_test_w2v = _transform_docs_with_processor(processors["word2vec"], texts=test_texts, tokens=test_tokens)
    feature_sets["word2vec"] = {
        "train": X_train_w2v,
        "dev": X_dev_w2v,
        "test": X_test_w2v,
        "dim": X_train_w2v.shape[1],
    }

    X_train_glove = _transform_docs_with_processor(processors["glove"], texts=train_texts, tokens=train_tokens)
    X_dev_glove = _transform_docs_with_processor(processors["glove"], texts=dev_texts, tokens=dev_tokens)
    X_test_glove = _transform_docs_with_processor(processors["glove"], texts=test_texts, tokens=test_tokens)
    feature_sets["glove"] = {
        "train": X_train_glove,
        "dev": X_dev_glove,
        "test": X_test_glove,
        "dim": X_train_glove.shape[1],
    }

    return feature_sets


def _build_feature_processors(cfg: P3Task5ClassifyConfig, train_docs: List[dict]) -> dict:
    train_texts = _texts(train_docs)
    train_tokens = _tokens(train_docs)

    count_vec = CountVectorizer(max_features=cfg.count_max_features)
    count_vec.fit(train_texts)

    tfidf_vec = TfidfVectorizer(max_features=cfg.tfidf_max_features)
    tfidf_vec.fit(train_texts)

    pmi_vectors, pmi_matrix = _build_ppmi_embeddings(
        train_tokens,
        vocab_size=cfg.pmi_vocab_size,
        window=cfg.pmi_window,
        vector_size=cfg.pmi_vector_size,
    )

    return {
        "count": {"kind": "vectorizer", "transformer": count_vec},
        "tfidf": {"kind": "vectorizer", "transformer": tfidf_vec},
        "pmi": {"kind": "embeddings", "vectors": pmi_vectors, "matrix_shape": pmi_matrix.shape},
        "word2vec": {"kind": "embeddings", "vectors": _load_vectors_from_report(cfg.word2vec_json)},
        "glove": {"kind": "embeddings", "vectors": _load_vectors_from_report(cfg.glove_json)},
    }


def _transform_docs_with_processor(processor: dict, *, texts: Sequence[str], tokens: Sequence[List[str]]) -> np.ndarray:
    if processor["kind"] == "vectorizer":
        return processor["transformer"].transform(texts).toarray().astype(np.float32)
    if processor["kind"] == "embeddings":
        return _average_embedding_features(tokens, processor["vectors"])
    raise ValueError(f"Unsupported processor kind: {processor['kind']}")


def _vectorize_single_text(text: str, *, lowercase: bool) -> tuple[str, List[str]]:
    tokens = doc_word_tokens(text, lowercase=lowercase)
    return " ".join(tokens), tokens


def _train_logreg_predictor(cfg: P3Task5ClassifyConfig, X_train: np.ndarray, y_train: np.ndarray):
    model = LogisticRegression(
        C=cfg.baseline_c,
        max_iter=cfg.baseline_max_iter,
        random_state=cfg.seed,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)
    return model


def _build_recurrent_predictor(cfg: P3Task5ClassifyConfig, X_train: np.ndarray, y_train: np.ndarray, num_classes: int, model_name: str):
    torch, nn, DataLoader, Dataset = _require_torch()

    class FeatureDataset(Dataset):
        def __init__(self, X: np.ndarray, y: np.ndarray):
            self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)
            self.y = torch.tensor(y, dtype=torch.long)

        def __len__(self):
            return len(self.y)

        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]

    class RecurrentClassifier(nn.Module):
        def __init__(self, model_name: str, hidden_size: int, num_layers: int, dropout: float, num_classes: int):
            super().__init__()
            bidirectional = model_name == "birnn"
            if model_name in {"rnn", "birnn"}:
                self.rnn = nn.RNN(
                    input_size=1,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    batch_first=True,
                    bidirectional=bidirectional,
                    dropout=dropout if num_layers > 1 else 0.0,
                )
            elif model_name == "lstm":
                self.rnn = nn.LSTM(
                    input_size=1,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    batch_first=True,
                    bidirectional=False,
                    dropout=dropout if num_layers > 1 else 0.0,
                )
            else:
                raise ValueError(f"Unsupported model: {model_name}")
            out_dim = hidden_size * (2 if bidirectional else 1)
            self.classifier = nn.Linear(out_dim, num_classes)
            self.model_name = model_name
            self.bidirectional = bidirectional

        def forward(self, x):
            out = self.rnn(x)
            if self.model_name == "lstm":
                _, (h_n, _) = out
            else:
                _, h_n = out
            if self.bidirectional:
                last = torch.cat([h_n[-2], h_n[-1]], dim=1)
            else:
                last = h_n[-1]
            return self.classifier(last)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_counts = np.bincount(y_train, minlength=num_classes).astype(np.float32)
    class_weights = (len(y_train) / (num_classes * np.maximum(class_counts, 1.0))).astype(np.float32)
    weight_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)

    dataset = FeatureDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)
    model = RecurrentClassifier(
        model_name=model_name,
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
        num_classes=num_classes,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)

    model.train()
    for _ in range(cfg.epochs):
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

    model.eval()
    return {"torch": torch, "model": model, "device": device}


def _train_recurrent_models(
    cfg: P3Task5ClassifyConfig,
    feature_sets: dict,
    y_train: np.ndarray,
    y_dev: np.ndarray,
    y_test: np.ndarray,
    num_classes: int,
):
    torch, nn, DataLoader, Dataset = _require_torch()

    class FeatureDataset(Dataset):
        def __init__(self, X: np.ndarray, y: np.ndarray):
            self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)
            self.y = torch.tensor(y, dtype=torch.long)

        def __len__(self):
            return len(self.y)

        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]

    class RecurrentClassifier(nn.Module):
        def __init__(self, model_name: str, hidden_size: int, num_layers: int, dropout: float, num_classes: int):
            super().__init__()
            bidirectional = model_name == "birnn"
            if model_name in {"rnn", "birnn"}:
                self.rnn = nn.RNN(
                    input_size=1,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    batch_first=True,
                    bidirectional=bidirectional,
                    dropout=dropout if num_layers > 1 else 0.0,
                )
            elif model_name == "lstm":
                self.rnn = nn.LSTM(
                    input_size=1,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    batch_first=True,
                    bidirectional=False,
                    dropout=dropout if num_layers > 1 else 0.0,
                )
            else:
                raise ValueError(f"Unsupported model: {model_name}")
            out_dim = hidden_size * (2 if bidirectional else 1)
            self.classifier = nn.Linear(out_dim, num_classes)
            self.model_name = model_name
            self.bidirectional = bidirectional

        def forward(self, x):
            out = self.rnn(x)
            if self.model_name == "lstm":
                _, (h_n, _) = out
            else:
                _, h_n = out

            if self.bidirectional:
                last = torch.cat([h_n[-2], h_n[-1]], dim=1)
            else:
                last = h_n[-1]
            return self.classifier(last)

    def evaluate(model, loader, device):
        model.eval()
        y_true = []
        y_pred = []
        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(device)
                logits = model(xb)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                y_pred.extend(preds.tolist())
                y_true.extend(yb.numpy().tolist())
        return _metrics(np.array(y_true), np.array(y_pred))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_counts = np.bincount(y_train, minlength=num_classes).astype(np.float32)
    class_weights = (len(y_train) / (num_classes * np.maximum(class_counts, 1.0))).astype(np.float32)
    weight_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)

    all_results: List[dict] = []
    summary_rows: List[dict] = []

    for feature_name, feat in feature_sets.items():
        train_loader = DataLoader(FeatureDataset(feat["train"], y_train), batch_size=cfg.batch_size, shuffle=True)
        dev_loader = DataLoader(FeatureDataset(feat["dev"], y_dev), batch_size=cfg.batch_size, shuffle=False)
        test_loader = DataLoader(FeatureDataset(feat["test"], y_test), batch_size=cfg.batch_size, shuffle=False)

        for model_name in ("rnn", "birnn", "lstm"):
            model = RecurrentClassifier(
                model_name=model_name,
                hidden_size=cfg.hidden_size,
                num_layers=cfg.num_layers,
                dropout=cfg.dropout,
                num_classes=num_classes,
            ).to(device)
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=cfg.learning_rate,
                weight_decay=cfg.weight_decay,
            )
            criterion = nn.CrossEntropyLoss(weight=weight_tensor)

            best_state = None
            best_dev_f1 = -1.0
            for _ in range(cfg.epochs):
                model.train()
                for xb, yb in train_loader:
                    xb = xb.to(device)
                    yb = yb.to(device)
                    optimizer.zero_grad()
                    logits = model(xb)
                    loss = criterion(logits, yb)
                    loss.backward()
                    optimizer.step()

                dev_metrics = evaluate(model, dev_loader, device)
                if dev_metrics["f1_macro"] > best_dev_f1:
                    best_dev_f1 = dev_metrics["f1_macro"]
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

            if best_state is not None:
                model.load_state_dict(best_state)

            train_metrics = evaluate(model, train_loader, device)
            dev_metrics = evaluate(model, dev_loader, device)
            test_metrics = evaluate(model, test_loader, device)

            all_results.append(
                {
                    "feature": feature_name,
                    "model": model_name,
                    "train": train_metrics,
                    "dev": dev_metrics,
                    "test": test_metrics,
                }
            )
            summary_rows.append(
                {
                    "feature": feature_name,
                    "model": model_name,
                    "test_accuracy": test_metrics["accuracy"],
                    "test_f1_macro": test_metrics["f1_macro"],
                    "test_f1_weighted": test_metrics["f1_weighted"],
                    "dev_f1_macro": dev_metrics["f1_macro"],
                }
            )

    return all_results, summary_rows


def _train_baselines(
    cfg: P3Task5ClassifyConfig,
    feature_sets: dict,
    y_train: np.ndarray,
    y_dev: np.ndarray,
    y_test: np.ndarray,
):
    all_results: List[dict] = []
    summary_rows: List[dict] = []

    for feature_name, feat in feature_sets.items():
        model = LogisticRegression(
            C=cfg.baseline_c,
            max_iter=cfg.baseline_max_iter,
            random_state=cfg.seed,
            class_weight="balanced",
        )
        model.fit(feat["train"], y_train)

        train_pred = model.predict(feat["train"])
        dev_pred = model.predict(feat["dev"])
        test_pred = model.predict(feat["test"])

        train_metrics = _metrics(y_train, train_pred)
        dev_metrics = _metrics(y_dev, dev_pred)
        test_metrics = _metrics(y_test, test_pred)

        all_results.append(
            {
                "feature": feature_name,
                "model": "logreg_baseline",
                "train": train_metrics,
                "dev": dev_metrics,
                "test": test_metrics,
            }
        )
        summary_rows.append(
            {
                "feature": feature_name,
                "model": "logreg_baseline",
                "test_accuracy": test_metrics["accuracy"],
                "test_f1_macro": test_metrics["f1_macro"],
                "test_f1_weighted": test_metrics["f1_weighted"],
                "dev_f1_macro": dev_metrics["f1_macro"],
            }
        )

    return all_results, summary_rows


def run_p3_task5_classify(cfg: P3Task5ClassifyConfig) -> dict:
    if not cfg.inp_jsonl.exists():
        raise FileNotFoundError(f"Missing input file: {cfg.inp_jsonl}")
    if not cfg.word2vec_json.exists():
        raise FileNotFoundError(f"Missing Word2Vec report: {cfg.word2vec_json}")
    if not cfg.glove_json.exists():
        raise FileNotFoundError(f"Missing GloVe report: {cfg.glove_json}")

    _set_seed(cfg.seed)
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    cfg.out_json.parent.mkdir(parents=True, exist_ok=True)

    docs = _load_labeled_docs(cfg)
    train_docs, dev_docs, test_docs = _split_docs(docs, cfg=cfg)

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(_labels(train_docs))
    y_dev = label_encoder.transform(_labels(dev_docs))
    y_test = label_encoder.transform(_labels(test_docs))

    feature_sets = _build_feature_sets(cfg, train_docs, dev_docs, test_docs)
    feature_summary = {
        name: {
            "train_shape": list(feat["train"].shape),
            "dev_shape": list(feat["dev"].shape),
            "test_shape": list(feat["test"].shape),
            "dim": int(feat["dim"]),
            **{k: v for k, v in feat.items() if k not in {"train", "dev", "test", "dim"}},
        }
        for name, feat in feature_sets.items()
    }

    result = {
        "dataset": {
            "input_path": cfg.inp_jsonl.as_posix(),
            "text_field": cfg.text_field,
            "label_field": cfg.label_field,
            "documents_total": len(docs),
            "documents_train": len(train_docs),
            "documents_dev": len(dev_docs),
            "documents_test": len(test_docs),
            "label_distribution_total": dict(Counter(_labels(docs))),
            "label_distribution_train": dict(Counter(_labels(train_docs))),
            "labels": label_encoder.classes_.tolist(),
        },
        "settings": {
            "lowercase": cfg.lowercase,
            "remove_timestamps": cfg.remove_timestamps,
            "train_ratio": cfg.train_ratio,
            "dev_ratio": cfg.dev_ratio,
            "seed": cfg.seed,
            "count_max_features": cfg.count_max_features,
            "tfidf_max_features": cfg.tfidf_max_features,
            "pmi_vocab_size": cfg.pmi_vocab_size,
            "pmi_window": cfg.pmi_window,
            "pmi_vector_size": cfg.pmi_vector_size,
            "hidden_size": cfg.hidden_size,
            "num_layers": cfg.num_layers,
            "dropout": cfg.dropout,
            "learning_rate": cfg.learning_rate,
            "weight_decay": cfg.weight_decay,
            "batch_size": cfg.batch_size,
            "epochs": cfg.epochs,
            "baseline_c": cfg.baseline_c,
            "baseline_max_iter": cfg.baseline_max_iter,
            "prepare_only": cfg.prepare_only,
        },
        "feature_summary": feature_summary,
    }

    if cfg.prepare_only:
        result["status"] = "prepared_only"
        cfg.out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        return result

    baseline_results, baseline_rows = _train_baselines(
        cfg=cfg,
        feature_sets=feature_sets,
        y_train=y_train,
        y_dev=y_dev,
        y_test=y_test,
    )

    neural_results, neural_rows = _train_recurrent_models(
        cfg=cfg,
        feature_sets=feature_sets,
        y_train=y_train,
        y_dev=y_dev,
        y_test=y_test,
        num_classes=len(label_encoder.classes_),
    )
    summary_rows = baseline_rows + neural_rows

    summary_csv = cfg.out_dir / "comparison_table.csv"
    pd.DataFrame(summary_rows).sort_values(["feature", "model"]).to_csv(summary_csv, index=False)

    result["status"] = "trained"
    result["baseline_results"] = baseline_results
    result["results"] = neural_results
    result["artifacts"] = {
        "summary_csv": summary_csv.as_posix(),
    }
    cfg.out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def format_p3_task5_classify_report(r: dict, out_json: Path) -> str:
    ds = r["dataset"]
    lines = []
    lines.append("=== PROJECT 3 - TASK 5: TEXT CLASSIFICATION ===\n")
    lines.append(f"Input: {ds['input_path']}\n")
    lines.append(f"Label field: {ds['label_field']}\n")
    lines.append(
        f"Split docs: train={ds['documents_train']}  dev={ds['documents_dev']}  "
        f"test={ds['documents_test']}  total={ds['documents_total']}\n"
    )
    lines.append(f"Labels: {', '.join(ds['labels'])}\n")
    lines.append(f"Status: {r['status']}\n")
    if r["status"] == "trained":
        lines.append(f"Saved comparison table: {r['artifacts']['summary_csv']}\n")
    lines.append(f"Saved JSON report: {out_json.resolve()}\n")
    return "".join(lines)


def build_p3_task5_predictor(
    cfg: P3Task5ClassifyConfig,
    *,
    feature_name: str,
    model_name: str,
) -> dict:
    docs = _load_labeled_docs(cfg)
    train_docs, dev_docs, _ = _split_docs(docs, cfg=cfg)
    fit_docs = train_docs + dev_docs

    label_encoder = LabelEncoder()
    y_fit = label_encoder.fit_transform(_labels(fit_docs))
    processors = _build_feature_processors(cfg, fit_docs)

    if feature_name not in processors:
        raise ValueError(f"Unsupported feature: {feature_name}")
    processor = processors[feature_name]
    fit_texts = _texts(fit_docs)
    fit_tokens = _tokens(fit_docs)
    X_fit = _transform_docs_with_processor(processor, texts=fit_texts, tokens=fit_tokens)

    if model_name == "logreg_baseline":
        model = _train_logreg_predictor(cfg, X_fit, y_fit)
        predictor_model = {"kind": "sklearn", "model": model}
    elif model_name in {"rnn", "birnn", "lstm"}:
        predictor_model = _build_recurrent_predictor(
            cfg,
            X_fit,
            y_fit,
            num_classes=len(label_encoder.classes_),
            model_name=model_name,
        )
        predictor_model["kind"] = "torch"
        predictor_model["model_name"] = model_name
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return {
        "cfg": cfg,
        "feature_name": feature_name,
        "model_name": model_name,
        "processor": processor,
        "label_encoder": label_encoder,
        "predictor_model": predictor_model,
    }


def predict_with_p3_task5(text: str, predictor: dict) -> dict:
    text = (text or "").strip()
    if not text:
        return {"predicted_label": None, "top_predictions": [], "tokens": []}

    cfg: P3Task5ClassifyConfig = predictor["cfg"]
    norm_text, tokens = _vectorize_single_text(text, lowercase=cfg.lowercase)
    X = _transform_docs_with_processor(
        predictor["processor"],
        texts=[norm_text],
        tokens=[tokens],
    )

    pred_model = predictor["predictor_model"]
    label_encoder: LabelEncoder = predictor["label_encoder"]

    if pred_model["kind"] == "sklearn":
        model = pred_model["model"]
        probs = model.predict_proba(X)[0]
    else:
        torch = pred_model["torch"]
        model = pred_model["model"]
        device = pred_model["device"]
        with torch.no_grad():
            xb = torch.tensor(X, dtype=torch.float32).unsqueeze(-1).to(device)
            logits = model(xb)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    order = np.argsort(-probs)
    top_predictions = [
        {"label": str(label_encoder.inverse_transform([idx])[0]), "score": float(probs[idx])}
        for idx in order[: min(5, len(order))]
    ]
    return {
        "predicted_label": top_predictions[0]["label"] if top_predictions else None,
        "top_predictions": top_predictions,
        "tokens": tokens,
        "feature_dim": int(X.shape[1]) if len(X.shape) == 2 else 0,
    }
