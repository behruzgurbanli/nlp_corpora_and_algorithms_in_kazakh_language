#!/usr/bin/env python3
"""
Shared utilities for Project 3.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterator, List

from nlp_project.tasks.sentseg import remove_timestamp_lines, segment_sentences
from nlp_project.tasks.tokenizers import tokenize_words


DEFAULT_STOPWORDS = {
    "jáne", "men", "bul", "dep", "deıin", "deyin", "úshin", "ushin",
    "ol", "da", "de", "boıynsha", "boyinsha", "búl", "sol", "osy",
    "bir", "eki", "ush", "tört", "tort", "jyl", "jılı", "jyly",
    "qalay", "qashan", "kim", "ne", "neler", "al", "biraq", "sonyń",
    "onyń", "onyn", "bar", "joq", "eken", "edi", "etedi", "bolady",
    "bolğan", "bolgan", "retinde", "qatysty", "arqyly", "turaly",
    "astana", "qazaqparat", "aqsh", "haa",
}


def iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def load_docs(
    path: Path,
    *,
    text_field: str,
    remove_timestamps: bool,
) -> List[Dict[str, Any]]:
    docs: List[Dict[str, Any]] = []
    for obj in iter_jsonl(path):
        text = obj.get(text_field, "") or ""
        if remove_timestamps:
            text = remove_timestamp_lines(text)
        docs.append(
            {
                "doc_id": str(obj.get("doc_id") or obj.get("url") or f"doc_{len(docs) + 1}"),
                "text": text,
                "category": obj.get("category"),
                "subcategory": obj.get("subcategory"),
            }
        )
    return docs


def doc_word_tokens(text: str, *, lowercase: bool) -> List[str]:
    return tokenize_words(text, lowercase=lowercase)


def doc_sentence_tokens(text: str, *, lowercase: bool) -> List[List[str]]:
    out: List[List[str]] = []
    for sent in segment_sentences(text):
        toks = tokenize_words(sent, lowercase=lowercase)
        if toks:
            out.append(toks)
    return out


def corpus_word_counts(docs: List[Dict[str, Any]], *, lowercase: bool) -> Counter:
    counts = Counter()
    for doc in docs:
        counts.update(doc_word_tokens(doc["text"], lowercase=lowercase))
    return counts

