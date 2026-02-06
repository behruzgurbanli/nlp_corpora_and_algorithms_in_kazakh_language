#!/usr/bin/env python3
"""
Shared tokenization helpers for tasks.

Important:
- Regex patterns are copied from your original scripts (no behavior change).
- Lowercasing is preserved exactly as before.
"""

from __future__ import annotations

import re


TOKEN_RE = re.compile(
    r"[A-Za-zÁáÓóÚúǴǵŃńÝýİıŊŋŞşÇçÖöÜüÄäḠḡ]+|\d+|[^\w\s]",
    flags=re.UNICODE,
)

WORD_RE = re.compile(
    r"[A-Za-zÁáÓóÚúǴǵŃńÝýİıŊŋŞşÇçÖöÜüÄäḠḡ]+",
    flags=re.UNICODE,
)


def tokenize_general(text: str, *, lowercase: bool = True) -> list[str]:
    if lowercase:
        text = text.lower()
    return TOKEN_RE.findall(text)


def tokenize_words(text: str, *, lowercase: bool = True) -> list[str]:
    if lowercase:
        text = text.lower()
    return WORD_RE.findall(text)
