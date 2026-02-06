#!/usr/bin/env python3
"""
Sentence segmentation evaluation (v2).

Uses GOLD blocks in sent_eval_gold.txt:
- GOLD section contains newline-separated sentences
- evaluation compares boundary offsets in normalized evaluation text
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Set

from nlp_project.tasks.sentseg import remove_timestamp_lines, segment_sentences


@dataclass(frozen=True)
class SentSegEvalConfig:
    gold_path: Path


def parse_gold_blocks(raw: str) -> List[List[str]]:
    blocks: List[List[str]] = []
    parts = raw.split("GOLD (YOU EDIT BELOW THIS LINE):")
    for p in parts[1:]:
        chunk = p.split("=== SAMPLE", 1)[0].strip()
        chunk = chunk.replace("[PASTE SNIPPET HERE AND ADD NEWLINES FOR SENTENCE BREAKS]", "").strip()
        if chunk:
            lines = [ln.strip() for ln in chunk.splitlines() if ln.strip()]
            blocks.append(lines)
    return blocks


# def gold_boundaries_and_text(gold_lines: List[str]) -> Tuple[str, Set[int]]:
#     eval_text = ""
#     bounds: Set[int] = set()
#     pos = 0
#     for i, sent in enumerate(gold_lines):
#         if i > 0:
#             bounds.add(pos)
#             eval_text += " "
#             pos += 1
#         eval_text += sent
#         pos += len(sent)
#     return eval_text, bounds

def gold_boundaries_and_text(gold_lines: List[str]) -> Tuple[str, Set[int]]:
    eval_text = ""
    bounds: Set[int] = set()
    pos = 0
    for i, sent in enumerate(gold_lines):
        if i > 0:
            eval_text += " "
            pos += 1
        eval_text += sent
        pos += len(sent)
        if i < len(gold_lines) - 1:
            bounds.add(pos)  # boundary at end of this sentence
    return eval_text, bounds



def pred_boundaries(eval_text: str) -> Optional[Set[int]]:
    sents = segment_sentences(eval_text)
    bounds: Set[int] = set()
    pos = 0
    for s in sents[:-1]:
        idx = eval_text.find(s, pos)
        if idx == -1:
            return None
        end = idx + len(s)
        bounds.add(end)
        pos = end
    return bounds


def run_sentseg_eval(cfg: SentSegEvalConfig) -> dict:
    if not cfg.gold_path.exists():
        raise FileNotFoundError(f"Missing gold file: {cfg.gold_path}")

    raw = cfg.gold_path.read_text(encoding="utf-8")
    gold_blocks = parse_gold_blocks(raw)

    if not gold_blocks:
        raise ValueError("No GOLD blocks found. Ensure GOLD sections contain newline-separated sentences.")

    tp = fp = fn = 0
    per_sample = []

    for i, gold_lines in enumerate(gold_blocks, start=1):
        gold_lines = [remove_timestamp_lines(x).strip() for x in gold_lines if x.strip()]

        eval_text, gold_bounds = gold_boundaries_and_text(gold_lines)
        pb = pred_boundaries(eval_text)
        if pb is None:
            per_sample.append((i, "PRED_TEXT_MISMATCH"))
            continue

        tp_i = len(gold_bounds & pb)
        fp_i = len(pb - gold_bounds)
        fn_i = len(gold_bounds - pb)

        tp += tp_i
        fp += fp_i
        fn += fn_i
        per_sample.append((i, tp_i, fp_i, fn_i))

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "samples": len(gold_blocks),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "per_sample": per_sample,
    }


def format_sentseg_eval_report(r: dict) -> str:
    lines = []
    lines.append("=== TASK 4 EVALUATION: Sentence Boundary Detection (v2) ===\n")
    lines.append(f"Samples evaluated: {r['samples']}\n")
    lines.append(f"TP={r['tp']}  FP={r['fp']}  FN={r['fn']}\n")
    lines.append(f"Precision: {r['precision']:.4f}\n")
    lines.append(f"Recall:    {r['recall']:.4f}\n")
    lines.append(f"F1:        {r['f1']:.4f}\n")
    lines.append("\nPer-sample results (sample, TP, FP, FN):\n")
    for row in r["per_sample"]:
        lines.append(f"{row}\n")
    return "".join(lines)
