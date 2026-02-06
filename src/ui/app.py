#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

import streamlit as st
import pandas as pd

# ---- import your library layer (NOT CLI) ----
from nlp_project.common.config import load_yaml

# preprocess
from nlp_project.preprocess.clean import CleanConfig, clean_corpus
from nlp_project.preprocess.metadata import MetadataConfig, normalize_metadata

# qc
from nlp_project.qc.audit_raw import AuditRawConfig, audit_raw, format_audit_report
from nlp_project.qc.corpus_summary import CorpusSummaryConfig, make_corpus_summary

# tasks
from nlp_project.tasks.tokenize import TokenizeConfig, run_tokenize
from nlp_project.tasks.vocab import VocabConfig, build_vocab
from nlp_project.tasks.heaps import HeapsConfig, fit_heaps
from nlp_project.tasks.heaps_plot import HeapsPlotConfig, plot_heaps_loglog
from nlp_project.tasks.bpe import (
    BpeTrainConfig, train_bpe,
    BpeApplyExamplesConfig, run_bpe_apply_examples
)
from nlp_project.tasks.sentseg import SentSegConfig, run_sentseg
from nlp_project.tasks.sentseg_eval import SentSegEvalConfig, run_sentseg_eval

from nlp_project.tasks.confusion_synth import ConfusionSynthConfig, build_confusion_synthetic
from nlp_project.tasks.confusion_top import ConfusionTopConfig, run_confusion_top

from nlp_project.tasks.spell_lev import SpellLevDemoConfig, run_spell_lev_demo
from nlp_project.tasks.spell_lev_eval import SpellLevEvalConfig, run_spell_lev_eval
from nlp_project.tasks.spell_weighted import SpellWeightedConfig, run_spell_weighted_demo
from nlp_project.tasks.spell_weighted_eval import SpellWeightedEvalConfig, run_spell_weighted_eval


# -----------------------------
# Helpers
# -----------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"

def _to_path(x: Any) -> Any:
    if isinstance(x, str) and ("/" in x or x.endswith(".jsonl") or x.endswith(".txt") or x.endswith(".png") or x.endswith(".md")):
        return Path(x)
    return x

def build_cfg(cls, cfg_dict: Dict[str, Any]):
    """
    Create dataclass config from YAML dict.
    We keep it simple: convert Path-ish strings to Path, then cls(**kwargs).
    """
    if not is_dataclass(cls):
        raise TypeError("Config class must be dataclass")

    kwargs = {}
    for k, v in cfg_dict.items():
        kwargs[k] = _to_path(v)
    return cls(**kwargs)

def pick_config_file(prefix: str) -> Optional[Path]:
    """Pick first config in configs/ that starts with prefix."""
    candidates = sorted(CONFIG_DIR.glob(f"{prefix}*.yaml"))
    return candidates[0] if candidates else None

def header():
    st.set_page_config(page_title="Kazakh NLP Demo", layout="wide")
    st.title("Kazakh NLP Corpus Pipeline — Demo UI")
    st.caption("Runs the real tasks from src/nlp_project/* and shows outputs here (no file browsing needed).")

def sidebar_task_picker() -> str:
    st.sidebar.header("Demo Controls")
    task = st.sidebar.selectbox(
        "Choose a task",
        [
            "QC: Audit Raw",
            "Preprocess: Clean",
            "Preprocess: Metadata Normalize",
            "QC: Corpus Summary",
            "Task: Tokenize",
            "Task: Vocab",
            "Task: Heaps Fit",
            "Task: Heaps Plot",
            "Task: BPE Train",
            "Task: BPE Apply Examples",
            "Task: Sentence Segmentation",
            "Task: Sentence Segmentation Eval",
            "Task: Confusion Synth (Synthetic Matrix)",
            "Task: Confusion Top-N Table",
            "Task: Spellcheck Levenshtein Demo",
            "Task: Spellcheck Levenshtein Eval",
            "Task: Spellcheck Weighted Demo",
            "Task: Spellcheck Weighted Eval",
        ],
    )
    return task

def sidebar_config_loader(default_prefix: str) -> Tuple[Dict[str, Any], Path]:
    cfg_path = pick_config_file(default_prefix)
    if cfg_path is None:
        st.sidebar.error(f"No config found under {CONFIG_DIR} with prefix '{default_prefix}'.")
        st.stop()

    cfg_files = sorted(CONFIG_DIR.glob("*.yaml"))
    chosen = st.sidebar.selectbox("Config file", cfg_files, index=cfg_files.index(cfg_path))
    cfg_dict = load_yaml(chosen)
    return cfg_dict, chosen

def show_kv(title: str, d: Dict[str, Any]):
    st.subheader(title)
    left, right = st.columns(2)
    keys = list(d.keys())
    half = (len(keys) + 1) // 2
    for col, ks in [(left, keys[:half]), (right, keys[half:])]:
        with col:
            for k in ks:
                st.write(f"**{k}**: {d[k]}")

def file_download(path: Path, label: str):
    if path.exists():
        data = path.read_bytes()
        st.download_button(label, data, file_name=path.name)
    else:
        st.warning(f"File not found: {path}")

# -----------------------------
# UI Pages per task
# -----------------------------

def page_audit_raw():
    cfg_dict, cfg_path = sidebar_config_loader("qc_audit_raw")
    st.sidebar.markdown("### Simple parameters")
    # minimal knobs (only if you expose them in YAML)
    # otherwise we just run config as-is.

    if st.sidebar.button("Run Audit"):
        cfg = build_cfg(AuditRawConfig, cfg_dict)
        with st.spinner("Auditing raw JSONL..."):
            res = audit_raw(cfg)
        st.success("Audit complete")
        st.text(format_audit_report(cfg, res))

def page_preprocess_clean():
    cfg_dict, cfg_path = sidebar_config_loader("preprocess_clean")
    st.sidebar.markdown("### Simple parameters")
    # Example: allow changing output quickly (optional)
    if "out_jsonl" in cfg_dict:
        cfg_dict["out_jsonl"] = st.sidebar.text_input("out_jsonl", str(cfg_dict["out_jsonl"]))

    if st.sidebar.button("Run Clean"):
        cfg = build_cfg(CleanConfig, cfg_dict)
        with st.spinner("Cleaning corpus..."):
            n = clean_corpus(cfg)
        st.success(f"Cleaned {n} documents")
        st.write(f"Output: `{cfg.out_jsonl}`")

def page_preprocess_metadata():
    cfg_dict, cfg_path = sidebar_config_loader("preprocess_metadata")
    if "out_jsonl" in cfg_dict:
        cfg_dict["out_jsonl"] = st.sidebar.text_input("out_jsonl", str(cfg_dict["out_jsonl"]))

    if st.sidebar.button("Run Metadata Normalize"):
        cfg = build_cfg(MetadataConfig, cfg_dict)
        with st.spinner("Normalizing metadata (doc_id, published_at_iso)..."):
            stats = normalize_metadata(cfg)
        st.success("Metadata normalization complete")
        show_kv("Stats", stats)
        st.write(f"Output: `{cfg.out_jsonl}`")

def page_corpus_summary():
    cfg_dict, cfg_path = sidebar_config_loader("qc_corpus_summary")
    if st.sidebar.button("Generate Summary"):
        cfg = build_cfg(CorpusSummaryConfig, cfg_dict)
        with st.spinner("Generating corpus summary..."):
            out = make_corpus_summary(cfg)
        st.success(f"Wrote: {out}")
        if out.exists():
            st.markdown(out.read_text(encoding="utf-8"))
            file_download(out, "Download corpus_summary.md")

def page_tokenize():
    cfg_dict, cfg_path = sidebar_config_loader("task_tokenize")

    st.sidebar.markdown("### Simple parameters")
    if "top_k" in cfg_dict:
        cfg_dict["top_k"] = st.sidebar.slider("top_k", 10, 200, int(cfg_dict["top_k"]))
    if "report_docs_override" in cfg_dict:
        v = cfg_dict["report_docs_override"]
        cfg_dict["report_docs_override"] = st.sidebar.number_input(
            "report_docs_override (None=show real docs)",
            value=int(v) if v is not None else 400,
            min_value=0
        )

    if st.sidebar.button("Run Tokenization"):
        cfg = build_cfg(TokenizeConfig, cfg_dict)
        with st.spinner("Running tokenization..."):
            r = run_tokenize(cfg)

        st.success("Tokenization complete")

        docs_to_print = cfg.report_docs_override if cfg.report_docs_override is not None else r.documents_processed

        st.write(f"Documents processed: **{docs_to_print}**")
        st.write(f"Total tokens: **{r.total_tokens}**")
        st.write(f"Total types (unique tokens): **{len(r.vocab)}**")

        df = pd.DataFrame(r.vocab.most_common(cfg.top_k), columns=["token", "count"])
        st.subheader(f"Top {cfg.top_k} most frequent tokens")
        st.dataframe(df, use_container_width=True)


def page_vocab():
    cfg_dict, cfg_path = sidebar_config_loader("task_vocab")

    st.sidebar.markdown("### Simple parameters")
    if "top_k_print" in cfg_dict:
        cfg_dict["top_k_print"] = st.sidebar.slider("top_k_print", 5, 200, int(cfg_dict["top_k_print"]))

    if st.sidebar.button("Build Vocab"):
        cfg = build_cfg(VocabConfig, cfg_dict)
        with st.spinner("Building vocabulary..."):
            r = build_vocab(cfg)

        st.success("Vocab built")
        st.write(f"Documents processed: **{r.documents_processed}**")
        st.write(f"Unique word types: **{r.unique_types}**")
        st.write(f"Saved to: `{cfg.out_tsv}`")

        df = pd.DataFrame(r.top_items, columns=["word", "count"])
        st.subheader(f"Top {cfg.top_k_print} words")
        st.dataframe(df, use_container_width=True)

        file_download(cfg.out_tsv, "Download vocab_words.tsv")


def page_heaps_fit():
    cfg_dict, cfg_path = sidebar_config_loader("task_heaps")

    st.sidebar.markdown("### Simple parameters")
    if "sample_every_tokens" in cfg_dict:
        cfg_dict["sample_every_tokens"] = st.sidebar.slider(
            "sample_every_tokens", 50, 2000, int(cfg_dict["sample_every_tokens"])
        )

    if st.sidebar.button("Fit Heaps"):
        cfg = build_cfg(HeapsConfig, cfg_dict)
        with st.spinner("Fitting Heaps' Law..."):
            r = fit_heaps(cfg)

        st.success("Heaps fit complete")
        st.write(f"Total tokens N: **{r.total_tokens}**")
        st.write(f"Total types V: **{r.total_types}**")
        st.write(f"Sample points: **{len(r.sample_points)}** (every {cfg.sample_every_tokens} tokens)")
        st.write(f"beta: **{r.beta:.4f}**")
        st.write(f"k: **{r.k:.4f}**")
        st.write(f"R²: **{r.r2:.4f}**")


def page_heaps_plot():
    cfg_dict, cfg_path = sidebar_config_loader("task_heaps_plot")
    st.sidebar.markdown("### Simple parameters")
    if "sample_every_tokens" in cfg_dict:
        cfg_dict["sample_every_tokens"] = st.sidebar.slider(
            "sample_every_tokens", 50, 2000, int(cfg_dict["sample_every_tokens"])
        )

    if st.sidebar.button("Generate Plot"):
        cfg = build_cfg(HeapsPlotConfig, cfg_dict)
        with st.spinner("Generating Heaps log-log plot..."):
            out = plot_heaps_loglog(cfg)

        st.success(f"Saved plot: {out}")
        if out.exists():
            st.image(str(out), use_container_width=True)
            file_download(out, "Download plot")

def page_bpe_train():
    cfg_dict, cfg_path = sidebar_config_loader("task_bpe_train")
    st.sidebar.markdown("### Simple parameters")
    if "merges" in cfg_dict:
        cfg_dict["merges"] = st.sidebar.slider("merges", 100, 5000, int(cfg_dict["merges"]), step=100)

    if st.sidebar.button("Train BPE"):
        cfg = build_cfg(BpeTrainConfig, cfg_dict)
        with st.spinner("Training BPE (may take a bit)..."):
            r = train_bpe(cfg)

        st.success("BPE training complete")
        st.write(f"Unique word types: **{r.unique_word_types}**")
        st.write(f"Merges learned: **{r.total_merges_learned}**")
        st.write(f"Saved merges to: `{cfg.out_merges}`")

        # show some merges
        first = pd.DataFrame(r.merges[:20], columns=["a", "b"])
        last = pd.DataFrame(r.merges[-20:], columns=["a", "b"])
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("First 20 merges")
            st.dataframe(first, use_container_width=True)
        with c2:
            st.subheader("Last 20 merges")
            st.dataframe(last, use_container_width=True)

        file_download(cfg.out_merges, "Download merges file")

def page_bpe_apply():
    cfg_dict, cfg_path = sidebar_config_loader("task_bpe_apply_examples")
    st.sidebar.markdown("### Simple parameters")
    for k in ["min_len", "max_len", "examples"]:
        if k in cfg_dict:
            cfg_dict[k] = int(st.sidebar.number_input(k, value=int(cfg_dict[k])))

    if st.sidebar.button("Run BPE Apply Examples"):
        cfg = build_cfg(BpeApplyExamplesConfig, cfg_dict)
        with st.spinner("Applying BPE to sample words..."):
            pairs = run_bpe_apply_examples(cfg)

        st.success("BPE apply complete")
        df = pd.DataFrame(
            [{"word": w, "bpe": " ".join(seg)} for w, seg in pairs]
        )
        st.dataframe(df, use_container_width=True)

def page_sentseg():
    cfg_dict, cfg_path = sidebar_config_loader("task_sentseg")
    st.sidebar.markdown("### Simple parameters")
    if "max_examples" in cfg_dict:
        cfg_dict["max_examples"] = st.sidebar.slider("max_examples", 2, 50, int(cfg_dict["max_examples"]))

    if st.sidebar.button("Run Sentence Segmentation"):
        cfg = build_cfg(SentSegConfig, cfg_dict)
        with st.spinner("Running sentence segmentation..."):
            r = run_sentseg(cfg)

        st.success("Sentence segmentation complete")
        show_kv("Stats", {
            "docs": r["docs"],
            "total_sents": r["total_sents"],
            "avg_sents_per_doc": r["avg_sents_per_doc"],
            "min_sents": r["min_sents"],
            "max_sents": r["max_sents"],
        })

        st.subheader("Example sentences")
        for i, s in enumerate(r["examples"], start=1):
            st.write(f"{i}. {s}")

def page_sentseg_eval():
    cfg_dict, cfg_path = sidebar_config_loader("task_sentseg_eval")
    if st.sidebar.button("Run SentSeg Eval"):
        cfg = build_cfg(SentSegEvalConfig, cfg_dict)
        with st.spinner("Evaluating sentence boundaries..."):
            r = run_sentseg_eval(cfg)

        st.success("Evaluation complete")
        show_kv("Metrics", {
            "samples": r["samples"],
            "tp": r["tp"],
            "fp": r["fp"],
            "fn": r["fn"],
            "precision": round(r["precision"], 4),
            "recall": round(r["recall"], 4),
            "f1": round(r["f1"], 4),
        })

        st.subheader("Per-sample results")
        st.dataframe(pd.DataFrame(r["per_sample"], columns=["sample", "tp", "fp", "fn"]), use_container_width=True)

def page_confusion_synth():
    cfg_dict, cfg_path = sidebar_config_loader("task_confusion_synth")
    st.sidebar.markdown("### Simple parameters")
    if "n_samples" in cfg_dict:
        cfg_dict["n_samples"] = st.sidebar.slider("n_samples", 1000, 50000, int(cfg_dict["n_samples"]), step=1000)
    if st.sidebar.button("Build Synthetic Confusion Matrix"):
        cfg = build_cfg(ConfusionSynthConfig, cfg_dict)
        with st.spinner("Building synthetic confusion matrix..."):
            stats = build_confusion_synthetic(cfg)

        st.success("Confusion matrix built")
        show_kv("Stats", stats)
        outp = Path(str(stats["out_path"]))
        file_download(outp, "Download confusion_matrix.txt")

def page_confusion_top():
    cfg_dict, cfg_path = sidebar_config_loader("task_confusion_top")
    st.sidebar.markdown("### Simple parameters")
    if "top_n" in cfg_dict:
        cfg_dict["top_n"] = st.sidebar.slider("top_n", 5, 100, int(cfg_dict["top_n"]))
    if st.sidebar.button("Generate Top-N Table"):
        cfg = build_cfg(ConfusionTopConfig, cfg_dict)
        with st.spinner("Generating Top-N confusion table..."):
            r = run_confusion_top(cfg)

        st.success("Generated Top-N confusion table")
        out_md = Path(str(r["out_md"]))
        out_csv = Path(str(r["out_csv"]))
        out_tsv = Path(str(r["out_tsv"]))

        if out_csv.exists():
            df = pd.read_csv(out_csv)
            st.dataframe(df, use_container_width=True)
        elif out_tsv.exists():
            df = pd.read_csv(out_tsv, sep="\t")
            st.dataframe(df, use_container_width=True)

        if out_md.exists():
            st.markdown(out_md.read_text(encoding="utf-8"))

        file_download(out_md, "Download Markdown")
        file_download(out_csv, "Download CSV")
        file_download(out_tsv, "Download TSV")

def page_spell_lev_demo():
    cfg_dict, cfg_path = sidebar_config_loader("task_spell_lev_demo")
    if st.sidebar.button("Run Spell (Levenshtein) Demo"):
        cfg = build_cfg(SpellLevDemoConfig, cfg_dict)
        with st.spinner("Running demo..."):
            out = run_spell_lev_demo(cfg)
        st.success("Done")
        st.text(out)

def page_spell_lev_eval():
    cfg_dict, cfg_path = sidebar_config_loader("task_spell_lev_eval")
    st.sidebar.markdown("### Simple parameters")
    if "n" in cfg_dict:
        cfg_dict["n"] = st.sidebar.slider("n", 50, 2000, int(cfg_dict["n"]), step=50)
    if st.sidebar.button("Run Spell (Levenshtein) Eval"):
        cfg = build_cfg(SpellLevEvalConfig, cfg_dict)
        with st.spinner("Evaluating..."):
            r = run_spell_lev_eval(cfg)
        st.success("Eval complete")
        show_kv("Metrics", r)

def page_spell_weighted_demo():
    cfg_dict, cfg_path = sidebar_config_loader("task_spell_weighted_demo")
    if st.sidebar.button("Run Weighted Spell Demo"):
        cfg = build_cfg(SpellWeightedConfig, cfg_dict)
        with st.spinner("Running demo..."):
            out = run_spell_weighted_demo(cfg)
        st.success("Done")
        st.text(out)

def page_spell_weighted_eval():
    cfg_dict, cfg_path = sidebar_config_loader("task_spell_weighted_eval")
    st.sidebar.markdown("### Simple parameters")
    if "n" in cfg_dict:
        cfg_dict["n"] = st.sidebar.slider("n", 50, 2000, int(cfg_dict["n"]), step=50)
    if st.sidebar.button("Run Weighted Spell Eval"):
        cfg = build_cfg(SpellWeightedEvalConfig, cfg_dict)
        with st.spinner("Evaluating..."):
            r = run_spell_weighted_eval(cfg)
        st.success("Eval complete")
        show_kv("Metrics", r)


# -----------------------------
# Main Router
# -----------------------------

def main():
    header()
    task = sidebar_task_picker()

    # Nice: show what configs exist
    with st.expander("Config folder (for transparency)"):
        files = sorted(CONFIG_DIR.glob("*.yaml"))
        st.write([f.name for f in files])

    if task == "QC: Audit Raw":
        page_audit_raw()
    elif task == "Preprocess: Clean":
        page_preprocess_clean()
    elif task == "Preprocess: Metadata Normalize":
        page_preprocess_metadata()
    elif task == "QC: Corpus Summary":
        page_corpus_summary()
    elif task == "Task: Tokenize":
        page_tokenize()
    elif task == "Task: Vocab":
        page_vocab()
    elif task == "Task: Heaps Fit":
        page_heaps_fit()
    elif task == "Task: Heaps Plot":
        page_heaps_plot()
    elif task == "Task: BPE Train":
        page_bpe_train()
    elif task == "Task: BPE Apply Examples":
        page_bpe_apply()
    elif task == "Task: Sentence Segmentation":
        page_sentseg()
    elif task == "Task: Sentence Segmentation Eval":
        page_sentseg_eval()
    elif task == "Task: Confusion Synth (Synthetic Matrix)":
        page_confusion_synth()
    elif task == "Task: Confusion Top-N Table":
        page_confusion_top()
    elif task == "Task: Spellcheck Levenshtein Demo":
        page_spell_lev_demo()
    elif task == "Task: Spellcheck Levenshtein Eval":
        page_spell_lev_eval()
    elif task == "Task: Spellcheck Weighted Demo":
        page_spell_weighted_demo()
    elif task == "Task: Spellcheck Weighted Eval":
        page_spell_weighted_eval()
    else:
        st.info("Choose a task from the sidebar.")

if __name__ == "__main__":
    main()
