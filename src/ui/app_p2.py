#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import streamlit as st

from nlp_project.common.config import load_yaml
from nlp_project.p2.task1_ngram import (
    P2Task1NgramConfig,
    run_p2_task1_ngram,
)
from nlp_project.p2.task2_smoothing import (
    P2Task2SmoothingConfig,
    run_p2_task2_smoothing,
)
from nlp_project.p2.task4_dot_lr import (
    P2Task4DotLRConfig,
    run_p2_task4_dot_lr,
    build_p2_task4_dot_lr_predictor,
    predict_sentences_with_p2_task4_dot_lr,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"


def _to_path(x: Any) -> Any:
    if isinstance(x, str) and ("/" in x or x.endswith(".jsonl") or x.endswith(".json")):
        return Path(x)
    return x


def build_cfg(cls, cfg_dict: Dict[str, Any]):
    kwargs = {k: _to_path(v) for k, v in cfg_dict.items()}
    return cls(**kwargs)


def pick_config_file(prefix: str) -> Optional[Path]:
    candidates = sorted(CONFIG_DIR.glob(f"{prefix}*.yaml"))
    return candidates[0] if candidates else None


def sidebar_config_loader(default_prefix: str) -> Tuple[Dict[str, Any], Path]:
    cfg_path = pick_config_file(default_prefix)
    if cfg_path is None:
        st.sidebar.error(f"No config found with prefix '{default_prefix}' in {CONFIG_DIR}.")
        st.stop()
    cfg_dict = load_yaml(cfg_path)
    return cfg_dict, cfg_path


def file_download(path: Path, label: str):
    if path.exists():
        st.download_button(label, path.read_bytes(), file_name=path.name)
    else:
        st.warning(f"File not found: {path}")


def header():
    st.set_page_config(page_title="P2 NLP Demo", layout="wide")
    st.title("Project 2 NLP Demo")
    st.caption("Interactive view for Task 1, Task 2, and Task 4 with adjustable parameters.")


def page_task1_ngram():
    cfg_dict, cfg_path = sidebar_config_loader("task_p2_ngram")
    st.subheader("Task 1: N-gram Models and Perplexity")
    st.caption(f"Config: {cfg_path.name}")
    st.info(
        "Builds unigram, bigram, trigram models on train split and reports perplexity on train/dev/test."
    )

    st.sidebar.markdown("### Task 1 Parameters")
    cfg_dict["train_ratio"] = st.sidebar.slider("train_ratio", 0.6, 0.9, float(cfg_dict["train_ratio"]), 0.01)
    cfg_dict["dev_ratio"] = st.sidebar.slider("dev_ratio", 0.05, 0.2, float(cfg_dict["dev_ratio"]), 0.01)
    cfg_dict["min_count"] = int(st.sidebar.number_input("min_count", min_value=1, value=int(cfg_dict["min_count"])))
    cfg_dict["seed"] = int(st.sidebar.number_input("seed", min_value=0, value=int(cfg_dict["seed"])))
    cfg_dict["top_k"] = int(st.sidebar.slider("top_k", 5, 50, int(cfg_dict["top_k"])))

    if st.sidebar.button("Run Task 1"):
        cfg = build_cfg(P2Task1NgramConfig, cfg_dict)
        with st.spinner("Running Task 1..."):
            r = run_p2_task1_ngram(cfg)
        st.success("Task 1 complete")

        ds = r["dataset"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Train Docs", ds["docs_train"])
        c2.metric("Dev Docs", ds["docs_dev"])
        c3.metric("Test Docs", ds["docs_test"])
        c4.metric("Vocab Size", r["vocab"]["size_with_specials"])

        rows = []
        for m in ("unigram", "bigram", "trigram"):
            for split in ("train", "dev", "test"):
                pr = r["models"][m]["perplexity"][split]
                rows.append(
                    {
                        "model": m,
                        "split": split,
                        "perplexity": pr["perplexity"],
                        "events": pr["events"],
                        "zero_events": pr["zero_events"],
                    }
                )
        df = pd.DataFrame(rows)
        df["perplexity"] = pd.to_numeric(df["perplexity"], errors="coerce").round(4)
        st.subheader("Perplexity Summary")
        st.dataframe(df, use_container_width=True)

        st.subheader("Top Unigrams")
        st.caption("`<UNK>` means rare/unknown words mapped into one token, `</s>` means sentence end token.")
        top_uni = pd.DataFrame(r["models"]["unigram"]["top_ngrams"], columns=["ngram", "count"])
        st.dataframe(top_uni, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Top Bigrams")
            st.dataframe(
                pd.DataFrame(r["models"]["bigram"]["top_ngrams"], columns=["ngram", "count"]),
                use_container_width=True,
            )
        with c2:
            st.subheader("Top Trigrams")
            st.dataframe(
                pd.DataFrame(r["models"]["trigram"]["top_ngrams"], columns=["ngram", "count"]),
                use_container_width=True,
            )
        file_download(cfg.out_json, "Download Task 1 JSON")


def page_task2_smoothing():
    cfg_dict, cfg_path = sidebar_config_loader("task_p2_smoothing")
    st.subheader("Task 2: Smoothing Method Comparison")
    st.caption(f"Config: {cfg_path.name}")
    st.info(
        "Compares Laplace, Interpolation, Backoff, and Kneser-Ney using perplexity. "
        "Lower perplexity is better."
    )

    st.sidebar.markdown("### Task 2 Parameters")
    cfg_dict["train_ratio"] = st.sidebar.slider("train_ratio", 0.6, 0.9, float(cfg_dict["train_ratio"]), 0.01)
    cfg_dict["dev_ratio"] = st.sidebar.slider("dev_ratio", 0.05, 0.2, float(cfg_dict["dev_ratio"]), 0.01)
    cfg_dict["min_count"] = int(st.sidebar.number_input("min_count", min_value=1, value=int(cfg_dict["min_count"])))
    cfg_dict["laplace_alpha"] = float(st.sidebar.slider("laplace_alpha", 0.1, 2.0, float(cfg_dict["laplace_alpha"]), 0.1))
    cfg_dict["backoff_gamma"] = float(st.sidebar.slider("backoff_gamma", 0.1, 0.9, float(cfg_dict["backoff_gamma"]), 0.05))
    cfg_dict["kn_discount"] = float(st.sidebar.slider("kn_discount", 0.1, 1.0, float(cfg_dict["kn_discount"]), 0.05))

    l1 = float(st.sidebar.slider("interp_lambda_1", 0.0, 1.0, float(cfg_dict["interpolation_l1"]), 0.05))
    l2 = float(st.sidebar.slider("interp_lambda_2", 0.0, 1.0, float(cfg_dict["interpolation_l2"]), 0.05))
    rem = max(0.0, 1.0 - l1 - l2)
    st.sidebar.write(f"interp_lambda_3 auto-set to: {rem:.2f}")
    cfg_dict["interpolation_l1"] = l1
    cfg_dict["interpolation_l2"] = l2
    cfg_dict["interpolation_l3"] = rem

    if st.sidebar.button("Run Task 2"):
        cfg = build_cfg(P2Task2SmoothingConfig, cfg_dict)
        with st.spinner("Running Task 2..."):
            r = run_p2_task2_smoothing(cfg)
        st.success("Task 2 complete")

        rows = []
        for method in ("laplace", "interpolation", "backoff", "kneser_ney"):
            for split in ("train", "dev", "test"):
                rows.append(
                    {
                        "method": method,
                        "split": split,
                        "perplexity": r["methods"][method][split]["perplexity"],
                        "events": r["methods"][method][split]["events"],
                    }
                )
        df = pd.DataFrame(rows)
        df["perplexity"] = pd.to_numeric(df["perplexity"], errors="coerce").round(4)
        st.subheader("Perplexity by Method")
        st.dataframe(df, use_container_width=True)
        st.success(f"Best method by dev perplexity: {r['best_method_by_dev_perplexity']}")
        file_download(cfg.out_json, "Download Task 2 JSON")


def page_task4_dot_lr():
    cfg_dict, cfg_path = sidebar_config_loader("task_p2_dot_lr")
    st.subheader("Task 4: Dot as Sentence-End (Logistic Regression)")
    st.caption(f"Config: {cfg_path.name}")
    st.info(
        "Classifies each dot (`.`) as end-of-sentence or not, compares L1 vs L2, "
        "then shows sentence detection preview."
    )

    st.sidebar.markdown("### Task 4 Parameters")
    cfg_dict["train_ratio"] = st.sidebar.slider("train_ratio", 0.6, 0.9, float(cfg_dict["train_ratio"]), 0.01)
    cfg_dict["dev_ratio"] = st.sidebar.slider("dev_ratio", 0.05, 0.2, float(cfg_dict["dev_ratio"]), 0.01)
    cfg_dict["c_l1"] = float(st.sidebar.slider("c_l1", 0.1, 3.0, float(cfg_dict["c_l1"]), 0.1))
    cfg_dict["c_l2"] = float(st.sidebar.slider("c_l2", 0.1, 3.0, float(cfg_dict["c_l2"]), 0.1))
    cfg_dict["max_iter"] = int(st.sidebar.number_input("max_iter", min_value=100, value=int(cfg_dict["max_iter"])))
    cfg_dict["threshold"] = float(st.sidebar.slider("threshold", 0.1, 0.9, float(cfg_dict["threshold"]), 0.05))

    if st.sidebar.button("Run Task 4"):
        cfg = build_cfg(P2Task4DotLRConfig, cfg_dict)
        with st.spinner("Running Task 4..."):
            r = run_p2_task4_dot_lr(cfg)
            predictor = build_p2_task4_dot_lr_predictor(cfg)

        st.session_state["p2_task4_result"] = r
        st.session_state["p2_task4_predictor"] = predictor
        st.success("Task 4 complete")

    if "p2_task4_result" in st.session_state:
        r = st.session_state["p2_task4_result"]

        c1, c2, c3 = st.columns(3)
        c1.metric("Train Dot Examples", r["examples"]["train_dots"])
        c2.metric("Dev Dot Examples", r["examples"]["dev_dots"])
        c3.metric("Test Dot Examples", r["examples"]["test_dots"])

        rows = []
        for reg in ("l1", "l2"):
            for split in ("train", "dev", "test"):
                m = r[reg]["metrics"][split]
                rows.append(
                    {
                        "regularization": reg,
                        "split": split,
                        "accuracy": m["accuracy"],
                        "precision": m["precision"],
                        "recall": m["recall"],
                        "f1": m["f1"],
                    }
                )
        df = pd.DataFrame(rows)
        for col in ("accuracy", "precision", "recall", "f1"):
            df[col] = pd.to_numeric(df[col], errors="coerce").round(4)
        st.subheader("L1 vs L2 Metrics")
        st.dataframe(df, use_container_width=True)

        st.success(f"Best model by dev F1: {r['best_model_by_dev_f1']}")
        if "chosen_threshold" in r:
            st.write(f"Chosen threshold for sentence splitting: **{r['chosen_threshold']:.2f}**")
        st.subheader("Sentence Detection Preview (Best Model)")
        with st.expander("Show preview JSON"):
            st.json(r["sentence_detection_preview_best_model"])

        st.subheader("Try Your Own Text")
        user_text = st.text_area(
            "Paste a paragraph and click 'Detect Boundaries'.",
            height=180,
            placeholder="Example: Bugin kun suyk boldy. Men universitetke bardym. Sabaq qyzqty boldy"
        )
        if st.button("Detect Boundaries"):
            if not user_text.strip():
                st.warning("Please enter some text first.")
            else:
                thr = float(r.get("chosen_threshold", cfg_dict["threshold"]))
                pred = predict_sentences_with_p2_task4_dot_lr(
                    user_text,
                    st.session_state["p2_task4_predictor"],
                    threshold=thr,
                )
                st.write(f"Detected sentences: **{pred['predicted_sentence_count']}**")
                for i, sent in enumerate(pred["sentences"], start=1):
                    st.write(f"{i}. {sent}")

        cfg = build_cfg(P2Task4DotLRConfig, cfg_dict)
        file_download(cfg.out_json, "Download Task 4 JSON")


def main():
    header()

    task = st.sidebar.selectbox(
        "Choose P2 task",
        [
            "Task 1: N-gram + Perplexity",
            "Task 2: Smoothing",
            "Task 4: Dot Logistic Regression",
        ],
    )

    with st.expander("Available P2 configs"):
        files = sorted(CONFIG_DIR.glob("task_p2*.yaml"))
        st.write([f.name for f in files])

    with st.expander("How to read results"):
        st.markdown(
            "- **Perplexity:** lower is better.\n"
            "- **`<UNK>`:** rare words grouped into unknown token.\n"
            "- **`</s>`:** sentence-end marker used for language modeling.\n"
            "- **Task 4 F1:** higher is better."
        )

    if task == "Task 1: N-gram + Perplexity":
        page_task1_ngram()
    elif task == "Task 2: Smoothing":
        page_task2_smoothing()
    elif task == "Task 4: Dot Logistic Regression":
        page_task4_dot_lr()


if __name__ == "__main__":
    main()
