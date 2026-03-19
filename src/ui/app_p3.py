#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import streamlit as st

from nlp_project.common.config import load_yaml
from nlp_project.p3.task1_dataset import P3Task1DatasetConfig, run_p3_task1_dataset
from nlp_project.p3.task2_word2vec import P3Task2Word2VecConfig, run_p3_task2_word2vec
from nlp_project.p3.task3_glove import P3Task3GloveConfig, run_p3_task3_glove
from nlp_project.p3.task4_compare import P3Task4CompareConfig, run_p3_task4_compare
from nlp_project.p3.task5_classify import (
    P3Task5ClassifyConfig,
    run_p3_task5_classify,
    build_p3_task5_predictor,
    predict_with_p3_task5,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"


def _to_path(x: Any) -> Any:
    if isinstance(x, str) and ("/" in x or x.endswith(".json") or x.endswith(".jsonl") or x.endswith(".csv") or x.endswith(".png")):
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


def load_json_report(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def header():
    st.set_page_config(page_title="P3 NLP Demo", layout="wide")
    st.title("Project 3 NLP Demo")
    st.caption("Interactive view for dataset analysis, embeddings, comparison, and classification.")


def _load_word2vec_model(model_path: Path):
    from gensim.models import Word2Vec

    if not model_path.exists():
        raise FileNotFoundError(model_path)
    return Word2Vec.load(str(model_path))


def _load_glove_vectors(path: Path) -> Dict[str, list[float]]:
    import numpy as np

    vectors: Dict[str, list[float]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip().split()
            if len(parts) < 3:
                continue
            try:
                vec = np.array(parts[1:], dtype=float)
            except ValueError:
                continue
            vectors[parts[0]] = vec
    return vectors


def _cosine_similarity(v1, v2) -> float:
    import numpy as np

    denom = float(np.linalg.norm(v1) * np.linalg.norm(v2))
    if denom == 0.0:
        return 0.0
    return float(np.dot(v1, v2) / denom)


def _glove_most_similar(word: str, vectors: Dict[str, Any], topn: int = 10):
    if word not in vectors:
        return None
    target = vectors[word]
    scores = []
    for other_word, vec in vectors.items():
        if other_word == word:
            continue
        scores.append((other_word, _cosine_similarity(target, vec)))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:topn]


def _glove_analogy(positive: list[str], negative: list[str], vectors: Dict[str, Any], topn: int = 5):
    import numpy as np

    missing = [w for w in positive + negative if w not in vectors]
    if missing:
        return None, missing
    result = np.zeros_like(next(iter(vectors.values())))
    for w in positive:
        result += vectors[w]
    for w in negative:
        result -= vectors[w]
    banned = set(positive + negative)
    scores = []
    for word, vec in vectors.items():
        if word in banned:
            continue
        scores.append((word, _cosine_similarity(result, vec)))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:topn], []


def page_task1():
    cfg_dict, cfg_path = sidebar_config_loader("task_p3_dataset")
    st.subheader("Task 1: Dataset Description and Matrices")
    st.caption(f"Config: {cfg_path.name}")

    cfg_dict["top_n"] = int(st.sidebar.slider("task1_top_n", 10, 30, int(cfg_dict["top_n"])))
    cfg_dict["tdm_max_docs_visual"] = int(
        st.sidebar.slider("task1_tdm_docs", 20, 100, int(cfg_dict["tdm_max_docs_visual"]))
    )

    if st.sidebar.button("Run Task 1"):
        cfg = build_cfg(P3Task1DatasetConfig, cfg_dict)
        with st.spinner("Running Task 1..."):
            r = run_p3_task1_dataset(cfg)
        st.session_state["p3_task1_result"] = r

    r = st.session_state.get("p3_task1_result")
    if r is None:
        r = load_json_report(Path(cfg_dict["out_json"]))

    if r:
        ds = r["dataset"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Documents", ds["documents"])
        c2.metric("Total Words", ds["total_words"])
        c3.metric("Distinct Words", ds["distinct_words"])
        c4.metric("Rare Words", ds["rare_words_freq_1"])

        st.subheader("Top Raw Words")
        st.dataframe(pd.DataFrame(r["top_words_raw"]), use_container_width=True)
        st.subheader("Top Filtered Words")
        st.dataframe(pd.DataFrame(r["top_words_filtered"]), use_container_width=True)

        art = r["artifacts"]
        c1, c2 = st.columns(2)
        with c1:
            if Path(art["top_words_raw_png"]).exists():
                st.image(art["top_words_raw_png"], caption="Top raw words")
            if Path(art["tdm_raw_png"]).exists():
                st.image(art["tdm_raw_png"], caption="Term-document matrix (raw)")
            if Path(art["wwm_raw_png"]).exists():
                st.image(art["wwm_raw_png"], caption="Word-word matrix (raw)")
        with c2:
            if Path(art["top_words_filtered_png"]).exists():
                st.image(art["top_words_filtered_png"], caption="Top filtered words")
            if Path(art["tdm_filtered_png"]).exists():
                st.image(art["tdm_filtered_png"], caption="Term-document matrix (filtered)")
            if Path(art["wwm_filtered_png"]).exists():
                st.image(art["wwm_filtered_png"], caption="Word-word matrix (filtered)")

        file_download(Path(cfg_dict["out_json"]), "Download Task 1 JSON")


def page_task2_word2vec():
    cfg_dict, cfg_path = sidebar_config_loader("task_p3_word2vec")
    st.subheader("Task 2: Word2Vec")
    st.caption(f"Config: {cfg_path.name}")

    if st.sidebar.button("Run Task 2"):
        cfg = build_cfg(P3Task2Word2VecConfig, cfg_dict)
        with st.spinner("Training Word2Vec..."):
            r = run_p3_task2_word2vec(cfg)
        st.session_state["p3_task2_result"] = r

    r = st.session_state.get("p3_task2_result")
    if r is None:
        r = load_json_report(Path(cfg_dict["out_json"]))

    if r:
        ds = r["dataset"]
        st.write(
            f"Documents: **{ds['documents']}**, sequences: **{ds['training_sequences']}**, "
            f"trained vocab: **{ds['trained_vocabulary']}**"
        )

        rows = []
        for word, sims in r["similar_words"].items():
            if not sims:
                rows.append({"query_word": word, "similar_word": "OOV_NOT_IN_VOCAB", "score": None})
            else:
                for item in sims[:5]:
                    rows.append({"query_word": word, "similar_word": item["word"], "score": item["score"]})
        st.subheader("Saved Similar Words")
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

        st.subheader("Try Your Own Word")
        model_path = Path(r["artifacts"]["model_path"])
        query_word = st.text_input("Word2Vec query word", value="qazaqstan")
        if st.button("Find Similar Words (Word2Vec)"):
            if not query_word.strip():
                st.warning("Please enter a word.")
            else:
                try:
                    model = _load_word2vec_model(model_path)
                    if query_word not in model.wv:
                        st.warning(f"'{query_word}' is not in the Word2Vec vocabulary.")
                    else:
                        sims = model.wv.most_similar(query_word, topn=10)
                        st.dataframe(pd.DataFrame(sims, columns=["word", "score"]), use_container_width=True)
                except Exception as e:
                    st.error(str(e))

        st.subheader("Try Your Own Analogy")
        pos_words = st.text_input("Positive words (comma-separated)", value="tramp, qazaqstan")
        neg_words = st.text_input("Negative words (comma-separated)", value="aqsh")
        if st.button("Run Word2Vec Analogy"):
            try:
                model = _load_word2vec_model(model_path)
                positive = [x.strip() for x in pos_words.split(",") if x.strip()]
                negative = [x.strip() for x in neg_words.split(",") if x.strip()]
                missing = [w for w in positive + negative if w not in model.wv]
                if missing:
                    st.warning(f"Missing words: {missing}")
                else:
                    sims = model.wv.most_similar(positive=positive, negative=negative, topn=10)
                    st.dataframe(pd.DataFrame(sims, columns=["word", "score"]), use_container_width=True)
            except Exception as e:
                st.error(str(e))

        file_download(Path(cfg_dict["out_json"]), "Download Task 2 JSON")


def page_task3_glove():
    cfg_dict, cfg_path = sidebar_config_loader("task_p3_glove")
    st.subheader("Task 3: GloVe")
    st.caption(f"Config: {cfg_path.name}")

    if st.sidebar.button("Run Task 3"):
        cfg = build_cfg(P3Task3GloveConfig, cfg_dict)
        with st.spinner("Training GloVe..."):
            r = run_p3_task3_glove(cfg)
        st.session_state["p3_task3_result"] = r

    r = st.session_state.get("p3_task3_result")
    if r is None:
        r = load_json_report(Path(cfg_dict["out_json"]))

    if r:
        ds = r["dataset"]
        st.write(
            f"Corpus lines: **{ds['corpus_lines']}**, total tokens: **{ds['total_tokens']}**, "
            f"trained vocab: **{ds['trained_vocabulary']}**"
        )

        rows = []
        for word, sims in r["similar_words"].items():
            if not sims:
                rows.append({"query_word": word, "similar_word": "OOV_NOT_IN_VOCAB", "score": None})
            else:
                for item in sims[:5]:
                    rows.append({"query_word": word, "similar_word": item["word"], "score": item["score"]})
        st.subheader("Saved Similar Words")
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

        vectors_path = Path(r["artifacts"]["vectors_path"])
        query_word = st.text_input("GloVe query word", value="qazaqstan")
        if st.button("Find Similar Words (GloVe)"):
            if not query_word.strip():
                st.warning("Please enter a word.")
            else:
                try:
                    vectors = _load_glove_vectors(vectors_path)
                    sims = _glove_most_similar(query_word, vectors, topn=10)
                    if sims is None:
                        st.warning(f"'{query_word}' is not in the GloVe vocabulary.")
                    else:
                        st.dataframe(pd.DataFrame(sims, columns=["word", "score"]), use_container_width=True)
                except Exception as e:
                    st.error(str(e))

        st.subheader("Try Your Own Analogy")
        pos_words = st.text_input("GloVe positive words", value="tramp, qazaqstan")
        neg_words = st.text_input("GloVe negative words", value="aqsh")
        if st.button("Run GloVe Analogy"):
            try:
                vectors = _load_glove_vectors(vectors_path)
                positive = [x.strip() for x in pos_words.split(",") if x.strip()]
                negative = [x.strip() for x in neg_words.split(",") if x.strip()]
                sims, missing = _glove_analogy(positive, negative, vectors, topn=10)
                if missing:
                    st.warning(f"Missing words: {missing}")
                else:
                    st.dataframe(pd.DataFrame(sims, columns=["word", "score"]), use_container_width=True)
            except Exception as e:
                st.error(str(e))

        file_download(Path(cfg_dict["out_json"]), "Download Task 3 JSON")


def page_task4_compare():
    cfg_dict, cfg_path = sidebar_config_loader("task_p3_compare")
    st.subheader("Task 4: Word2Vec vs GloVe")
    st.caption(f"Config: {cfg_path.name}")

    if st.sidebar.button("Run Task 4"):
        cfg = build_cfg(P3Task4CompareConfig, cfg_dict)
        with st.spinner("Running comparison..."):
            r = run_p3_task4_compare(cfg)
        st.session_state["p3_task4_result"] = r

    r = st.session_state.get("p3_task4_result")
    if r is None:
        r = load_json_report(Path(cfg_dict["out_json"]))

    if r:
        st.write(f"Average top-k overlap: **{r['summary']['average_jaccard_overlap']:.4f}**")
        st.subheader("Per-word Overlap")
        st.dataframe(pd.DataFrame(r["per_word_overlap"]), use_container_width=True)
        st.subheader("Pairwise Similarity Comparison")
        st.dataframe(pd.DataFrame(r["pairwise_similarity_comparison"]), use_container_width=True)
        file_download(Path(cfg_dict["out_json"]), "Download Task 4 JSON")


def page_task5_classify():
    cfg_dict, cfg_path = sidebar_config_loader("task_p3_classify")
    st.subheader("Task 5: Text Classification")
    st.caption(f"Config: {cfg_path.name}")

    cfg_dict["label_field"] = st.sidebar.selectbox(
        "task5_label_field",
        ["category", "subcategory"],
        index=0 if cfg_dict.get("label_field", "category") == "category" else 1,
    )
    cfg_dict["epochs"] = int(st.sidebar.slider("task5_epochs", 3, 15, int(cfg_dict["epochs"])))
    cfg_dict["hidden_size"] = int(st.sidebar.slider("task5_hidden_size", 32, 128, int(cfg_dict["hidden_size"]), step=32))

    if st.sidebar.button("Run Task 5 Full Comparison"):
        cfg = build_cfg(P3Task5ClassifyConfig, cfg_dict)
        with st.spinner("Running full Task 5 comparison..."):
            r = run_p3_task5_classify(cfg)
        st.session_state["p3_task5_result"] = r

    r = st.session_state.get("p3_task5_result")
    if r is None:
        r = load_json_report(Path(cfg_dict["out_json"]))

    if r:
        ds = r["dataset"]
        st.write(
            f"Label field: **{ds['label_field']}**, labels: **{len(ds['labels'])}**, "
            f"train/dev/test: **{ds['documents_train']}/{ds['documents_dev']}/{ds['documents_test']}**"
        )

        summary_csv = Path(r["artifacts"]["summary_csv"]) if r.get("artifacts") else None
        if summary_csv and summary_csv.exists():
            df = pd.read_csv(summary_csv)
            st.subheader("Comparison Table")
            st.dataframe(df, use_container_width=True)
            top_df = df.sort_values("test_f1_macro", ascending=False).head(10)
            st.subheader("Top Results by Test Macro F1")
            st.dataframe(top_df, use_container_width=True)

        st.subheader("Build Interactive Predictor")
        feature_name = st.selectbox("Feature", ["count", "tfidf", "pmi", "word2vec", "glove"], key="p3_task5_feature")
        model_name = st.selectbox(
            "Model",
            ["logreg_baseline", "rnn", "birnn", "lstm"],
            key="p3_task5_model",
        )

        if st.button("Build Selected Predictor"):
            cfg = build_cfg(P3Task5ClassifyConfig, cfg_dict)
            with st.spinner(f"Training {model_name} on {feature_name}..."):
                predictor = build_p3_task5_predictor(cfg, feature_name=feature_name, model_name=model_name)
            st.session_state["p3_task5_predictor"] = predictor
            st.success(f"Predictor ready: {feature_name} + {model_name}")

        predictor = st.session_state.get("p3_task5_predictor")
        st.subheader("Try Your Own Text")
        user_text = st.text_area(
            "Paste text to classify",
            height=180,
            placeholder="Example: Qazaqstan men AQSh arasynda halyqaralyq kelisim jasaldy...",
        )
        if st.button("Predict Class"):
            if predictor is None:
                st.warning("Build a predictor first.")
            elif not user_text.strip():
                st.warning("Please enter some text.")
            else:
                pred = predict_with_p3_task5(user_text, predictor)
                st.write(f"Predicted label: **{pred['predicted_label']}**")
                st.write(f"Token count: **{len(pred['tokens'])}**")
                st.write(f"Feature dimension used: **{pred['feature_dim']}**")
                st.dataframe(pd.DataFrame(pred["top_predictions"]), use_container_width=True)

        file_download(Path(cfg_dict["out_json"]), "Download Task 5 JSON")


def main():
    header()

    task = st.sidebar.selectbox(
        "Choose P3 task",
        [
            "Task 1: Dataset + Matrices",
            "Task 2: Word2Vec",
            "Task 3: GloVe",
            "Task 4: Compare Embeddings",
            "Task 5: Classification",
        ],
    )

    with st.expander("Available P3 configs"):
        files = sorted(CONFIG_DIR.glob("task_p3*.yaml"))
        st.write([f.name for f in files])

    with st.expander("How to use this UI"):
        st.markdown(
            "- Run the saved task reports or rerun them from the sidebar.\n"
            "- Use custom word queries for Word2Vec and GloVe.\n"
            "- Build an interactive classifier predictor, then test custom text inputs live.\n"
            "- For professor demos, keep a trained predictor in session and try multiple inputs quickly."
        )

    if task == "Task 1: Dataset + Matrices":
        page_task1()
    elif task == "Task 2: Word2Vec":
        page_task2_word2vec()
    elif task == "Task 3: GloVe":
        page_task3_glove()
    elif task == "Task 4: Compare Embeddings":
        page_task4_compare()
    elif task == "Task 5: Classification":
        page_task5_classify()


if __name__ == "__main__":
    main()
