import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import binomtest, ttest_rel
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from load_dataset import load_imdb_dataset

DEFAULT_STOPWORDS = [
    "the", "and", "of", "to", "is", "in", "it", "this", "that", "as",
    "with", "for", "on", "but", "are", "you", "was", "not",
]

MODEL_ORDER = [
    "Multinomial NB (BoW)",
    "Bernoulli NB (BoW)",
    "Logistic Regression (BoW)",
    "Logistic Regression (Lexicon)",
]


def parse_stopwords(text: str):
    return [w.strip().lower() for w in text.split(",") if w.strip()]


@st.cache_data(show_spinner=False)
def load_selected_dataset(use_cleaned: bool):
    if use_cleaned:
        return pd.read_csv("IMDB_Cleaned_Dataset.csv")
    return load_imdb_dataset(csv_path="IMDB Dataset.csv", save_cleaned=False)


def build_vader_matrix(texts):
    analyzer = SentimentIntensityAnalyzer()

    def scores_for(text):
        scores = analyzer.polarity_scores(str(text))
        return [scores["pos"], scores["neg"], scores["neu"], scores["compound"]]

    return np.array([scores_for(t) for t in texts])


def evaluate(model, x_test, y_test):
    preds = model.predict(x_test)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    correct = (preds == y_test).astype(int)
    return acc, f1, preds, correct


def run_experiment(df, params):
    reviews = df["review"].astype(str)
    labels = df["sentiment"].astype(int)

    if params["sample_size"] < len(df):
        sampled = df.sample(n=params["sample_size"], random_state=params["random_state"])
        reviews = sampled["review"].astype(str)
        labels = sampled["sentiment"].astype(int)

    x_train, x_test, y_train, y_test = train_test_split(
        reviews,
        labels,
        test_size=params["test_size"],
        random_state=params["random_state"],
        stratify=labels,
    )

    vectorizer = CountVectorizer(
        max_features=params["max_features"],
        stop_words=params["stopwords"],
    )
    x_train_bow = vectorizer.fit_transform(x_train)
    x_test_bow = vectorizer.transform(x_test)

    need_lexicon = "Logistic Regression (Lexicon)" in params["selected_models"]
    x_train_lex, x_test_lex = None, None
    if need_lexicon:
        x_train_lex = build_vader_matrix(x_train)
        x_test_lex = build_vader_matrix(x_test)

    models = {}
    if "Multinomial NB (BoW)" in params["selected_models"]:
        m = MultinomialNB()
        m.fit(x_train_bow, y_train)
        models["Multinomial NB (BoW)"] = (m, "bow")

    if "Bernoulli NB (BoW)" in params["selected_models"]:
        m = BernoulliNB()
        m.fit(x_train_bow, y_train)
        models["Bernoulli NB (BoW)"] = (m, "bow")

    if "Logistic Regression (BoW)" in params["selected_models"]:
        m = LogisticRegression(max_iter=params["max_iter"])
        m.fit(x_train_bow, y_train)
        models["Logistic Regression (BoW)"] = (m, "bow")

    if need_lexicon:
        m = LogisticRegression(max_iter=params["max_iter"])
        m.fit(x_train_lex, y_train)
        models["Logistic Regression (Lexicon)"] = (m, "lex")

    results = []
    prediction_map = {}
    correctness_map = {}

    for model_name, (model, kind) in models.items():
        x_eval = x_test_bow if kind == "bow" else x_test_lex
        acc, f1, preds, correct = evaluate(model, x_eval, y_test)
        prediction_map[model_name] = preds
        correctness_map[model_name] = correct
        results.append(
            {
                "Model": model_name,
                "Accuracy": round(float(acc), 4),
                "F1": round(float(f1), 4),
            }
        )

    results_df = pd.DataFrame(results).sort_values("F1", ascending=False)

    return {
        "results_df": results_df,
        "models": models,
        "vectorizer": vectorizer,
        "y_test": y_test.to_numpy(),
        "predictions": prediction_map,
        "correctness": correctness_map,
        "x_test_size": len(y_test),
        "train_size": len(y_train),
        "test_size": len(y_test),
    }


def paired_significance(correct_a, correct_b):
    t_stat, t_p = ttest_rel(correct_a, correct_b)
    a_right_b_wrong = int(np.sum((correct_a == 1) & (correct_b == 0)))
    a_wrong_b_right = int(np.sum((correct_a == 0) & (correct_b == 1)))
    discordant = a_right_b_wrong + a_wrong_b_right

    mcnemar_p = np.nan
    if discordant > 0:
        mcnemar_p = binomtest(min(a_right_b_wrong, a_wrong_b_right), discordant, p=0.5).pvalue

    return {
        "t_stat": float(t_stat),
        "t_p": float(t_p),
        "a_right_b_wrong": a_right_b_wrong,
        "a_wrong_b_right": a_wrong_b_right,
        "mcnemar_p": float(mcnemar_p) if not np.isnan(mcnemar_p) else np.nan,
    }


def predict_sentence(text, model_name, state):
    model, kind = state["models"][model_name]

    if kind == "bow":
        x = state["vectorizer"].transform([text])
    else:
        x = build_vader_matrix([text])

    pred = int(model.predict(x)[0])
    label = "Positive" if pred == 1 else "Negative"

    prob = None
    if hasattr(model, "predict_proba"):
        prob = float(model.predict_proba(x)[0][pred])

    return label, prob


def main():
    st.set_page_config(page_title="NLP Sentiment Demo", layout="wide")
    st.title("Sentiment Analysis Model Demo")
    st.caption("Naive Bayes, Binary Naive Bayes, and Logistic Regression with BoW/Lexicon features")

    with st.sidebar:
        st.header("Experiment Settings")
        use_cleaned = st.toggle("Use cleaned dataset", value=True)
        selected_models = st.multiselect(
            "Models to train",
            MODEL_ORDER,
            default=MODEL_ORDER,
        )
        sample_size = st.slider("Rows to use", min_value=2000, max_value=50000, value=10000, step=2000)
        test_size = st.slider("Test split", min_value=0.1, max_value=0.4, value=0.2, step=0.05)
        max_features = st.slider("BoW max features", min_value=1000, max_value=30000, value=10000, step=1000)
        max_iter = st.slider("Logistic max iterations", min_value=200, max_value=3000, value=1000, step=100)
        random_state = st.number_input("Random seed", min_value=0, max_value=9999, value=42)
        stopwords_text = st.text_input("Stopwords (comma-separated)", value=", ".join(DEFAULT_STOPWORDS))

        run_btn = st.button("Run experiment", type="primary")

    if not selected_models:
        st.warning("Select at least one model in the sidebar.")
        return

    df = load_selected_dataset(use_cleaned=use_cleaned)
    class_counts = df["sentiment"].astype(str).value_counts().rename_axis("sentiment").reset_index(name="count")

    c1, c2 = st.columns([2, 1])
    with c1:
        st.write(f"Dataset rows: **{len(df):,}**")
        st.dataframe(df.head(5), use_container_width=True)
    with c2:
        st.write("Class balance")
        st.bar_chart(class_counts.set_index("sentiment"))

    if run_btn:
        params = {
            "selected_models": selected_models,
            "sample_size": sample_size,
            "test_size": test_size,
            "max_features": max_features,
            "max_iter": max_iter,
            "random_state": int(random_state),
            "stopwords": parse_stopwords(stopwords_text),
        }

        with st.spinner("Training models and evaluating..."):
            output = run_experiment(df, params)

        st.session_state["run_output"] = output
        st.session_state["run_params"] = params

    if "run_output" not in st.session_state:
        st.info("Set parameters and click 'Run experiment'.")
        return

    output = st.session_state["run_output"]
    results_df = output["results_df"]

    st.subheader("Model Performance")
    st.dataframe(results_df, use_container_width=True)

    best_row = results_df.iloc[0]
    m1, m2, m3 = st.columns(3)
    m1.metric("Best model", best_row["Model"])
    m2.metric("Best accuracy", f"{best_row['Accuracy']:.4f}")
    m3.metric("Best F1", f"{best_row['F1']:.4f}")

    st.bar_chart(results_df.set_index("Model")[["Accuracy", "F1"]])

    st.subheader("Statistical Significance")
    available = list(output["correctness"].keys())
    if len(available) < 2:
        st.info("Train at least two models to run significance testing.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            model_a = st.selectbox("Model A", available, index=0)
        with c2:
            model_b = st.selectbox("Model B", available, index=min(1, len(available) - 1))

        if model_a == model_b:
            st.warning("Pick two different models.")
        else:
            sig = paired_significance(output["correctness"][model_a], output["correctness"][model_b])
            st.write(f"Paired t-test p-value: **{sig['t_p']:.6f}**")
            if np.isnan(sig["mcnemar_p"]):
                st.write("McNemar exact p-value: not available (no disagreement cases).")
            else:
                st.write(f"McNemar exact p-value: **{sig['mcnemar_p']:.6f}**")
            st.caption(
                f"A correct/B wrong: {sig['a_right_b_wrong']} | A wrong/B correct: {sig['a_wrong_b_right']}"
            )

    st.subheader("Try Your Own Sentence")
    sentence = st.text_area(
        "Input sentence",
        value="This movie was surprisingly good and emotionally engaging.",
        height=100,
    )
    pred_model = st.selectbox("Model for prediction", list(output["models"].keys()), key="prediction_model")

    if st.button("Predict sentiment"):
        label, prob = predict_sentence(sentence, pred_model, output)
        st.success(f"Prediction: **{label}**")
        if prob is not None:
            st.write(f"Confidence (predicted class): **{prob:.4f}**")


if __name__ == "__main__":
    main()
