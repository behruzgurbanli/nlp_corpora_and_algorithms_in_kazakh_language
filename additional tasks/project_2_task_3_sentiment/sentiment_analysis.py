# sentiment_analysis.py
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from scipy.stats import ttest_rel
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from load_dataset import load_imdb_dataset
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# STEP 1: Load Cleaned Dataset
# -----------------------------
df = pd.read_csv("IMDB_Cleaned_Dataset.csv")
print(f"Loaded cleaned dataset with shape: {df.shape}")

# -----------------------------
# STEP 2: Split Train/Test
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    df["review"], df["sentiment"], test_size=0.2, random_state=42
)

# -----------------------------
# STEP 3: Bag-of-Words Features
# -----------------------------
custom_stopwords = ['the','and','of','to','is','in','it','this','that','as','with','for','on','but','are','you','was']
vectorizer = CountVectorizer(max_features=10000, stop_words=custom_stopwords)
X_train_bow = vectorizer.fit_transform(X_train)
X_test_bow = vectorizer.transform(X_test)

# -----------------------------
# STEP 4: VADER Lexicon Features (Logistic only)
# -----------------------------
analyzer = SentimentIntensityAnalyzer()
def vader_features(text):
    scores = analyzer.polarity_scores(text)
    return [scores['pos'], scores['neg'], scores['neu'], scores['compound']]

X_train_lex = np.array([vader_features(t) for t in X_train])
X_test_lex = np.array([vader_features(t) for t in X_test])

# -----------------------------
# STEP 5: Train Models
# -----------------------------
# Bag-of-Words
nb = MultinomialNB()
nb.fit(X_train_bow, y_train)

bnb = BernoulliNB()
bnb.fit(X_train_bow, y_train)

lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_bow, y_train)

# Lexicon only Logistic Regression
lr_lex = LogisticRegression(max_iter=1000)
lr_lex.fit(X_train_lex, y_train)

# -----------------------------
# STEP 6: Evaluate Models
# -----------------------------
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    return acc, f1, y_pred

results = []

# Bag-of-Words models
for model, name in [(nb,"Multinomial NB"), (bnb,"Bernoulli NB"), (lr,"Logistic Regression")]:
    acc, f1, y_pred = evaluate_model(model, X_test_bow, y_test)
    results.append({"Model": name, "Feature": "Bag-of-Words", "Accuracy": acc, "F1": f1, "Predictions": y_pred})

# Lexicon model
acc, f1, y_pred = evaluate_model(lr_lex, X_test_lex, y_test)
results.append({"Model": "Logistic Regression", "Feature": "Lexicon", "Accuracy": acc, "F1": f1, "Predictions": y_pred})

# -----------------------------
# STEP 7: Statistical Significance (Paired t-test NB vs Logistic BoW)
# -----------------------------
pred_nb = results[0]["Predictions"]  # Multinomial NB BoW
pred_lr = results[2]["Predictions"]  # Logistic Regression BoW
t_stat, p_value = ttest_rel(pred_nb, pred_lr)
print(f"\nPaired t-test (Multinomial NB vs Logistic) on BoW p-value: {p_value:.5f}")

# -----------------------------
# STEP 8: Display Results
# -----------------------------
print("\nFinal Model Comparison:")
print(f"{'Model':<20} {'Feature':<15} {'Accuracy':<10} {'F1':<10}")
for r in results:
    print(f"{r['Model']:<20} {r['Feature']:<15} {r['Accuracy']:<10.4f} {r['F1']:<10.4f}")

# -----------------------------
# STEP 9: Save Results to CSV
# -----------------------------
results_df = pd.DataFrame([{"Model": r["Model"], "Feature": r["Feature"], "Accuracy": r["Accuracy"], "F1": r["F1"]} for r in results])
results_df.to_csv("model_comparison_results.csv", index=False)
print("\n✅ Results saved to model_comparison_results.csv")

# -----------------------------
# STEP 10: Plot Bar Chart
# -----------------------------
plt.figure(figsize=(8,5))
x_labels = [f"{r['Model']}\n({r['Feature']})" for r in results]
accuracies = [r['Accuracy'] for r in results]
f1_scores = [r['F1'] for r in results]

bar_width = 0.35
x = np.arange(len(x_labels))

plt.bar(x - bar_width/2, accuracies, bar_width, label='Accuracy', color='skyblue')
plt.bar(x + bar_width/2, f1_scores, bar_width, label='F1 Score', color='lightcoral')

plt.xticks(x, x_labels, rotation=20, ha='right')
plt.ylim(0,1)
plt.ylabel('Score')
plt.title('Model Performance Comparison')
plt.legend()
plt.tight_layout()
plt.savefig("model_comparison_chart.png")  # saves the chart
plt.show()
print("\n✅ Bar chart saved as model_comparison_chart.png")