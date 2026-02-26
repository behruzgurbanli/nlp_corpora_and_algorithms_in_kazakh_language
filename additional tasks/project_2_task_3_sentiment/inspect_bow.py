# inspect_bow.py
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

# -----------------------------
# STEP 1: Load Cleaned Dataset
# -----------------------------
df = pd.read_csv("IMDB_Cleaned_Dataset.csv")
print(f"Loaded cleaned dataset with shape: {df.shape}")

# -----------------------------
# STEP 2: Fit Bag-of-Words
# -----------------------------
custom_stopwords = ['the','and','of','to','is','in','it','this','that','as','with','for','on','but','are','you','was','not']
vectorizer = CountVectorizer(max_features=10000, stop_words=custom_stopwords)
X_bow = vectorizer.fit_transform(df["review"])
print(f"Bag-of-Words matrix shape: {X_bow.shape}")

# -----------------------------
# STEP 3: Show Most Frequent 20 Words
# -----------------------------
# Sum counts across all documents
word_counts = np.asarray(X_bow.sum(axis=0)).flatten()
# Map to feature names
vocab = vectorizer.get_feature_names_out()
# Sort by counts descending
top_indices = word_counts.argsort()[::-1][:20]
print("\nTop 20 most frequent words:")
for i in top_indices:
    print(f"{vocab[i]}: {word_counts[i]}")

# -----------------------------
# STEP 4: Print First 5 Rows of Matrix
# -----------------------------
print("\nFirst 5 rows of BoW matrix (counts):")
print(X_bow[:5].toarray())