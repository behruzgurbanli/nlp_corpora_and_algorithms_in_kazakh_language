# Project Report: Sentiment Analysis Model Comparison and Statistical Evaluation

## Purpose of the Project

The goal of this project is to develop and benchmark a sentiment classification pipeline using the **IMDB Movie Reviews dataset**. By comparing traditional probabilistic models (**Naive Bayes**) against discriminative classifiers (**Logistic Regression**), we aim to identify the most effective feature extraction technique—**Bag-of-Words (BoW)** versus **VADER Lexicon**—for binary sentiment prediction.

---

## Technical Implementation

### 1. Data Pipeline
- **Dataset:** `IMDB_Cleaned_Dataset.csv`
- **Pre-processing:** The data was partitioned into **80% training** and **20% testing** sets to ensure a robust evaluation of model generalization.
- **Vectorization:** `CountVectorizer` was utilized to transform raw text into a Bag-of-Words matrix, limited to the **10,000 most frequent features**.

### 2. Feature Engineering Strategies
The project compared two distinct methodologies:
- **Statistical (BoW):** Captures word frequency and distribution specific to the movie review corpus.
- **Lexicon (VADER):** Uses the `SentimentIntensityAnalyzer` to generate a 4-dimensional feature vector consisting of **Positive, Negative, Neutral, and Compound** scores.



### 3. Model Configurations
Four model-feature combinations were trained and evaluated:
1.  **Multinomial Naive Bayes (BoW):** A probabilistic approach optimized for discrete word counts.
2.  **Bernoulli Naive Bayes (BoW):** A variant focusing on the binary presence or absence of words.
3.  **Logistic Regression (BoW):** A discriminative linear model designed for high-dimensional feature spaces.
4.  **Logistic Regression (Lexicon):** A low-dimensional classifier utilizing VADER sentiment polarities.

---

## Performance Evaluation

Models were benchmarked using **Accuracy** and **F1-Score** to measure the balance between precision and recall.

### Results Summary

| Model | Feature Extraction | Accuracy | F1-Score |
| :--- | :--- | :--- | :--- |
| **Logistic Regression** | **Bag-of-Words** | **0.8742** | **0.8755** |
| Bernoulli NB | Bag-of-Words | 0.8515 | 0.8513 |
| Multinomial NB | Bag-of-Words | 0.8429 | 0.8409 |
| Logistic Regression | Lexicon | 0.7289 | 0.7346 |


### Statistical Significance (Paired T-Test)
To validate the performance gap, a **Paired Sample T-test** was conducted between the predictions of **Multinomial NB** and **Logistic Regression** (BoW).

- **p-value:** `0.00000`
- **Interpretation:** The resulting p-value is significantly lower than the standard threshold ($\alpha = 0.05$). This confirms that the superior performance of **Logistic Regression** is statistically significant and not a result of random variance in the test set.



---

## Key Findings

- **Feature Performance:** The **Bag-of-Words** approach significantly outperformed the **VADER Lexicon** method. This suggests that models trained on domain-specific vocabulary (movie-related terms) are more effective than general-purpose sentiment dictionaries.
- **Algorithm Efficacy:** **Logistic Regression** yielded the highest accuracy. Unlike Naive Bayes, which assumes feature independence, Logistic Regression can better model the relationship between overlapping words in text data.
- **Robustness:** The top model achieved an F1-score of **0.8753**, indicating high reliability in correctly identifying both positive and negative reviews.

---

## Conclusion

The combination of **Logistic Regression and Bag-of-Words** is the optimal choice for this sentiment analysis task. While lexicon-based methods (VADER) are computationally faster and require no training, they resulted in a **14.5% drop in accuracy** compared to the trained statistical models.

---

**Next Step:** Would you like me to generate a Python script to create a **Confusion Matrix** or a **Classification Report** for the top-performing Logistic Regression model?