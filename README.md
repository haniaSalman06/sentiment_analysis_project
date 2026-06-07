# Sentiment Analysis: BoW vs TF-IDF

A complete NLP sentiment analysis system comparing Bag of Words and TF-IDF
feature extraction techniques using Logistic Regression on the NLTK Movie Reviews dataset.

---

## Dataset
- **Source:** NLTK Movie Reviews Corpus
- **Size:** 2000 reviews (1000 positive, 1000 negative)
- **Labels:** Binary (positive / negative)

---

## Project Structure
---

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/sentiment_analysis_project.git
cd sentiment_analysis_project
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Download NLTK data:
```bash
python -c "import nltk; nltk.download('movie_reviews'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('punkt')"
```

---

## Methodology

1. **Preprocessing:** Lowercasing, punctuation removal, tokenization, stopword removal, lemmatization
2. **Feature Extraction:** Bag of Words and TF-IDF (max 5000 features)
3. **Model:** Logistic Regression (scikit-learn)
4. **Evaluation:** Accuracy, Precision, Recall, F1-Score, Confusion Matrix

---

## Results

| Metric    | BoW    | TF-IDF |
|-----------|--------|--------|
| Accuracy  | 0.8250 | 0.8300 |
| Precision | 0.8227 | 0.8308 |
| Recall    | 0.8308 | 0.8308 |
| F1 Score  | 0.8267 | 0.8308 |

> TF-IDF outperformed Bag of Words on Accuracy, Precision and F1-Score.

> Fill in your actual results from the notebook output.

---

## How to Run

Open the notebook:
```bash
jupyter notebook notebooks/sentiment_analysis.ipynb
```

Or run training scripts directly:
```bash
python src/train_bow.py
python src/train_tfidf.py
```

---

## Author
- **Name:** Your Name
- **Course:** Programming for AI
- **Assignment:** #4 — Comparative Sentiment Analysis