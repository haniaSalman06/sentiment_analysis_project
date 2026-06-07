import joblib, os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from nltk.corpus import movie_reviews
import nltk, os

from src.preprocessing import preprocess
from src.feature_extraction import get_bow
from src.utils import RESULTS_DIR, ensure_dirs

def load_data():
    docs, labels = [], []
    for cat in movie_reviews.categories():
        for fileid in movie_reviews.fileids(cat):
            text = movie_reviews.raw(fileid)
            docs.append(preprocess(text))
            labels.append(1 if cat == 'pos' else 0)
    return docs, labels

def train_bow_model():
    ensure_dirs()
    print("Loading and preprocessing data...")
    docs, labels = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        docs, labels, test_size=0.2, random_state=42)

    print("Extracting BoW features...")
    X_train_bow, X_test_bow, bow_vec = get_bow(X_train, X_test)

    print("Training Logistic Regression (BoW)...")
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_bow, y_train)

    # Save model and vectorizer
    joblib.dump(model,   os.path.join(RESULTS_DIR, 'bow_model.pkl'))
    joblib.dump(bow_vec, os.path.join(RESULTS_DIR, 'bow_vectorizer.pkl'))
    print("BoW model saved to results/")
    return model, bow_vec, X_test_bow, y_test

if __name__ == '__main__':
    train_bow_model()