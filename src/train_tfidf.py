import joblib, os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from nltk.corpus import movie_reviews

from src.preprocessing import preprocess
from src.feature_extraction import get_tfidf
from src.utils import RESULTS_DIR, ensure_dirs

def load_data():
    docs, labels = [], []
    for cat in movie_reviews.categories():
        for fileid in movie_reviews.fileids(cat):
            text = movie_reviews.raw(fileid)
            docs.append(preprocess(text))
            labels.append(1 if cat == 'pos' else 0)
    return docs, labels

def train_tfidf_model():
    ensure_dirs()
    print("Loading and preprocessing data...")
    docs, labels = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        docs, labels, test_size=0.2, random_state=42)

    print("Extracting TF-IDF features...")
    X_train_tfidf, X_test_tfidf, tfidf_vec = get_tfidf(X_train, X_test)

    print("Training Logistic Regression (TF-IDF)...")
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_tfidf, y_train)

    joblib.dump(model,      os.path.join(RESULTS_DIR, 'tfidf_model.pkl'))
    joblib.dump(tfidf_vec,  os.path.join(RESULTS_DIR, 'tfidf_vectorizer.pkl'))
    print("TF-IDF model saved to results/")
    return model, tfidf_vec, X_test_tfidf, y_test

if __name__ == '__main__':
    train_tfidf_model()