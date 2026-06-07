from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

def get_bow(train_texts, test_texts, max_features=5000):
    vectorizer = CountVectorizer(max_features=max_features)
    X_train = vectorizer.fit_transform(train_texts)
    X_test  = vectorizer.transform(test_texts)
    return X_train, X_test, vectorizer

def get_tfidf(train_texts, test_texts, max_features=5000):
    vectorizer = TfidfVectorizer(max_features=max_features)
    X_train = vectorizer.fit_transform(train_texts)
    X_test  = vectorizer.transform(test_texts)
    return X_train, X_test, vectorizer