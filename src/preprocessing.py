import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess(text, verbose=False):
    steps = {}

    # Step 1: Lowercase
    text = text.lower()
    if verbose: steps['1_lowercase'] = text

    # Step 2: Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    if verbose: steps['2_no_punctuation'] = text

    # Step 3: Remove special characters / numbers
    text = re.sub(r'\d+', '', text)
    if verbose: steps['3_no_numbers'] = text

    # Step 4: Tokenize
    tokens = word_tokenize(text)
    if verbose: steps['4_tokens'] = tokens

    # Step 5: Remove stopwords
    tokens = [t for t in tokens if t not in stop_words]
    if verbose: steps['5_no_stopwords'] = tokens

    # Step 6: Lemmatize
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    if verbose: steps['6_lemmatized'] = tokens

    final = ' '.join(tokens)
    if verbose:
        steps['7_final'] = final
        return final, steps
    return final