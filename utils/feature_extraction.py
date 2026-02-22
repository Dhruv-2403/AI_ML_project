from sklearn.feature_extraction.text import TfidfVectorizer

def extract_features(texts):
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=10000,
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.95,
        sublinear_tf=True,
    )
    X = vectorizer.fit_transform(texts)
    return X, vectorizer

def transform_text(texts, vectorizer):
    return vectorizer.transform(texts)
