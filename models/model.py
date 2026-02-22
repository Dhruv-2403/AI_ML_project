from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
from config.config import Config

def get_vectorizer():
    return TfidfVectorizer(stop_words='english', max_features=Config.MAX_FEATURES, ngram_range=Config.NGRAM_RANGE, min_df=Config.MIN_DF, max_df=Config.MAX_DF)

def get_model():
    return LogisticRegression(max_iter=Config.MAX_ITER, C=Config.C_PARAM, solver=Config.SOLVER, random_state=Config.RANDOM_STATE)

def save_model(model, vectorizer):
    joblib.dump(model, Config.MODEL_PATH)
    joblib.dump(vectorizer, Config.VECTORIZER_PATH)

def load_model():
    model = joblib.load(Config.MODEL_PATH)
    vectorizer = joblib.load(Config.VECTORIZER_PATH)
    return model, vectorizer
