class Config:
    DATASET_PATH = "data/WELFake_Dataset.csv"
    MODEL_PATH = "models/lr_model.joblib"
    VECTORIZER_PATH = "models/tfidf_vectorizer.joblib"
    MAX_FEATURES = 10000
    NGRAM_RANGE = (1, 2)
    MIN_DF = 3
    MAX_DF = 0.95
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    MAX_ITER = 1500
    C_PARAM = 1.5
    SOLVER = "saga"
