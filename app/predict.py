import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

model = joblib.load(
    BASE_DIR / "models" / "logistic_model.pkl"
)

vectorizer = joblib.load(
    BASE_DIR / "models" / "tfidf_vectorizer.pkl"
)

def predict_sentiment(text):

    text_vec = vectorizer.transform([text])

    prediction = model.predict(text_vec)[0]

    probability = model.predict_proba(text_vec)[0]

    confidence = max(probability)

    return prediction, confidence