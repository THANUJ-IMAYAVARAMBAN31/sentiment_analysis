import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("sentiment_analysis/data/IMDB Dataset.csv")

df["sentiment"] = df["sentiment"].map({
    "negative": 0,
    "positive": 1
})

X_train, X_test, y_train, y_test = train_test_split(
    df["review"],
    df["sentiment"],
    test_size=0.2,
    random_state=42,
    stratify=df["sentiment"]
)

vectorizer = TfidfVectorizer(
    max_features=1000,
    stop_words="english"
)

X_train_vec = vectorizer.fit_transform(X_train)

model = LogisticRegression()

model.fit(X_train_vec, y_train)

joblib.dump(model, "sentiment_analysis/models/logistic_model.pkl")
joblib.dump(vectorizer, "sentiment_analysis/models/tfidf_vectorizer.pkl")

print("Saved")