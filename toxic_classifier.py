# =========================================
# TOXIC + HATE SPEECH MULTI-LABEL CLASSIFIER
# (NO NLTK VERSION)
# =========================================

import pandas as pd
import re

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# -------------------------
# 1. Load Dataset
# -------------------------
df = pd.read_csv("train.csv")

# Keep required columns
df = df[[
    "comment_text",
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate"
]]

# Rename columns to match project labels
df.rename(columns={
    "obscene": "swear",
    "identity_hate": "racism",
    "severe_toxic": "bullying"
}, inplace=True)

labels = ["toxic", "swear", "racism", "bullying", "threat", "insult"]

# -------------------------
# 2. Text Cleaning Function
# -------------------------
STOPWORDS = ENGLISH_STOP_WORDS

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)   # remove URLs
    text = re.sub(r"[^a-z\s]", "", text)         # remove punctuation/numbers
    text = " ".join(word for word in text.split() if word not in STOPWORDS)
    return text

df["comment_text"] = df["comment_text"].astype(str).apply(clean_text)

# -------------------------
# 3. Train-Test Split
# -------------------------
X = df["comment_text"]
y = df[labels]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------
# 4. TF-IDF Vectorization
# -------------------------
vectorizer = TfidfVectorizer(
    max_features=15000,
    ngram_range=(1, 2)
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# -------------------------
# 5. Model Training
# -------------------------
model = OneVsRestClassifier(
    LogisticRegression(max_iter=2000)
)

model.fit(X_train_vec, y_train)

print("\n✅ Model trained successfully\n")

# -------------------------
# 6. Evaluation
# -------------------------
y_pred = model.predict(X_test_vec)

print("📊 Classification Report:\n")
print(classification_report(y_test, y_pred, target_names=labels))

# -------------------------
# 7. Prediction Function
# -------------------------
def predict_toxicity(text):
    text = clean_text(text)
    vec = vectorizer.transform([text])
    prediction = model.predict(vec)[0]
    return dict(zip(labels, prediction))

# -------------------------
# 8. Sample Predictions
# -------------------------
samples = [
    "You are a stupid idiot",
    "I will kill you",
    "I love this movie",
    "Go back to your country"
]

for s in samples:
    print("\nComment:", s)
    print("Prediction:", predict_toxicity(s))
