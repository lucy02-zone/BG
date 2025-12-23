from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import joblib
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# --------------------------------
# Load Model & Vectorizer
# --------------------------------
model = joblib.load("model.joblib")
vectorizer = joblib.load("vectorizer.joblib")

labels = ["toxic", "swear", "racism", "bullying", "threat", "insult"]
STOPWORDS = ENGLISH_STOP_WORDS

# --------------------------------
# FastAPI App
# --------------------------------
app = FastAPI(
    title="Toxic Comment Moderation API",
    description="Multi-label toxicity and hate speech detection",
    version="1.0"
)

# --------------------------------
# Enable CORS (React → FastAPI)
# --------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # for development
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------
# Request Schema
# --------------------------------
class TextRequest(BaseModel):
    text: str

# --------------------------------
# Text Cleaning
# --------------------------------
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = " ".join(word for word in text.split() if word not in STOPWORDS)
    return text

# --------------------------------
# Routes
# --------------------------------
@app.get("/")
def root():
    return {"status": "Backend running successfully"}

@app.post("/predict")
def predict_toxicity(req: TextRequest):
    cleaned_text = clean_text(req.text)
    vec = vectorizer.transform([cleaned_text])

    prediction = model.predict(vec)[0]

    result = {
        label: int(value)
        for label, value in zip(labels, prediction)
    }

    return result
