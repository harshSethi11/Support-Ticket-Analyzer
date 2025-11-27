from fastapi import FastAPI
from pydantic import BaseModel
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline
)
import torch
import re

app = FastAPI(title="Support Ticket Analyzer API")

class Ticket(BaseModel):
    text: str

# -------------------------------------------------
# Load sentiment model
# -------------------------------------------------
classifier_name = "distilbert-base-uncased-finetuned-sst-2-english"
classifier_tokenizer = AutoTokenizer.from_pretrained(classifier_name)
classifier_model = AutoModelForSequenceClassification.from_pretrained(classifier_name)
classifier_model.eval()

# -------------------------------------------------
# Load TWO summarizers
# -------------------------------------------------
summ_short = pipeline(
    "summarization",
    model="facebook/bart-large-cnn",
    device=0 if torch.cuda.is_available() else -1
)

summ_long = pipeline(
    "summarization",
    model="philschmid/bart-large-cnn-samsum",
    device=0 if torch.cuda.is_available() else -1
)

# -------------------------------------------------
# Helpers
# -------------------------------------------------
def extract_customer_text(text):
    """Extract only customer parts from dialog-style input."""
    lines = text.strip().split("\n")
    collected = []

    for line in lines:
        if ":" in line:
            speaker, msg = line.split(":", 1)
            if speaker.strip().lower() in ["customer", "user", "client"]:
                collected.append(msg.strip())
        else:
            collected.append(line.strip())

    return " ".join(collected).strip()


def summarize_text(text: str) -> str:
    cleaned = extract_customer_text(text)
    wc = len(cleaned.split())

    # SHORT INPUT → BART CNN
    if wc < 18:
        out = summ_short(
            cleaned,
            max_length=30,
            min_length=5,
            do_sample=False,
            truncation=True
        )
        summary = out[0]["summary_text"]
        return summary

    # LONG DIALOG → SAMSUM model
    out = summ_long(
        cleaned,
        max_length=60,
        min_length=10,
        do_sample=False,
        truncation=True
    )
    return out[0]["summary_text"]


# -------------------------------------------------
# API
# -------------------------------------------------
@app.post("/analyze")
def analyze(ticket: Ticket):
    cleaned = extract_customer_text(ticket.text)

    # Sentiment prediction
    with torch.no_grad():
        tokens = classifier_tokenizer(
            cleaned,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256
        )
        logits = classifier_model(**tokens).logits
        pred = int(torch.argmax(logits))

    sentiment = "negative" if pred == 0 else "positive"

    summary = summarize_text(ticket.text)

    return {
        "sentiment": sentiment,
        "summary": summary,
        "extracted_text": cleaned
    }


@app.get("/")
def root():
    return {"message": "Support Ticket Analyzer API"}


@app.get("/health")
def health():
    return {"status": "healthy"}

