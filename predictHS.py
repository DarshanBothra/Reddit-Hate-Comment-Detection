# Packages and Dependencies

import torch
import torch.nn as nn
import pickle
import os
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import numpy as np

nltk.download("punkt", quiet = True)
nltk.download("punkt_tab", quiet = True)
nltk.download("stopwords", quiet = True)

# Classes 

class MLPModel(nn.Module):
    def __init__(self, input_size, hidden_size=256, num_classes=3):
        super(MLPModel, self).__init__()
        self.model = nn.Sequential(
        nn.Linear(input_size, hidden_size),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(hidden_size, hidden_size // 2),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(hidden_size // 2, num_classes)
    )

    def forward(self, x):
        return self.model(x)


# Functions

def preprocess(text):

    # Get stop words (common english words to be removed)
    stop_words = set(stopwords.words("english"))

    # Text preprocessing
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    return " ".join(tokens)

def predict(text):
    loadResources()
    processed = preprocess(text)
    features = _vectorizer.transform([processed]).toarray()
    features_tensor = torch.tensor(features, dtype = torch.float32)
    
    with torch.no_grad():
        outputs = _model(features_tensor)

        _, predicted = torch.max(outputs, 1) # Prediction

        return _class_labels[predicted.item()]

# Caching Setup

_vectorizer = None
_model = None
_class_labels = {
    0: "Hate Speech",
    1: "Offensive",
    2: "Neither",
}

# Load once

def loadResources():
    global _vectorizer, _model
    if _vectorizer is None or _model is None:
        # Load the vectorizer
        with open("vectorizer.pkl", "rb") as f:
            _vectorizer = pickle.load(f)
        input_size = _vectorizer.max_features
        _model = MLPModel(input_size=input_size)
        _model.load_state_dict(torch.load("mlp_tfidf_adam.pt"))
        _model.eval()

