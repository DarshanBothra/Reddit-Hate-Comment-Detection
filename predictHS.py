# Packages and Dependencies

import torch
import torch.nn as nn
import pickle
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import numpy as np

nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")

# Classes 

class MLPModel(nn.Module):
    def __init__(self, input_size, hidden_size=256, num_classes=3):
        super(MLPModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),  # model.0
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, num_classes)  # model.3
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
    processed = preprocess(text)
    features = vectorizer.transform([processed]).toarray()
    features_tensor = torch.tensor(features, dtype = torch.float32)
    
    with torch.no_grad():
        outputs = model(features_tensor)

        _, predicted = torch.max(outputs, 1) # Prediction

        return class_labels[predicted.item()]

# Main
def main():
    # Load the vectorizer
    with open("vectorizer.pkl", "rb") as f:
        global vectorizer
        vectorizer = pickle.load(f)

    input_size = vectorizer.max_features
    global model
    model = MLPModel(input_size=input_size)
    model.load_state_dict(torch.load("mlp_tfidf_adam.pt"))
    model.eval()

    global class_labels
    class_labels = {
        0: "Hate Speech",
        1: "Offensive",
        2: "Neither",
    }

    sentence = input("Enter a sentence: ")
    label = predict(sentence)
    print(f"Predicted Class: {label}")

    sentence = input("Enter another sentence: ")
    label = predict(sentence)
    print(f"Predicted Class: {label}")

if __name__ == "__main__":
    main()
