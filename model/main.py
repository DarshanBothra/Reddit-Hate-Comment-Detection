# Packages and Dependencies

import nltk
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.utils import resample
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import TensorDataset, DataLoader
import pickle

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

def modelDefinition():


    # Load the dataset
    df = pd.read_csv("model/labeled_data.csv")
    texts = df['tweet']
    labels = df['class']

    # Check the distribution of classes before upsampling
    print("Original class distribution:")
    print(df['class'].value_counts())

    # Step 1: Separate the majority and minority classes
    # Separate each class
    df_hate = df[df['class'] == 0]
    df_offensive = df[df['class'] == 1]
    df_neither = df[df['class'] == 2]

    # Find the largest class size
    max_class_size = max(len(df_hate), len(df_offensive), len(df_neither))

    # Upsample all to max_class_size
    df_hate_balanced = resample(df_hate, replace=True, n_samples=max_class_size, random_state=42)
    df_offensive_balanced = resample(df_offensive, replace=True, n_samples=max_class_size, random_state=42)
    df_neither_balanced = resample(df_neither, replace=True, n_samples=max_class_size, random_state=42)

    # Combine all into one balanced DataFrame
    df_balanced = pd.concat([df_hate_balanced, df_offensive_balanced, df_neither_balanced])

    # Shuffle rows
    df= df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

    # Step 4: Check if balancing worked
    print("\nBalanced class distribution:")
    print(df['class'].value_counts())

    
    processed_texts = texts.apply(preprocess) # You don't have to use map() function with a pandas data field, just use apply()

    # Text to feature conversion (CountVectorizer)

    vectorizer = TfidfVectorizer(max_features = 5000, sublinear_tf= True, stop_words = 'english', ngram_range=(1, 3))
    X = vectorizer.fit_transform(processed_texts).toarray()
    
    # Store the vectorizer
    with open("vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

    y = labels.values

    # Split training and testing data
    global X_train, X_test, y_train, y_test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=69) 

    input_size = X_train.shape[1]
    num_classes = len(set(y))
    global model
    model = MLPModel(input_size = X_train.shape[1], hidden_size = 256, num_classes = 3)

    # Create Dataloader

    X_train_tensor = torch.tensor(X_train, dtype = torch.float32)
    y_train_tensor = torch.tensor(y_train)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    global train_loader
    train_loader = DataLoader(train_dataset, batch_size = 64, shuffle = True)

def trainingLoop():
    

    class_weights = compute_class_weight('balanced', classes = np.unique(y_train), y = y_train)
    class_weights_tensor = torch.tensor(class_weights, dtype = torch.float)

    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001)

    epochs = 12

    for epoch in range(epochs):
        total_loss = 0
        model.train()
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {total_loss/len(train_loader):.4f}")

    # Save the model configuration

    torch.save(model.state_dict(), "mlp_tfidf_adam.pt")
    print("Model Saved to mlp_tfidf_adam.pt")

def evaluation():
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test, dtype = torch.float32)
        outputs = model(X_test_tensor)
        _, predicted = torch.max(outputs, 1)
        predicted = predicted.numpy()
    print("Classification Report: ")
    print(classification_report(y_test, predicted, target_names=["Hate Speech", "Offensive", "Neither"]))


if __name__ == "__main__":
    exit() # Remove before re-training the model, otherwise the program will exit!
    modelDefinition()
    trainingLoop()
    evaluation()


    