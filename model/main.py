# Packages and Dependencies
import re
import nltk
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.utils import resample
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
import pickle

nltk.download("punkt", quiet = True)
nltk.download("punkt_tab", quiet = True)
nltk.download("stopwords", quiet = True)
nltk.download('wordnet', quiet = True)
nltk.download('omw-1.4', quiet = True)  # For lemmatizer
nltk.download("averaged_perceptron_tagger", quiet = True)  # Needed to get POS tags
nltk.download('averaged_perceptron_tagger_eng', quiet = True)


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

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha if alpha is not None else [1.0] * 3

    def forward(self, inputs, targets):
        CE_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-CE_loss)
        focal_loss = torch.tensor(self.alpha)[targets] * (1 - pt) ** self.gamma * CE_loss
        return focal_loss.mean()

# Functions

def tune_thresholds():
    model.eval()
    with torch.no_grad():
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
        logits = model(X_val_tensor)
        probs = F.softmax(logits, dim=1).numpy()

    best_f1 = 0
    best_thresholds = [0.5, 0.5, 0.5]

    thresholds_range = np.arange(0.3, 0.9, 0.05)

    for t0 in thresholds_range:
        for t1 in thresholds_range:
            for t2 in thresholds_range:
                predictions = []
                for prob in probs:
                    pred = np.argmax(prob)  # fallback
                    for i, thresh in enumerate([t0, t1, t2]):
                        if prob[i] > thresh:
                            pred = i
                            break
                    predictions.append(pred)

                score = f1_score(y_val, predictions, average='macro')
                if score > best_f1:
                    best_f1 = score
                    best_thresholds = [t0, t1, t2]

    print(f"Best thresholds: {best_thresholds} with macro F1: {best_f1:.4f}")
    return best_thresholds

def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN  # Default to noun
    

def remove_emojis(text):
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags
        u"\U00002700-\U000027BF"  # dingbats
        u"\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

def preprocess(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))
    
    text = remove_emojis(text)  # <- emoji removal
    tokens = word_tokenize(text.lower())
    pos_tags = nltk.pos_tag(tokens)

    lemmatized_tokens = [
        lemmatizer.lemmatize(word, get_wordnet_pos(pos))
        for word, pos in pos_tags
        if word.isalpha() and word not in stop_words and len(word) > 2
    ]

    return " ".join(lemmatized_tokens)

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
    sample_size = sum([len(df_hate), len(df_offensive), len(df_neither)]) - max_class_size
    # Upsample all to max_class_size
    df_hate_balanced = resample(df_hate, replace=True, n_samples=max(len(df_hate), 2*sample_size), random_state=57)
    df_offensive_balanced = resample(df_offensive, replace=True, n_samples=max(len(df_offensive), 2*sample_size), random_state=42)
    df_neither_balanced = resample(df_neither, replace=True, n_samples=max(len(df_neither), 2*sample_size), random_state=96)

    # Combine all into one balanced DataFrame
    df_balanced = pd.concat([df_hate_balanced, df_offensive_balanced, df_neither_balanced])

    # Shuffle rows
    df= df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

    # Step 4: Check if balancing worked
    print("\nBalanced class distribution:")
    print(df['class'].value_counts())

    
    processed_texts = texts.apply(preprocess) # You don't have to use map() function with a pandas data field, just use apply()

    # Text to feature conversion (CountVectorizer)

    vectorizer = TfidfVectorizer(max_features = 10000, sublinear_tf= True, stop_words = 'english', ngram_range=(1, 3), min_df = 3, max_df = 0.85)
    X = vectorizer.fit_transform(processed_texts).toarray()
    
    # Store the vectorizer
    with open("vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

    y = labels.values

    # Split training and testing data
    global X_train, X_test, y_train, y_test, X_val, y_val
    # Split training and testing data
    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=69)

    # Further split train into train + validation
    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.1, random_state=42)


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
    class_weights[0] *= 0.6 # Reduce aggressiveness
    class_weights_tensor = torch.tensor(class_weights, dtype = torch.float)

    criterion = FocalLoss(alpha=[0.8, 0.1, 0.1])
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001, weight_decay=1e-5)

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
    thresholds = tune_thresholds()
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        logits = model(X_test_tensor)
        probs = F.softmax(logits, dim=1).numpy()

        predictions = []
        for prob in probs:
            pred = np.argmax(prob)
            for i, thresh in enumerate(thresholds):
                if prob[i] > thresh:
                    pred = i
                    break
            predictions.append(pred)

    print("Classification Report: ")
    print(classification_report(y_test, predictions, target_names=["Hate Speech", "Offensive", "Neither"]))



if __name__ == "__main__":
    # exit() # Remove before re-training the model, otherwise the program will exit!
    modelDefinition()
    trainingLoop()
    evaluation()


    