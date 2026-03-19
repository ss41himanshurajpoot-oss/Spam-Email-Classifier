# ==========================================
# Spam Email Classifier using Naive Bayes
# ==========================================

import pandas as pd
import numpy as np
import string
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

# ------------------------------------------
# Step 1: Load Dataset
# ------------------------------------------

def load_data(file_path):
    try:
        data = pd.read_csv(file_path, encoding='latin-1')
        X = data['message']
        y = data['label']
        data.columns = ['label', 'message']
        return data 
    except Exception as e:
        print("Error loading dataset:", e)
        return None

# ------------------------------------------
# Step 2: Data Cleaning
# ------------------------------------------

def clean_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    return text

def preprocess_data(data):
    data['cleaned'] = data['message'].apply(clean_text)
    return data

# ------------------------------------------
# Step 3: Train Model
# ------------------------------------------

def train_model(data):
    X = data['cleaned']
    y = data['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = Pipeline([
        ('vectorizer', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('classifier', MultinomialNB())
    ])

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("\nModel Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    return model

# ------------------------------------------
# Step 4: Save Model
# ------------------------------------------

def save_model(model, filename="model.pkl"):
    with open(filename, 'wb') as file:
        pickle.dump(model, file)
    print("Model saved as", filename)

# ------------------------------------------
# Step 5: Load Model
# ------------------------------------------

def load_model(filename="model.pkl"):
    try:
        with open(filename, 'rb') as file:
            model = pickle.load(file)
        return model
    except:
        print("Model not found.")
        return None

# ------------------------------------------
# Step 6: Predict Function
# ------------------------------------------

def predict_message(model, message):
    message = clean_text(message)
    prediction = model.predict([message])[0]
    probability = np.max(model.predict_proba([message]))

    return prediction, probability

# ------------------------------------------
# Step 7: Interactive Mode
# ------------------------------------------

def run_interface(model):
    print("\n===== Spam Classifier =====")
    print("Type 'exit' to quit\n")

    while True:
        user_input = input("Enter message: ")

        if user_input.lower() == 'exit':
            break

        prediction, prob = predict_message(model, user_input)

        print(f"Prediction: {prediction}")
        print(f"Confidence: {round(prob*100,2)}%")
        print("-" * 40)

# ------------------------------------------
# Step 8: Main Function
# ------------------------------------------

def main():
    print("Loading dataset...")

    data = load_data("spam.csv")

    if data is None:
        print("Dataset not found. Please add spam.csv file.")
        return

    data = preprocess_data(data)

    print("Training model...")
    model = train_model(data)

    save_model(model)

    run_interface(model)

# ------------------------------------------
# Step 9: Extra Testing
# ------------------------------------------

def test_examples(model):
    samples = [
        "Congratulations! You won a free lottery ticket",
        "Hey, are we meeting tomorrow?",
        "URGENT! Claim your prize now",
        "Let's study together for exams"
    ]

    for msg in samples:
        pred, prob = predict_message(model, msg)
        print(f"Message: {msg}")
        print(f"Prediction: {pred}, Confidence: {prob}")
        print()

# ------------------------------------------
# Entry Point
# ------------------------------------------

if __name__ == "__main__":
    main()

# ==========================================
# END OF PROGRAM
# ==========================================