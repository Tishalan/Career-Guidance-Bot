import json
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INTENTS_PATH = os.path.join(BASE_DIR, "intents.json")
MODEL_PATH = os.path.join(BASE_DIR, "bot_model.pkl")

def train():
    with open(INTENTS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    X = []
    y = []
    
    # Process intents
    for intent in data["intents"]:
        tag = intent["tag"]
        for pattern in intent["patterns"]:
            X.append(pattern)
            y.append(tag)
            
    # Create an ML Pipeline: TF-IDF feature extraction -> Logistic Regression Classifier
    # Using sublinear_tf, both word and character n-grams, no stop word removal
    # (patterns are too short; removing stop words kills signal for inputs like "yes", "PhD")
    print(f"Training model with {len(X)} samples across {len(set(y))} intents...")
    model = make_pipeline(
        TfidfVectorizer(
            lowercase=True,
            analyzer='word',
            ngram_range=(1, 2),
            sublinear_tf=True
        ),
        LogisticRegression(max_iter=2000, C=5.0, class_weight='balanced')
    )
    model.fit(X, y)
    print("Training complete!")
    
    # Save the model and the intents data to a file
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"model": model, "data": data}, f)
    print(f"Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    train()
