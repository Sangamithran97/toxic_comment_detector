import joblib
import re
import nltk
from nltk.corpus import stopwords

# Load model and vectorizer
print("Loading model and vectorizer...")
model = joblib.load("models/toxic_model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

# Setup text cleaner
nltk.download("stopwords")
stop = set(stopwords.words("english"))

def clean_text(text):
    text = re.sub(r"[^a-zA-Z ]", "", str(text))
    text = text.lower()
    text = " ".join([w for w in text.split() if w not in stop])
    return text

# Prediction Function
def predict_toxicity(text):
    clean = clean_text(text)
    vec = vectorizer.transform([clean])
    pred = model.predict(vec)
    labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    result = {labels[i]: int(pred[0][i]) for i in range(len(labels))}
    return result

# User Input Section
print("\nTOXIC COMMENT DETECTOR")
print("Type a comment and press Enter (type 'exit' to quit)\n")

while True:
    user_input = input("Enter comment: ")
    if user_input.lower() in ["exit", "quit"]:
        print("Exiting... Goodbye!")
        break
    prediction = predict_toxicity(user_input)
    print("Prediction:", prediction, "\n")
