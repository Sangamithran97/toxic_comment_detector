import pandas as pd
import re
import nltk
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os
import numpy as np

print("Loading dataset...")
df = pd.read_csv("data/train.csv")
print("Dataset shape:", df.shape)

# Step 1: Clean Text
nltk.download("stopwords")
stop = set(stopwords.words("english"))

def clean_text(text):
    text = re.sub(r"[^a-zA-Z ]", "", str(text))
    text = text.lower()
    text = " ".join([w for w in text.split() if w not in stop])
    return text

df["clean_comment"] = df["comment_text"].apply(clean_text)

# Step 2: Visualization
y = df[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]]
plt.figure(figsize=(8,4))
y.sum().plot(kind="bar", title="Label Distribution")
plt.xlabel("Labels")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("label_distribution.png")
plt.close()
print("Saved label distribution as 'label_distribution.png'")

# Step 3: TF-IDF Features
print("Extracting TF-IDF features...")
vectorizer = TfidfVectorizer(stop_words="english")  
X = vectorizer.fit_transform(df["clean_comment"])
print(f"TF-IDF shape: {X.shape}")  # shows how many features were created

# Step 4: Train-Test Split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Compare Models
models = {
    "Logistic Regression": OneVsRestClassifier(LogisticRegression(max_iter=200, class_weight='balanced')),
    "SVM": OneVsRestClassifier(LinearSVC(class_weight='balanced')),
    "Naive Bayes": OneVsRestClassifier(MultinomialNB())
}

results = {}
results1={}

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    x_pred = model.predict(X_train)
    acc1 = accuracy_score(y_train, x_pred)  
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    results1[name] = acc1

    print(f"{name} -> Train Accuracy: {acc1:.4f}, Test Accuracy: {acc:.4f}")
    print(f"\nClassification Report for {name}:")
    print(classification_report(y_test, y_pred, target_names=y.columns))

print("\nModel Comparison:")
for name in results.keys():
    print(f"{name:20} | Train: {results1[name]:.4f} | Test: {results[name]:.4f}")

best_model_name = max(results, key=results.get)
print(f"\n Best model: {best_model_name}")

best_model = models[best_model_name]

# Step 7: Save Best Model
os.makedirs("models", exist_ok=True)
joblib.dump(best_model, "models/toxic_model.pkl")
joblib.dump(vectorizer, "models/vectorizer.pkl")
print("\nSaved best model and vectorizer in 'models/' folder.")
