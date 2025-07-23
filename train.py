from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Load SST2 dataset (from Hugging Face)
dataset = load_dataset("sst2")

# Combine train + validation
texts = list(dataset['train']['sentence']) + list(dataset['validation']['sentence'])
labels = list(dataset['train']['label']) + list(dataset['validation']['label'])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Build lightweight pipeline (TF-IDF + LogisticRegression)
model = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LogisticRegression(max_iter=1000))
])

# Train
model.fit(X_train, y_train)

# Test accuracy
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")

# Save model
joblib.dump(model, "sentiment_model.joblib")