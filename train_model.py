import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load CSV
df = pd.read_csv("customer_queries.csv")

# Features and labels
X = df['query']
y = df['category']

# Vectorize
vectorizer = TfidfVectorizer(stop_words='english')
X_vect = vectorizer.fit_transform(X)

# Train model
clf = RandomForestClassifier()
clf.fit(X_vect, y)

# Save model and vectorizer
joblib.dump(clf, 'chatbot_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

print("✅ Model and vectorizer saved.")
