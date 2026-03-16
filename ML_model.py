import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
print("Loading dataset...")
data = pd.read_csv("Reviews.csv", nrows=20000)
X = data["Text"]
y = data["Score"]
print("Vectorizing text...")
vectorizer = TfidfVectorizer(
    ngram_range=(1,2),
    max_features=5000
)
X_vector = vectorizer.fit_transform(X)

print("Splitting data...")

X_train, X_test, y_train, y_test = train_test_split(
    X_vector, y, test_size=0.2, random_state=42
)
print("Training model...")
model = LogisticRegression(max_iter=1000, class_weight="balanced")
model.fit(X_train, y_train)
print("Saving model...")
pickle.dump(model, open("model.pkl","wb"))
pickle.dump(vectorizer, open("vectorizer.pkl","wb"))
print("Model trained successfully")
