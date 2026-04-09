import pandas as pd
import re
import nltk
import pickle

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Download stopwords
nltk.download('stopwords')

# Load dataset
cols = ['sentiment', 'id', 'date', 'query', 'user', 'text']
df = pd.read_csv(
    "training.1600000.processed.noemoticon.csv",
    encoding="latin-1",
    header=None,
    names=cols
)

df = df[['sentiment', 'text']]

# Optional: reduce size for faster training
df = df.sample(200000, random_state=42)

# Text preprocessing
ps = PorterStemmer()
stop_words = set(stopwords.words("english"))

def clean_tweet(text):
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = text.lower()
    words = [ps.stem(w) for w in text.split() if w not in stop_words]
    return " ".join(words)

df['clean_text'] = df['text'].apply(clean_tweet)

# Feature extraction
tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(df['clean_text'])
y = df['sentiment']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model training
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluation
pred = model.predict(X_test)
accuracy = accuracy_score(y_test, pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Save model and vectorizer
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("tfidf.pkl", "wb") as f:
    pickle.dump(tfidf, f)

print("Model and TF-IDF saved successfully!")
