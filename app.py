import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords')

# Load model & tfidf
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("tfidf.pkl", "rb") as f:
    tfidf = pickle.load(f)

ps = PorterStemmer()
stop_words = set(stopwords.words("english"))

def clean_tweet(text):
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = text.lower()
    words = [ps.stem(w) for w in text.split() if w not in stop_words]
    return " ".join(words)

# Streamlit UI
st.set_page_config(page_title="Tweet Sentiment Analysis")

st.title("🐦 Tweet Sentiment Analysis")
st.write("Enter a tweet to predict sentiment")

tweet = st.text_area("Tweet text")

if st.button("Predict Sentiment"):
    if tweet.strip() == "":
        st.warning("Please enter a tweet")
    else:
        cleaned = clean_tweet(tweet)
        vector = tfidf.transform([cleaned])
        prediction = model.predict(vector)[0]

        sentiment_map = {
            0: "❌ Negative",
            2: "⚪ Neutral",
            4: "✅ Positive"
        }

        st.success(f"Predicted Sentiment: {sentiment_map[prediction]}")
