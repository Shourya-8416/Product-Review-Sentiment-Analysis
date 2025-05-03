import streamlit as st
import pickle
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import re

nltk.download('stopwords')

# Load model and vectorizer
model = pickle.load(open('models/sentiment_model.pkl', 'rb'))
vectorizer = pickle.load(open('models/vectorizer.pkl', 'rb'))

ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = re.sub(r"[^a-zA-Z]", " ", text)
    text = text.lower().split()
    text = [ps.stem(word) for word in text if word not in stop_words]
    return " ".join(text)

st.title("📝 Sentiment Analysis on Product Review")
review = st.text_area("Enter your review:")

if st.button("Predict Sentiment"):
    processed = preprocess(review)
    vec = vectorizer.transform([processed])
    prediction = model.predict(vec)[0]
    st.success(f"Prediction: {'Positive 😊' if prediction else 'Negative 😞'}")
