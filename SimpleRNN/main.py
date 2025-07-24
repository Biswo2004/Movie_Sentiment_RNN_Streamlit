import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

# Load word index and reverse mapping
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

# Load pre-trained RNN model
model = load_model('SimpleRNN/simplernn_imdb_model.h5')

# -- Helper functions --
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

def predict_sentiment(review):
    input_data = preprocess_text(review)
    prediction = model.predict(input_data)
    sentiment = '👍 Positive' if prediction[0][0] > 0.5 else '👎 Negative'
    return sentiment, prediction[0][0]

# -- Streamlit UI --
st.set_page_config(page_title="🎬 Movie Review Sentiment", page_icon="🎞️", layout="centered")

st.markdown("""
    <div style='text-align: center'>
        <h1 style='color: #ff4b4b;'>🎬 IMDB Sentiment Predictor</h1>
        <p>Is your review full of love or loathing? Let’s find out!</p>
    </div>
    """, unsafe_allow_html=True)

# Input text
review_input = st.text_area("✍️ Enter your movie review below:", "I absolutely loved the movie!")

# Predict button
if st.button("🔍 Predict Sentiment"):
    sentiment, probability = predict_sentiment(review_input)

    st.success(f"**Predicted Sentiment:** {sentiment}")
    st.write(f"**Confidence Score:** `{probability:.4f}`")

    # Optional visual bar
    st.progress(float(probability) if sentiment.startswith('👍') else float(1 - probability))

else:
    st.info("Ready when you are! Just type a review and hit 'Predict Sentiment'.")
