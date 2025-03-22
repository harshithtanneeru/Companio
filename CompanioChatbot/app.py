import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Load trained models
sentiment_model = tf.keras.models.load_model("sentiment_model.h5")
emotion_model = tf.keras.models.load_model("emotion_model.h5")

# Load tokenizer and label encoders
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)
with open("sentiment_encoder.pkl", "rb") as f:
    sentiment_encoder = pickle.load(f)
with open("emotion_encoder.pkl", "rb") as f:
    emotion_encoder = pickle.load(f)

# Function to predict sentiment
def predict_sentiment(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=100)
    prediction = np.argmax(sentiment_model.predict(padded), axis=1)
    return sentiment_encoder.inverse_transform(prediction)[0]

# Function to predict emotion
def predict_emotion(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=100)
    prediction = np.argmax(emotion_model.predict(padded), axis=1)
    return emotion_encoder.inverse_transform(prediction)[0]

# Streamlit UI
st.title("Mental Health Support Chatbot")
st.write("Enter your thoughts below, and the AI will assess your sentiment and emotional state.")

user_input = st.text_area("Your Message")

if st.button("Analyze"):
    if user_input:
        sentiment = predict_sentiment(user_input)
        emotion = predict_emotion(user_input)

        st.subheader("Analysis Result:")
        st.write(f"*Sentiment:* {sentiment}")
        st.write(f"*Emotion:* {emotion}")

        # Provide recommendations based on emotion
        recommendations = {
            "Neutral": "You are completely alright .Just thing of some of your happy memories to feel even better",
            "Stress": "Try deep breathing exercises.",
            "Anxiety": "Practice mindfulness meditation.",
            "Sadness": "Listen to calming music.",
            "Happy": "Engage in a gratitude journal.",
            "Happiness": "Engage in a gratitude journal.",
            "Fear": "Use grounding techniques (5-4-3-2-1 method).",
            "Excitement": "Channel energy into a creative activity."
        }
        
        if emotion in recommendations:
            st.subheader("Suggested Coping Strategy:")
            st.write(recommendations[emotion])
    else:
        st.warning("Please enter a message.")