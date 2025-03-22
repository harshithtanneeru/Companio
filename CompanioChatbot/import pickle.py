import pickle
from sklearn.preprocessing import LabelEncoder

# Define labels (same as used during training)
sentiment_labels = ["Positive", "Negative", "Neutral"]  # Modify if needed
emotion_labels = ["Happy", "Sad", "Fear", "Anger", "Surprise", "Disgust", "Neutral"]  # Modify if needed

# Create label encoders
sentiment_encoder = LabelEncoder()
sentiment_encoder.fit(sentiment_labels)

emotion_encoder = LabelEncoder()
emotion_encoder.fit(emotion_labels)

# Save the encoders
with open("encoders/sentiment_encoder.pkl", "wb") as f:
    pickle.dump(sentiment_encoder, f)

with open("encoders/emotion_encoder.pkl", "wb") as f:
    pickle.dump(emotion_encoder, f)

print("Encoders saved successfully!")
