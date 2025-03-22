import pickle
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer

# Example texts used during training (replace with your real training data)
df=pd.read_excel("mental_health_datasets .xlsx",sheet_name="Sentiment Analysis")
training_texts = df["Text"]
'''[
    "I feel happy today",
    "I am very sad",
    "This is frustrating",
    "I'm so excited",
    "I'm feeling anxious",
    "I love this moment",
    "I'm stressed about my job",
    "I feel hopeful for the future"
]'''

# Create and fit tokenizer
tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(training_texts)

# Save tokenizer to a file
with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

print("✅ Tokenizer saved successfully as tokenizer.pkl!")
