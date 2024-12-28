import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

# Load the IMDB dataset word index
word_index=imdb.get_word_index()
reverse_word_index={value:key for key,value in word_index.items()}

#Load the pretrained model with relu activation
model=load_model('simple_rnn_imdb_relu.h5')

# Helper Functions
def decode_review(encoded_review):
    return " ".join([reverse_word_index.get(i-3,'?')for i in encoded_review])

# Function to preprocess user input
def preprocess_text(text):
    words=text.lower().split()
    encoded_review=[word_index.get(word,2)+3 for word in words]
    padded_review=sequence.pad_seuences([encoded_review],maxlen=500)
    return padded_review

# Prediction function

def predict_sentiment(review):
    preprocessed_input=preprocess_text(review)

    prediction=model.predict(preprocessed_input)

    sentiment="Positive" if prediction[0][0]>0.5 else "Negative"

    return sentiment,prediction[0][0]

# Streamlit app
import streamlit as st
st.write("IMDB Movie Review Sentiment Analysis")
st.write("Enter a movie to classify it as positive or negative.")


#User INput

user_input =st.text_area("Movie Review")

if st.button("Classify"):
    preprocessed_input=preprocess_text(user_input)

    # Make Prediction
    prediction=model.predict(preprocessed_input)
    sentiment="Positive" if prediction[0][0]>0.5 else "Negative"

    # Display result
    st.write(f"sentiment: {sentiment}")
    st.write(f"Prediction score: {prediction[0][0]}")

else:
    st.write("Please enter a movie review.")
