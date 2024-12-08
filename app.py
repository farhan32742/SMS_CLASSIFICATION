import streamlit as st
import pickle
import re
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load pre-trained models and word2vec model
word2vec_model = pickle.load(open('word2vec.pkl', 'rb'))
rf_model = pickle.load(open('model.pkl', 'rb'))

# Download NLTK resources
import nltk
nltk.download('punkt')
nltk.download('stopwords')

# Define preprocessing functions
def remove_html_tags(data):
    pattern = re.compile('<.*?>')
    return pattern.sub(r'', data)

def remove_punct_alphanumeric(text):
    tokens = word_tokenize(text)
    cleaned_tokens = [word for word in tokens if word.isalpha()]
    return " ".join(cleaned_tokens)

def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Remove HTML tags
    text = remove_html_tags(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    text = " ".join([word for word in text.split() if word not in stop_words])
    # Remove punctuation and alphanumeric characters
    text = remove_punct_alphanumeric(text)
    return text

# Define function to generate the average embedding for each document
def document_vector(doc):
    doc = [word for word in doc if word in word2vec_model.wv]
    return np.mean(word2vec_model.wv[doc], axis=0) if doc else np.zeros(word2vec_model.vector_size)

# Streamlit app layout
st.title("Text Classification Web App")
st.write("Enter a text to classify:")

# User input
input_text = st.text_area("Text", "")

# Button to trigger prediction
if st.button("Classify Text"):
    if input_text:
        # Preprocess the input text
        cleaned_text = preprocess_text(input_text)
        tokenized_text = word_tokenize(cleaned_text)

        # Generate document vector (average word embeddings)
        input_vector = document_vector(tokenized_text)

        # Prepare input for prediction (reshape if needed for model)
        input_vector = input_vector.reshape(1, -1)  # Reshape for a single input

        # Make prediction using the trained RandomForest model
        prediction = rf_model.predict(input_vector)

        # Display prediction
        if prediction[0] == 0:
            st.write("Prediction: Ham")
        else:
            st.write("Prediction: Spam")

        # If you want to show more details, you can also display confidence scores or other metrics
        prediction_proba = rf_model.predict_proba(input_vector)
        st.write("Prediction Probabilities:", prediction_proba)
    else:
        st.write("Please enter some text to classify.")
