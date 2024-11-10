import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gtts import gTTS
import os

# Load data
data = pd.read_csv("train.csv")

# Use 'Question' as the context and 'Answer' as the answer column
question_column = "Question"
answer_column = "Answer"

# Vectorize questions
vectorizer = TfidfVectorizer()
question_vectors = vectorizer.fit_transform(data[question_column])

# Function to get the best answer based on similarity
def get_best_answer(user_question):
    user_question_vector = vectorizer.transform([user_question])
    similarities = cosine_similarity(user_question_vector, question_vectors)
    best_match_index = similarities.argmax()
    predicted_answer = data.iloc[best_match_index][answer_column]
    return predicted_answer

# Streamlit UI for the Q&A System
st.title("Healthcare Question Answering System with Text-to-Speech")
user_question = st.text_input("Ask a healthcare-related question:")

if st.button("Get Answer"):
    if user_question:
        answer = get_best_answer(user_question)
        st.write("Answer:", answer)
        
        # Convert answer text to speech
        tts = gTTS(answer)
        tts.save("answer.mp3")
        st.audio("answer.mp3", format="audio/mp3")
    else:
        st.write("Please enter a question.")
