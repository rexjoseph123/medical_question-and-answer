import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load data (update path as needed)
data = pd.read_csv(r"C:\Users\rexjo\Downloads\archive (1)\train.csv")

# Use 'Question' as the context and 'Answer' as the answer column
question_column = "Question"
answer_column = "Answer"

# Vectorize questions
vectorizer = TfidfVectorizer()
question_vectors = vectorizer.fit_transform(data[question_column])


# Function to get the best answer based on similarity
def get_best_answer(user_question):
    # Vectorize the user's question
    user_question_vector = vectorizer.transform([user_question])

    # Calculate cosine similarity between the user's question and each question in the dataset
    similarities = cosine_similarity(user_question_vector, question_vectors)

    # Get the index of the most similar question
    best_match_index = similarities.argmax()

    # Retrieve the answer corresponding to the most similar question
    predicted_answer = data.iloc[best_match_index][answer_column]
    return predicted_answer


# Streamlit UI for the Q&A System
st.title("Healthcare Question Answering System")
user_question = st.text_input("Ask a healthcare-related question:")

if st.button("Get Answer"):
    if user_question:
        answer = get_best_answer(user_question)
        st.write("Answer:", answer)
    else:
        st.write("Please enter a question.")
