import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# Load the dataset
df = pd.read_csv("training_data.csv")

# Preprocess the questions
def clean_text(text):
    """
    Cleans the text by removing special characters, extra spaces, and lowercasing.
    """
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)  # Remove special characters
    text = re.sub(r"\s+", " ", text).strip()  # Remove extra spaces
    return text

df['question'] = df['question'].apply(clean_text)

# Create a TF-IDF Vectorizer
vectorizer = TfidfVectorizer(stop_words='english')  # Use stop words for better vectorization
tfidf_matrix = vectorizer.fit_transform(df['question'])

def get_similar_answer(input_question, threshold=0.2):
    """
    Finds the best matching answer based on cosine similarity.
    """
    # Preprocess the input question
    input_question = clean_text(input_question)
    
    # Vectorize the input question
    input_vector = vectorizer.transform([input_question])
    
    # Compute cosine similarities
    similarities = cosine_similarity(input_vector, tfidf_matrix).flatten()
    
    # Find the most similar question index
    most_similar_idx = similarities.argmax()
    highest_similarity = similarities[most_similar_idx]
    
    # Return an answer if similarity is above the threshold
    if highest_similarity >= threshold:
        return df.iloc[most_similar_idx]['answer']
    return f"Sorry, I don't have an answer to that question.{highest_similarity}"

def get_answer(input_question):
    """
    Finds a direct match answer; if no match, falls back on get_similar_answer.
    """
    filtered = df[df['question'].str.contains(input_question, case=False)]
    if not filtered.empty:
        return filtered['answer'].iloc[0] 
    return get_similar_answer(input_question)

st.header("Speak Out AI")
st.write("I am Speak out AI which is made by Mr. G. Omprakash from EEC,Chennai")
st.write("My aim is to provide you a better mental health analysis")
st.write("To help you out from mental health issues and feel free to talk.")
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display the existing chat messages via `st.chat_message`.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Create a chat input field to allow the user to enter a message. This will display
# automatically at the bottom of the page.
if prompt := st.chat_input("What is up?"):
    # Store and display the current prompt.
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    o=get_answer(prompt)+"\n"
    # Generate a response using the OpenAI API.
    stream = [o,"\nCurrently We are working on it"]
    # Stream the response to the chat using `st.write_stream`, then store it in 
    # session state.
    with st.chat_message("assistant"):
        response = st.write_stream(stream)
    st.session_state.messages.append({"role": "assistant", "content": response})