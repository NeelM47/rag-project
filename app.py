# File: app.py

import streamlit as st
from chatbot_logic import load_vector_store, create_rag_chain

# ==================================================================
# App Configuration
# ==================================================================
st.set_page_config(
    page_title="CV-Bot: Your Personal Career Assistant",
    page_icon="🤖",
    layout="centered"
)

# ==================================================================
# Main App UI
# ==================================================================

# --- Title and Header ---
st.title("🤖 CV-Bot: Your Personal Career Assistant")
st.write("""
Welcome! I am a chatbot built to answer questions about John Doe's professional experience.
I use a Retrieval-Augmented Generation (RAG) system to ensure my answers are based
directly on his CV and other professional documents. Ask me anything about his skills, projects, or work history!
""")

# --- Load the RAG Chain (with caching) ---
# st.cache_resource is a special Streamlit function that runs a function once
# and caches the result. This prevents our app from reloading the model every
# time the user does something.
@st.cache_resource
def load_chain():
    vector_store = load_vector_store()
    return create_rag_chain(vector_store)

chain = load_chain()

# --- User Input Section ---
st.header("Ask a Question")
user_query = st.text_input("e.g., What is his experience with Python and cloud technologies?")

# --- Generate and Display Response ---
if user_query: # This block runs only when the user has typed something
    with st.spinner("Thinking..."): # Shows a nice loading spinner
        try:
            # Call the RAG chain using the invoke method
            result = chain.invoke(user_query)
            
            # Display the answer
            st.subheader("Answer:")
            st.write(result['result'])

        except Exception as e:
            st.error(f"An error occurred: {e}")
