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

st.title("🤖 CV-Bot: Your Personal Career Assistant")
st.write("""
Welcome! I am a chatbot built to answer questions about  Neel More's professional experience.
Ask me anything about his skills, projects, or work history!
""")

# --- Load the RAG Chain (with caching) ---
@st.cache_resource
def load_chain():
    vector_store = load_vector_store()
    return create_rag_chain(vector_store)

chain = load_chain()

# --- NEW: Initialize Chat History ---
# st.session_state is a dictionary-like object that persists across reruns.
# We initialize our 'messages' list in it if it doesn't already exist.
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- NEW: Display Chat History ---
# Loop through the stored messages and display them on the screen.
for message in st.session_state.messages:
    with st.chat_message(message["role"]): # "user" or "assistant"
        st.markdown(message["content"])

# --- NEW: User Input at the Bottom ---
# Use st.chat_input, which is designed for chat applications.
if user_query := st.chat_input("What is his experience with Python?"):
    # Add user message to session state and display it
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # Generate and display the assistant's response
    with st.spinner("Thinking..."):
        try:
            # Call the RAG chain
            result = chain.invoke(user_query)
            response_text = result['result']
            
            # Add assistant response to session state and display it
            st.session_state.messages.append({"role": "assistant", "content": response_text})
            with st.chat_message("assistant"):
                st.markdown(response_text)

        except Exception as e:
            error_message = f"An error occurred: {e}"
            st.session_state.messages.append({"role": "assistant", "content": error_message})
            with st.chat_message("assistant"):
                st.error(error_message)
