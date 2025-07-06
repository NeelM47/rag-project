# File: chatbot_logic.py

import os
from dotenv import load_dotenv

# LangChain Imports
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA

# --- Configuration Constants ---
KNOWLEDGE_BASE_FILE = "knowledge_base.md"
CHROMA_PERSIST_DIR = "chroma_db"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
LLM_MODEL_NAME = "gemini-2.5-pro"

# Load environment variables once
load_dotenv()

def load_vector_store():
    """
    Loads or creates the Chroma vector store.
    Returns:
        Chroma: The loaded or newly created vector store.
    """
    print("Configuring the local Hugging Face embedding model...")
    embeddings_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    
    if os.path.exists(CHROMA_PERSIST_DIR):
        print("Loading existing vector store from disk...")
        vector_store = Chroma(
            persist_directory=CHROMA_PERSIST_DIR,
            embedding_function=embeddings_model
        )
    else:
        print("Creating a new vector store...")
        loader = TextLoader(KNOWLEDGE_BASE_FILE)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        chunks = text_splitter.split_documents(documents)
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings_model,
            persist_directory=CHROMA_PERSIST_DIR
        )
    print("✅ Vector store ready.")
    return vector_store

def create_rag_chain(vector_store):
    """
    Creates the RAG chain for answering questions.
    Args:
        vector_store (Chroma): The vector store containing the document embeddings.
    Returns:
        RetrievalQA: The configured RAG chain.
    """
    print("Configuring the Gemini Pro LLM...")
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        raise ValueError("GEMINI_API_KEY not found in environment variables.")
        
    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL_NAME,
        temperature=0.3,
        google_api_key=gemini_api_key
    )
    
    print("Creating the RAG chain...")
    retriever = vector_store.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever
    )
    print("✅ RAG chain created.")
    return qa_chain
