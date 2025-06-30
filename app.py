from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# --- Part 1: Setup and Retrieval (Simulated) ---

# In a real RAG system, you would have this part:
# 1. Load your documents (PDFs, text files, etc.).
# 2. Split them into chunks.
# 3. Use a local embedding model (like from sentence-transformers) to create embeddings.
# 4. Store these embeddings in a local vector store (like ChromaDB or FAISS).
# 5. When a user asks a question, embed the question and retrieve relevant chunks.

# Let's simulate the output of the retrieval step for this example.
user_query = "What did our fathers bring forth on this continent?"
retrieved_context = "Four score and seven years ago our fathers brought forth on this continent, a new nation, conceived in Liberty, and dedicated to the proposition that all men are created equal."

print(f"User Query: {user_query}")
print(f"Retrieved Context: {retrieved_context}\n")

# --- Part 2: Connect to the Local LLM (Ollama) ---

print("Connecting to local Ollama instance...")
# Initialize the Ollama LLM
# Make sure "ollama run orca-mini" is running in your terminal!
# The 'model' parameter should match the name of the model you are running with Ollama.
llm = Ollama(model="orca-mini")
print("Connection successful.")

# --- Part 3: Augmentation and Generation ---

# 1. Create a prompt template. This structures how the context and question are sent to the LLM.
# The template instructs the LLM to answer based *only* on the provided context.
prompt_template_str = """
Use the following context to answer the question at the end. If you don't know the answer from the context, just say that you do not know. Do not use any other information.

Context: {context}

Question: {question}

Answer:"""

prompt_template = PromptTemplate(
    template=prompt_template_str,
    input_variables=["context", "question"]
)

# 2. Create an LLMChain. This chain will take the user's question and the retrieved context,
# format them using the prompt template, and then send the formatted prompt to the Ollama LLM.
chain = LLMChain(llm=llm, prompt=prompt_template)

# 3. Run the chain to generate the answer.
print("\nGenerating answer using RAG pipeline...")
# The chain expects a dictionary with keys matching the input_variables in the prompt template.
response = chain.invoke({
    "context": retrieved_context,
    "question": user_query
})

# The actual generated text is usually in the 'text' key of the response dictionary.
generated_answer = response.get('text', 'No text found in response.')

print(f"\nGenerated Answer:\n{generated_answer.strip()}")


# --- Example of a simple, non-RAG question to show direct interaction ---
print("\n--- Testing a simple, direct question ---")

simple_query = "Why is the sky blue?"
print(f"Direct Query: {simple_query}")

# You can invoke the LLM directly as well
simple_response = llm.invoke(simple_query)
print(f"Direct Response:\n{simple_response.strip()}")
