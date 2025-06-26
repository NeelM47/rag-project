from icecream import ic
ic.configureOutput(prefix=f'Debug | ', includeContext=True)

import os
import datetime
import torch
# import torch.nn as nn # Not needed for pre-trained model architecture
# import torch.optim as optim # Not needed for inference
# from torch.utils.data import Dataset, DataLoader # Not needed for simple inference
# from torch.optim.lr_scheduler import ReduceLROnPlateau # Not needed for inference
import math
import inspect

# Import for pre-trained models
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Memorized_Speech is not directly used for training the pre-trained LLM,
# but we can use it as a source for example prompts or context in a RAG scenario.
Memorized_Speech = """
Four score and seven years ago our fathers brought forth on this continent, a new nation, conceived in Liberty, and dedicated to the proposition that all men are created equal.
Now we are engaged in a great civil war, testing whether that nation, or any nation so conceived and so dedicated, can long endure. We are met on a great battle-field of that war. We have come to dedicate a portion of that field, as a final resting place for those who here gave their lives that that nation might live. It is altogether fitting and proper that we should do this.
But, in a larger sense, we can not dedicate - we can not consecrate - we can not hallow-this ground. The brave men, living and dead, who struggled here, have consecrated it, far above our poor power to add or detract. The world will little note, nor long remember what we say here, but it can never forget what they did here. It is for us the living, rather, to be dedicated here to the unfinished work which they who fought here have thus far so nobly advanced. It is rather for us to be here dedicated to the great task remaining before us - that from these honored dead we take increased devotion to that cause for which they gave the last full measure of devotion - that we here highly resolve that these dead shall not have died in vain - that this nation, under God, shall have a new birth of freedom - and that government of the people, by the people, for the people, shall not perish from the earth.
"""

# --- Tokenizer (Using the pre-trained LLM's tokenizer is preferred) ---
# We'll load the LLM's tokenizer in main()
# The custom Tokenizer class below is kept for conceptual understanding if needed,
# but for pre-trained models, always use THEIR tokenizer.

# --- (ToyGPT2, CausalSelfAttention, Block, Dataset, Trainer classes are REMOVED as we are not training from scratch) ---

# --- Main Execution ---
def main():
    # Determine device (GPU if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load Pre-trained LLM and its Tokenizer ---
    model_name = "meta-llama/Llama-2-7b-chat-hf" # Or your desired open-source model
    print(f"Loading model: {model_name}...")

    # Configure Quantization (adjust as needed for your hardware)
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16  # or torch.float16
        # bnb_4bit_use_double_quant=True,
        # bnb_4bit_quant_type="nf4",
    )

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto",  # Automatically distributes model across GPU/CPU
            # torch_dtype=torch.bfloat16 # Optional: can improve performance if supported
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please ensure you have sufficient VRAM/RAM and have accepted Hugging Face terms for gated models like Llama-2.")
        return

    model.eval()  # Set model to evaluation mode (very important for inference)
    print(f"Model {model_name} loaded successfully.")
    if hasattr(model, 'hf_device_map'):
        print(f"Model device map: {model.hf_device_map}")
    else:
        print(f"Model is on device: {next(model.parameters()).device}")


    # --- Example 1: Simple Question Answering (like your initial code) ---
    print("\n--- Example 1: Simple Question Answering ---")
    prompt1 = "What is the capital of France?"
    print(f"Prompt: {prompt1}")

    # Tokenize the prompt using the LLM's tokenizer
    # Inputs need to be on the same device as the model (or its first module if device_mapped)
    input_ids1 = tokenizer.encode(prompt1, return_tensors="pt").to(model.device)

    with torch.no_grad(): # Disable gradient calculations for inference
        output1 = model.generate(
            input_ids1,
            max_new_tokens=50, # Generate new tokens beyond the prompt
            temperature=0.7,
            do_sample=True # Enable sampling for more creative responses
        )

    # Decode only the generated part, not the input prompt
    generated_text1_ids = output1[0][input_ids1.shape[-1]:]
    generated_text1 = tokenizer.decode(generated_text1_ids, skip_special_tokens=True)
    print(f"Response:\n{generated_text1}")


    # --- Example 2: RAG-like scenario (using Gettysburg Address as context) ---
    print("\n--- Example 2: RAG-like Scenario ---")
    user_query = "What did our fathers bring forth on this continent?"

    # In a real RAG, this context would be retrieved from a vector store
    # For this example, we'll just take a snippet of Memorized_Speech
    retrieved_context = "Four score and seven years ago our fathers brought forth on this continent, a new nation, conceived in Liberty, and dedicated to the proposition that all men are created equal."

    # Construct the augmented prompt
    rag_prompt = f"""Based on the following context, answer the question.
Context: {retrieved_context}
Question: {user_query}
Answer:"""
    print(f"\nAugmented Prompt:\n{rag_prompt}")

    input_ids_rag = tokenizer.encode(rag_prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_rag = model.generate(
            input_ids_rag,
            max_new_tokens=100,
            temperature=0.1, # Lower temperature for more factual, less creative answers based on context
            do_sample=False, # For more deterministic output based on context (can also try True with low temp)
            num_beams=1      # Use greedy search if do_sample is False
        )

    generated_text_rag_ids = output_rag[0][input_ids_rag.shape[-1]:]
    generated_text_rag = tokenizer.decode(generated_text_rag_ids, skip_special_tokens=True)
    print(f"\nRAG-style Response:\n{generated_text_rag.strip()}")


    # --- Example 3: Chat-style interaction (if using a chat model like Llama-2-chat-hf) ---
    # Chat models often expect a specific prompt format.
    # You'll need to consult the model card on Hugging Face for the correct chat template.
    # For Llama-2-chat, a simplified version might look like:
    print("\n--- Example 3: Chat-style Interaction (Llama-2-chat specific format) ---")

    # Llama-2 Chat Prompt Template (Simplified for this example)
    # Real applications should use tokenizer.apply_chat_template if available and correct
    chat_history = [
        {"role": "user", "content": "Hello, who are you?"},
        {"role": "assistant", "content": "I am a large language model."}, # Simulate previous turn
        {"role": "user", "content": "Tell me a short story about a brave knight."}
    ]

    # This is a very basic way to format. For robust chat, use tokenizer.apply_chat_template
    # if your transformers version supports it for this model.
    prompt_parts = []
    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
    DEFAULT_SYSTEM_PROMPT = "You are a helpful, respectful and honest assistant."

    formatted_prompt_chat = ""
    if chat_history[0]['role'] == "system":
        formatted_prompt_chat += B_SYS + chat_history[0]['content'] + E_SYS
        chat_history = chat_history[1:]

    for i, turn in enumerate(chat_history):
        if turn['role'] == 'user':
            formatted_prompt_chat += f"{B_INST} {turn['content'].strip()} {E_INST}"
        elif turn['role'] == 'assistant':
            # Add a space before assistant's response if it's not the first message after system prompt
            if i > 0 or (i==0 and B_SYS not in formatted_prompt_chat):
                 formatted_prompt_chat += " "
            formatted_prompt_chat += f"{turn['content'].strip()} " # Llama expects a space after assistant's response before next user turn

    # Ensure the last part is the user's query ready for completion
    if not formatted_prompt_chat.endswith(E_INST): # If last was assistant, add a dummy user start
         # This case usually means we need to append the assistant's response directly after the prompt.
         # For generation, the prompt should typically end with the user's query within [INST] [/INST]
         # or just before the assistant is supposed to speak.
         # If the last message was an assistant, we are asking the model to continue that, or start a new user turn.
         # For this demo, we ensure it ends as if the assistant is about to speak to complete the last user turn.
         if chat_history[-1]['role'] == 'user':
             pass # Already ends with user's turn within INST tags
         elif chat_history[-1]['role'] == 'assistant':
              # For generation, we want the model to generate the *next* turn, which would be after user's prompt.
              # If the last turn was an assistant, we might just print it, or ask a new user question.
              # For this example, let's assume we're generating a response to the last user prompt in history.
              pass


    print(f"Formatted Chat Prompt (ensure this matches Llama-2's expected format):\n{formatted_prompt_chat}")
    input_ids_chat = tokenizer.encode(formatted_prompt_chat, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_chat = model.generate(
            input_ids_chat,
            max_new_tokens=150,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            eos_token_id=tokenizer.eos_token_id # Stop generation at end of sentence token
        )
    generated_text_chat_ids = output_chat[0][input_ids_chat.shape[-1]:]
    generated_text_chat = tokenizer.decode(generated_text_chat_ids, skip_special_tokens=True)
    print(f"\nChat Response:\n{generated_text_chat.strip()}")


if __name__ == "__main__":
    main()
