import os
import streamlit as st
from unsloth import FastLanguageModel

# Set environment to disable GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Model configuration
model_name_or_path = "Eugenius0/lora_model" 
max_seq_length = 2048
dtype = None  # Keep it None to auto-detect for CPU environments

# Ensure CPU-only usage by explicitly setting device
import torch
torch.device("cpu")

# Load the model and tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name_or_path,
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=True,  # Use 4-bit quantization for efficiency
)
FastLanguageModel.for_inference(model)  # Enable native 2x faster inference

# Function to generate chatbot response
def chatbot_response(prompt):
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to("cpu")  # Force inputs to CPU

    outputs = model.generate(
        input_ids=inputs, max_new_tokens=128, use_cache=True
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Streamlit app layout
st.title("Simple LLM-Powered App")
st.subheader("Ask your LLM anything!")

# User input
user_input = st.text_input("Enter your question:")

# Generate and display response
if st.button("Submit"):
    if user_input:
        response = chatbot_response(user_input)
        st.write("LLM Response:", response)
    else:
        st.warning("Please enter a question.")