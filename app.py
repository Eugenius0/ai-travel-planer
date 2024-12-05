import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU for CPU-only environments
import streamlit as st
from unsloth import FastLanguageModel

#test comment, test

# Model configuration
model_name_or_path = "Eugenius0/lora_model"  # Replace with your Hugging Face model name
max_seq_length = 2048
dtype = None  # Keep it None to auto-detect for CPU environments

# Load the model and tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name_or_path,
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=True  # Use 4-bit quantization for memory efficiency
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
    ).to("cpu")  # Use CPU explicitly

    outputs = model.generate(
        input_ids=inputs, max_new_tokens=128, use_cache=True
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Streamlit app layout
st.title("AI-Powered Travel Planner 🌍")
st.subheader("Explore cities and plan your trips effortlessly!")

# User input
city = st.text_input("Enter a city or region:")
preferences = st.text_input("Enter your travel preferences (e.g., 'historical sites, food'):")

# Generate response
if st.button("Generate Travel Plan"):
    if city:
        prompt = f"Create a travel plan for {city} based on {preferences}."
        response = chatbot_response(prompt)
        st.text_area("Your Travel Plan:", value=response, height=200)
    else:
        st.warning("Please enter a city to proceed.")
