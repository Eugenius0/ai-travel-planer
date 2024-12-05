import os
import streamlit as st
from unsloth import FastLanguageModel
from transformers import TextStreamer

# Disable GPU (for CPU-only environments)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Model Configuration
model_name_or_path = "Eugenius0/lora_model"  # Replace with your model path
max_seq_length = 2048
dtype = None  # Let it auto-detect CPU environments

# Load the model and tokenizer
st.title("AI Travel Planner 🌍")
st.subheader("Powered by Your Fine-Tuned LLM")

# Streamlit loading spinner while the model initializes
with st.spinner("Loading the model, please wait..."):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name_or_path,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=True,  # Efficient loading
        device_map="cpu",  # Explicitly use CPU
    )
    FastLanguageModel.for_inference(model)  # Enable faster inference

# Function to get model response
def get_response(prompt):
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to("cpu")  # Use CPU for inference
    streamer = TextStreamer(tokenizer, skip_prompt=True)
    outputs = model.generate(
        input_ids=inputs, streamer=streamer, max_new_tokens=128, use_cache=True
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Streamlit Layout
city = st.text_input("Enter a city or region:")
preferences = st.text_input("Enter your travel preferences (e.g., 'historical sites, food'):")

if st.button("Generate Travel Plan"):
    if city:
        with st.spinner("Generating your travel plan..."):
            prompt = f"Create a travel itinerary for {city} focusing on {preferences}."
            response = get_response(prompt)
        st.text_area("Your AI-Generated Travel Plan:", value=response, height=200)
    else:
        st.warning("Please enter a city to proceed.")
