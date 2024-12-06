import streamlit as st
import os
from unsloth import FastLanguageModel

# Disable GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Model loading
st.title("AI Travel Planner - LLM Integration")

try:
    model_name = "Eugenius0/lora_model"
    st.write(f"Loading model: {model_name}")
    
    with st.spinner("Loading model..."):
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=2048,
            load_in_4bit=True,  # Efficient for low-memory environments
            device_map="cpu", # mps also doesn't work here
            # More details about the assertion error "Torch not compiled with CUDA enabled": https://github.com/Eugenius0/ai-travel-planner/issues/1
        )
        FastLanguageModel.for_inference(model)  # Enable faster inference

    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {str(e)}")

# User input
user_input = st.text_input("Enter your question:")

# Model response
if st.button("Submit"):
    if user_input:
        try:
            messages = [{"role": "user", "content": user_input}]
            inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to("cpu")
            outputs = model.generate(input_ids=inputs, max_new_tokens=128, use_cache=True)
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            st.write("Response:", response)
        except Exception as e:
            st.error(f"Error during inference: {str(e)}")
    else:
        st.warning("Please enter a question.")
