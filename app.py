import streamlit as st
import folium
from streamlit_folium import st_folium
from fpdf import FPDF
from unsloth import FastLanguageModel

from transformers import AutoModel, AutoTokenizer
max_seq_length = 2048
dtype = None

model_name_or_path = "Eugenius0/lora_model"

from unsloth import FastLanguageModel
import torch

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name_or_path, 
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = True,
    )
FastLanguageModel.for_inference(model)  # Enable faster inference

# Function to generate chatbot response
def chatbot_response(prompt):
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to("cuda")  # Ensure the model is on GPU

    outputs = model.generate(
        input_ids=inputs, max_new_tokens=128, use_cache=True
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Function to generate PDF
def generate_pdf(text, filename="itinerary.pdf"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, text)
    pdf.output(filename)
    return filename

# Streamlit app layout
st.title("AI-Powered Personalized Travel Planner 🌍")
st.subheader("Explore cities, plan trips, and create itineraries!")

# User input
city = st.text_input("Enter a city or region:")
preferences = st.text_input("Enter your travel preferences (e.g., 'historical sites, food'):")

# Generate response
if st.button("Generate Travel Plan"):
    if city:
        prompt = f"Create a travel plan for {city} based on {preferences}."
        response = chatbot_response(prompt)
        st.text_area("Your Travel Plan:", value=response, height=200)

        # Show map with Folium
        st.subheader("Map of the Destination:")
        map_center = [48.0, 7.85]  # Replace with actual coordinates
        travel_map = folium.Map(location=map_center, zoom_start=12)
        folium.Marker(location=map_center, popup="Start here!").add_to(travel_map)
        st_folium(travel_map, width=700, height=500)

        # Generate and download PDF
        pdf_file = generate_pdf(response)
        st.download_button("Download Itinerary as PDF", data=open(pdf_file, "rb"), file_name=pdf_file)
    else:
        st.warning("Please enter a city to proceed.")
