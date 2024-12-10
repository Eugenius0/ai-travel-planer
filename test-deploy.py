import gradio as gr
from unsloth import FastLanguageModel
from transformers import TextIteratorStreamer
import torch
import threading

# Load the fine-tuned model and tokenizer
model_name_or_path = "Eugenius0/lora_model_tuned"
max_seq_length = 2048
dtype = None

# Detect and set the appropriate device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load the model and tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name_or_path,
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)  # Enable native 2x faster inference

# Define the travel planner response generation logic
def generate_travel_plan(city, preferences, nb_days):
    try:
        prompt = (
            f"Create a travel plan to visit {city} during {nb_days} days, focusing on {preferences}. Include suggested activities, "
            f"landmarks to visit, and any local tips."
        )
        messages = [{"role": "user", "content": prompt}]
        inputs = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to(device)  # Use the detected device

        # Generate the response in a single step
        outputs = model.generate(
            input_ids=inputs,
            max_new_tokens=1024,
            use_cache=True,
            temperature=1.2,
            repetition_penalty=1.1,  # Avoid repetitive loops
            min_p=0.1,
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response

    except Exception as e:
        error_message = f"Error during response generation: {e}"
        print(error_message)
        return error_message

# Simplified Gradio UI for Travel Planner
interface = gr.Interface(
    fn=generate_travel_plan,
    inputs=[
        gr.Textbox(label="City", placeholder="Enter the city you want to visit"),
        gr.Textbox(label="Preferences", placeholder="E.g., historical sites, food, nightlife"),
        gr.Number(label="Trip Duration (Days)", value=1, interactive=True, minimum=1, maximum=7),
    ],
    outputs=gr.Textbox(label="Generated Travel Plan"),
    title="AI Travel Planner",
    description=(
        "Plan your trips with the help of an AI Travel Planner! "
        "Enter the city you want to visit, your preferences, and the duration of your trip, "
        "and get a personalized itinerary tailored to your interests."
    ),
    examples=[
        ["Paris", "art museums, romantic spots", 2],
        ["Tokyo", "anime culture, food, nightlife", 1],
        ["New York", "Broadway, Central Park, landmarks", 3],
    ],
)

# Launch Gradio app
interface.launch(share=True)
