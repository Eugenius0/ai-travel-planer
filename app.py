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