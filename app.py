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
