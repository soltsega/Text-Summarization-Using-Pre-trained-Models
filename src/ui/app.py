import streamlit as st
import yaml
import os
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# page config
st.set_page_config(
    page_title="Text Summarizer",
    page_icon="📝",
    layout="wide"
)

# Load configuration
@st.cache_data
def load_config():
    config_path = "config.yaml"
    if not os.path.exists(config_path):
        # Handle case when run from different directory
        config_path = os.path.join("..", "..", "config.yaml")
    
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

config = load_config()
model_name = config['model']['name']

# Load model and pipeline (cached)
@st.cache_resource
def load_summarizer():
    device = 0 if torch.cuda.is_available() else -1
    return pipeline("summarization", model=model_name, device=device)

with st.spinner(f"Loading {model_name}... This might take a minute on first run."):
    summarizer = load_summarizer()

# UI Layout
st.title("📝 AI Text Summarizer")
st.markdown(f"Using pre-trained model: **{model_name}**")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Input Text")
    input_text = st.text_area(
        "Paste your long article or text here:",
        height=400,
        placeholder="Once upon a time in a galaxy cluster far, far away..."
    )
    
    generate_btn = st.button("Generate Summary", type="primary")

with col2:
    st.subheader("Summary")
    if generate_btn and input_text:
        with st.spinner("Summarizing..."):
            try:
                # Basic inference
                summary = summarizer(
                    input_text,
                    max_length=config['model']['max_output_length'],
                    min_length=30,
                    do_sample=False,
                    truncation=True
                )
                st.success("Summary Generated!")
                st.write(summary[0]['summary_text'])
            except Exception as e:
                st.error(f"An error occurred: {e}")
    elif generate_btn and not input_text:
        st.warning("Please enter some text first.")
    else:
        st.info("The summary will appear here once you click 'Generate Summary'.")

# Sidebar info
st.sidebar.title("About")
st.sidebar.info(
    "This app uses the Hugging Face Transformers library to perform abstractive summarization. "
    "It leverages the BART model which is pre-trained on the CNN/DailyMail dataset."
)
