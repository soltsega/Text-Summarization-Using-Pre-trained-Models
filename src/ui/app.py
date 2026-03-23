import streamlit as st
import yaml
import os
from transformers import pipeline
import torch
from pathlib import Path
import PyPDF2
import docx

# Page configuration
st.set_page_config(
    page_title="SummarizeAI",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load CSS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Try to load local CSS if it exists
css_path = Path(__file__).parent / "style.css"
if css_path.exists():
    local_css(str(css_path))
else:
    # Fallback minimal CSS if file not found
    st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stTextArea textarea {
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# File Parsers
def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def extract_text_from_docx(file):
    doc = docx.Document(file)
    text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
    return text

# Load configuration
@st.cache_data
def load_config():
    config_paths = [
        Path("config.yaml"),
        Path(__file__).parent.parent.parent / "config.yaml",
        Path("../../config.yaml")
    ]
    for path in config_paths:
        if path.exists():
            with open(path, "r") as f:
                return yaml.safe_load(f)
    return {
        'model': {'name': 'facebook/bart-base', 'max_output_length': 128},
        'logging': {'project_name': 'Summarization_App'}
    }

config = load_config()

# Sidebar - Settings
st.sidebar.image("https://img.icons8.com/clouds/100/000000/document.png", width=100)
st.sidebar.title("Settings")

# Model selection
models_list = [
    "facebook/bart-base",
    "facebook/bart-large-cnn",
    "sshleifer/distilbart-cnn-12-6",
    "google/pegasus-xsum"
]
selected_model = st.sidebar.selectbox(
    "Select Model",
    models_list,
    index=1 # Default to large cnn for better results
)

st.sidebar.markdown("---")
st.sidebar.subheader("Inference Parameters")

max_len = st.sidebar.slider("Max Length", 50, 500, config['model']['max_output_length'])
min_len = st.sidebar.slider("Min Length", 10, 100, 30)
num_beams = st.sidebar.slider("Number of Beams", 1, 10, 4)

# Load model and pipeline (cached)
@st.cache_resource
def load_summarizer(model_name):
    device = 0 if torch.cuda.is_available() else -1
    return pipeline("summarization", model=model_name, device=device)

with st.sidebar:
    with st.spinner(f"Loading {selected_model}..."):
        summarizer = load_summarizer(selected_model)
    st.success(f"Model loaded!")

# Main UI
st.title("✨ SummarizeAI")
st.markdown("##### *Transform long documents into concise summaries using state-of-the-art AI.*")

tab1, tab2 = st.tabs(["📝 Manual Input", "📁 File Upload"])

input_text = ""

with tab1:
    input_text = st.text_area(
        "Paste your text here:",
        height=300,
        placeholder="Enter or paste the text you want to summarize..."
    )

with tab2:
    uploaded_file = st.file_uploader("Upload a document", type=["txt", "pdf", "docx"])
    if uploaded_file is not None:
        if uploaded_file.name.endswith(".txt"):
            input_text = uploaded_file.read().decode("utf-8")
        elif uploaded_file.name.endswith(".pdf"):
            input_text = extract_text_from_pdf(uploaded_file)
        elif uploaded_file.name.endswith(".docx"):
            input_text = extract_text_from_docx(uploaded_file)
        
        st.text_area("File Content Preview:", value=input_text, height=200, disabled=True)

# Generation Section
if st.button("Generate Summary", type="primary"):
    if not input_text.strip():
        st.warning("Please provide some text to summarize.")
    else:
        with st.status("Summarizing text...", expanded=True) as status:
            try:
                st.write("Processing input...")
                summary = summarizer(
                    input_text,
                    max_length=max_len,
                    min_length=min_len,
                    num_beams=num_beams,
                    do_sample=False,
                    truncation=True
                )
                status.update(label="Summary Generated!", state="complete", expanded=False)
                
                st.markdown("### 🎯 Summary Output")
                summary_text = summary[0]['summary_text']
                st.markdown(f"> {summary_text}")
                
                # Action Buttons
                col_c1, col_c2 = st.columns([1, 4])
                with col_c1:
                    st.download_button(
                        label="Download as TXT",
                        data=summary_text,
                        file_name="summary.txt",
                        mime="text/plain"
                    )
                
                # Show stats
                st.info(f"Summary length: {len(summary_text.split())} words | Original length: {len(input_text.split())} words")
                
            except Exception as e:
                status.update(label="Error occurred", state="error")
                st.error(f"Details: {e}")

# Footer
st.markdown("---")
st.caption("Powered by Hugging Face Transformers • Built with Streamlit")
