import os
from pathlib import Path

import streamlit as st
import torch
import yaml
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

try:
    from pypdf import PdfReader
except ImportError:
    from PyPDF2 import PdfReader

try:
    import docx
except ImportError:
    docx = None


st.set_page_config(
    page_title="SummarizeAI",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)


def local_css(file_name):
    if os.path.exists(file_name):
        with open(file_name, encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def extract_text_from_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text


def extract_text_from_docx(file):
    if docx is None:
        raise ImportError("python-docx is not installed.")
    document = docx.Document(file)
    return "\n".join(paragraph.text for paragraph in document.paragraphs)


@st.cache_data
def load_config():
    config_paths = [
        Path("config.yaml"),
        Path(__file__).parent.parent.parent / "config.yaml",
        Path("../../config.yaml"),
    ]
    for path in config_paths:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
    return {
        "model": {"name": "facebook/bart-base", "max_output_length": 128},
        "logging": {"project_name": "Summarization_App"},
    }


@st.cache_resource
def load_model_and_tokenizer(model_name):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    return model, tokenizer, device


css_path = Path(__file__).parent / "style.css"
local_css(str(css_path))
config = load_config()


st.sidebar.image("https://img.icons8.com/clouds/100/000000/document.png", width=100)
st.sidebar.title("Settings")

models_list = [
    "facebook/bart-base",
    "facebook/bart-large-cnn",
    "sshleifer/distilbart-cnn-12-6",
    "google/pegasus-xsum",
]
selected_model = st.sidebar.selectbox("Select Model", models_list, index=1)

st.sidebar.markdown("---")
st.sidebar.subheader("Inference Parameters")

max_len = st.sidebar.slider("Max Length", 10, 500, config["model"]["max_output_length"])
min_len = st.sidebar.slider("Min Length", 5, 100, 20)
num_beams = st.sidebar.slider("Number of Beams", 1, 10, 4)

with st.sidebar:
    with st.spinner(f"Loading {selected_model}..."):
        try:
            model, tokenizer, device = load_model_and_tokenizer(selected_model)
            st.success(f"Model loaded on {device}!")
        except Exception as e:
            st.error(f"Failed to load model: {e}")
            st.info("If dependencies are missing, install `requirements-dashboard.txt` first.")
            st.stop()


st.title("SummarizeAI")
st.markdown("##### *Transform long documents into concise summaries using state-of-the-art AI.*")

tab1, tab2 = st.tabs(["Manual Input", "File Upload"])

input_text = ""

with tab1:
    input_text = st.text_area(
        "Paste your text here:",
        height=300,
        placeholder="Enter or paste the text you want to summarize...",
    )

with tab2:
    uploaded_file = st.file_uploader("Upload a document", type=["txt", "pdf", "docx"])
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".txt"):
                input_text = uploaded_file.read().decode("utf-8")
            elif uploaded_file.name.endswith(".pdf"):
                input_text = extract_text_from_pdf(uploaded_file)
            elif uploaded_file.name.endswith(".docx"):
                input_text = extract_text_from_docx(uploaded_file)
        except Exception as e:
            st.error(f"Failed to read file: {e}")

        if input_text:
            st.text_area("File Content Preview:", value=input_text, height=200, disabled=True)


if st.button("Generate Summary", type="primary"):
    if not input_text.strip():
        st.warning("Please provide some text to summarize.")
    else:
        with st.status("Summarizing text...", expanded=True) as status:
            try:
                st.write("Tokenizing input...")
                if "t5" in selected_model.lower():
                    input_text = "summarize: " + input_text

                inputs = tokenizer(
                    input_text,
                    return_tensors="pt",
                    max_length=1024,
                    truncation=True,
                ).to(device)

                st.write("Generating summary (this may take a few seconds)...")
                summary_ids = model.generate(
                    inputs["input_ids"],
                    max_length=max_len,
                    min_length=min_len,
                    num_beams=num_beams,
                    no_repeat_ngram_size=3,
                    length_penalty=2.0,
                    early_stopping=True,
                )

                status.update(label="Summary generated", state="complete", expanded=False)

                st.markdown("### Summary Output")
                summary_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
                st.markdown(f"> {summary_text}")

                col_download, _ = st.columns([1, 4])
                with col_download:
                    st.download_button(
                        label="Download as TXT",
                        data=summary_text,
                        file_name="summary.txt",
                        mime="text/plain",
                    )

                st.info(
                    f"Summary length: {len(summary_text.split())} words | "
                    f"Original length: {len(input_text.split())} words"
                )

            except Exception as e:
                status.update(label="Error occurred during summarization", state="error")
                st.error(f"Details: {e}")


st.markdown("---")
st.caption("Powered by Hugging Face Transformers | Built with Streamlit")
