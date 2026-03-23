# SummarizeAI: Text Summarization Using Pre-trained Models

This project implements an abstractive text summarization system using state-of-the-art **Transformer** models (BART, Pegasus). It includes an exploratory data analysis of the CNN/DailyMail dataset, an interactive inference notebook, and a premium Streamlit-based web interface.

## 🚀 Features
- **Exploratory Data Analysis**: Jupyter Notebooks analyzing article and summary lengths.
- **Deep Inference Pipeline**: Step-by-step summary generation using the BART-base model.
- **Premium Web App**: A sleek, dark-themed Streamlit interface for real-time summarization.
- **Multi-Format Upload**: Support for `.txt`, `.pdf`, and `.docx` file summarization.
- **Model Selection**: Choice of state-of-the-art models (BART-Base, BART-Large, Pegasus).
- **Customizable Inference**: Silders for `max_length`, `min_length`, and `num_beams`.
- **Export Options**: Download generated summaries as text files.

## 🛠️ Setup & Installation

### 1. Prerequisites
- Python 3.10 or higher
- `pip` (Python package manager)

### 2. Environment Initialization
Clone or download the project, then create and activate a virtual environment:
```bash
# Create venv
python -m venv venv

# Activate (Windows)
.\venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

## 📈 Usage

### Run the Streamlit Web Application
For the full UI-based summarization experience:
```bash
streamlit run src/ui/app.py
```

### Run the Inference Notebook
To experiment with the model and see how it works internally:
1. Open `notebooks/inference_pipeline.ipynb` in VS Code or Jupyter.
2. Ensure you have selected the virtual environment kernel.
3. Run the cells to load the model and generate summaries.

## 🌐 Deployment

This app is optimized for deployment on **Streamlit Cloud**:
1. Push this repository to GitHub.
2. Sign in to [Streamlit Cloud](https://share.streamlit.io/).
3. Click "New app" and select this repository and `src/ui/app.py` as the main file.
4. The `.streamlit/config.toml` handles the dark theme automatically.

## 🏗️ Project Structure
```text
├── .streamlit/          # Deployment configurations
├── config.yaml          # Model and data configurations
├── docs/                # Project plans and checklists
├── notebooks/           # EDA and Inference notebooks
├── reports/             # Phase-wise findings and results
├── requirements.txt     # Python dependencies
├── src/
│   ├── ui/              # Streamlit application (app.py, style.css)
│   └── utils/           # Shared utilities (logger, etc.)
└── README.md            # This documentation
```

## 🧠 Model Information
Utilizes `facebook/bart-base`, `facebook/bart-large-cnn`, and `google/pegasus-xsum` via Hugging Face.
- **Context Window**: 1024 tokens
- **Inference**: Cached for performance using `st.cache_resource`.

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
