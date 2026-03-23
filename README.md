# Text Summarization Using Pre-trained Models

This project implements an abstractive text summarization system using the **BART (Bidirectional and Auto-Regressive Transformers)** model. It includes an exploratory data analysis of the CNN/DailyMail dataset, an interactive inference notebook, and a Streamlit-based web interface.

## 🚀 Features
- **Exploratory Data Analysis**: Jupyter Notebooks analyzing article and summary lengths.
- **Deep Inference Pipeline**: Step-by-step summary generation using the BART-base model.
- **Interactive Web App**: A user-friendly Streamlit interface to paste and summarize articles in real-time.
- **Configurable**: Model parameters (max length, min length) are managed via `config.yaml`.

## 🛠️ Setup & Installation

### 1. Prerequisites
- Python 3.10 or higher
- `pip` (Python package manager)

### 2. Environment Initialization
Clone or download the project, then create and activate a virtual environment:
```bash
# Create venv
python -m venv .venv

# Activate (Windows)
.\.venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

## 📈 Usage

### Run the Inference Notebook
To experiment with the model and see how it works internally:
1. Open `notebooks/inference_pipeline.ipynb` in VS Code or Jupyter.
2. Ensure you have selected the `.venv` kernel.
3. Run the cells to load the model and generate summaries.

### Run the Streamlit Web Application
For a clean, UI-based summarization experience:
```bash
streamlit run src/ui/app.py
```

## 🏗️ Project Structure
```text
├── config.yaml          # Model and data configurations
├── docs/                # Project plans and checklists
├── notebooks/           # EDA and Inference notebooks
├── reports/             # Phase-wise findings and results
├── requirements.txt     # Python dependencies
├── src/
│   ├── ui/              # Streamlit application
│   └── utils/           # Shared utilities (logger, etc.)
└── README.md            # This documentation
```

## 🧠 Model Information
This project utilizes the `facebook/bart-base` model, which is pre-trained on the CNN/DailyMail dataset.
- **Model Size**: ~558 MB
- **Context Window**: 1024 tokens
- **Output Limit**: 128 tokens (configurable in `config.yaml`)

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
