# SummarizeAI: Text Summarization Using Pre-trained Models

This project implements an abstractive text summarization system using pre-trained transformer models such as BART and Pegasus. It includes exploratory analysis, notebook-based inference, and a Streamlit dashboard for interactive summarization.

## Features
- Exploratory data analysis for article and summary length distributions.
- Notebook workflow for step-by-step model inference.
- Streamlit dashboard for manual text entry and file-based summarization.
- Support for `.txt`, `.pdf`, and `.docx` uploads.
- Configurable generation settings such as `max_length`, `min_length`, and `num_beams`.
- Download generated summaries as text files.

## Setup and Installation

### 1. Prerequisites
- Python 3.10 or higher
- `pip`

### 2. Create a virtual environment
```bash
python -m venv venv
.\venv\Scripts\activate
```

### 3. Install dependencies
Install the full project dependencies:
```bash
pip install -r requirements.txt
```

If you only need the Streamlit dashboard, install the lighter dashboard-only dependencies:
```bash
pip install -r requirements-dashboard.txt
```

The dashboard uses `pypdf` as the primary PDF dependency and falls back to `PyPDF2` when available. This avoids common issues with older PDF packages on newer environments.

## Usage

### Run the Streamlit dashboard
```bash
streamlit run src/ui/app.py
```

### Run the inference notebook
1. Open `notebooks/inference_pipeline.ipynb` in VS Code or Jupyter.
2. Select the project virtual environment as the notebook kernel.
3. Run the cells to load the model and generate summaries.

## Deployment

This app can be deployed on Streamlit Community Cloud:
1. Push the repository to GitHub.
2. Create a new Streamlit app.
3. Set `src/ui/app.py` as the entry point.
4. Use `requirements-dashboard.txt` if you want a smaller deployment footprint for the UI.

## Project Structure
```text
.streamlit/               # Streamlit configuration
config.yaml               # Model and data configuration
docs/                     # Project plans and notes
notebooks/                # EDA and inference notebooks
reports/                  # Findings and reports
requirements.txt          # Full project dependencies
requirements-dashboard.txt # Streamlit dashboard dependencies
src/ui/                   # Streamlit application
src/utils/                # Shared utilities
README.md                 # Project documentation
```

## Model Information

The app supports:
- `facebook/bart-base`
- `facebook/bart-large-cnn`
- `sshleifer/distilbart-cnn-12-6`
- `google/pegasus-xsum`

The summarization input is truncated to 1024 tokens and model loading is cached with `st.cache_resource`.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
