# Task 6: Question Answering with Transformers Project Checklist

## Phase I: Project Initialization and Setup
- [x] Initialize Python virtual environment and set up project directory structure.
- [x] Create `requirements.txt` with necessary libraries: `transformers`, `tokenizers`, `pandas`, `torch` or `tensorflow`, `datasets`.
- [x] Set up tracking/logging mechanisms (e.g., simple python logging or MLflow).
- [x] Create initial configuration file (e.g., `config.yaml` or `config.json`) for model paths and hyperparameters.

## Phase II: Data Acquisition and Preparation
- [ ] Load the SQuAD v1.1 Dataset from Hugging Face `datasets` library or direct download from Kaggle/Stanford.
- [ ] Perform exploratory data analysis (EDA) using `pandas` to understand context lengths, question types, and answer locations.
- [ ] Implement data preprocessing scripts to handle tokenization using Hugging Face `tokenizers`.
- [ ] Handle max token length limits: implement logic to split long contexts (sliding window approach) or truncate strings.
- [ ] Map answer character spans to token spans for model training/evaluation.

## Phase III: Model Setup and Inference Pipeline
- [ ] Select and load a pre-trained Transformer Model fined-tuned for QA (e.g., `bert-base-uncased-squad` or `distilbert-base-cased-distilled-squad`).
- [ ] Construct the inference pipeline that tokenizes a (context, question) pair.
- [ ] Implement PyTorch/TensorFlow logic to feed inputs into the model and retrieve start & end logits.
- [ ] Develop span extraction logic: find the most probable start and end token indices from the logits.
- [ ] Convert the extracted token span back into the readable string answer.

## Phase IV: Evaluation Strategy implementation
- [ ] Implement standard QA metrics:
  - [ ] Exact Match (EM) evaluation calculation.
  - [ ] F1 Score evaluation calculation.
- [ ] Run evaluation on the full SQuAD v1.1 validation dataset.
- [ ] Document model performance metrics.

## Phase V: Bonus Features & User Interface
- [ ] Compare alternative base models (e.g., RoBERTa, ALBERT, ELECTRA) with the baseline BERT/DistilBERT approach and log performance differences.
- [ ] Develop a Streamlit application or an interactive Command Line Interface (CLI) allowing users to dynamically input a passage and ask multiple questions about it.
- [ ] Document final findings and how to run the project in `README.md`.
