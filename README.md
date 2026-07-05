# Resume Screening Agent

An end-to-end Python pipeline that parses resumes (PDF, DOCX, TXT), extracts structured features, computes NLP similarity scores, and ranks candidates against a job description using a trained machine learning model.

## Features

- **Resume Parsing**: Extracts clean text from PDF, DOCX, and TXT files.
- **Feature Extraction**: Captures skills, experience, education, and section presence.
- **NLP Similarity**: Combines TF-IDF lexical similarity with Sentence-BERT semantic similarity.
- **ML Ranking**: XGBoost regressor trained on synthetic data predicts a relevance score (0-100).
- **CLI & Python API**: Run rankings from the command line or import as a module.

## Project Structure

```
Resume Screening Agent/
├── README.md
├── requirements.txt
├── setup.py
├── .gitignore
├── tradeoffs.md
├── data/
│   ├── sample_resumes/          # Sample resumes in TXT/PDF/DOCX
│   ├── sample_job_descriptions/ # Sample JDs
│   └── generate_sample_files.py # Convert TXT samples to PDF/DOCX
├── models/                      # Trained model artifacts
├── notebooks/
│   └── 01_model_development.ipynb
├── src/
│   ├── __init__.py
│   ├── parse_resume.py          # Resume text extraction
│   ├── extract_features.py      # Feature engineering
│   ├── similarity.py            # NLP similarity scoring
│   ├── train_model.py           # Synthetic data + model training
│   ├── rank_candidates.py       # End-to-end inference
│   └── utils.py                 # Helpers and skill taxonomy
└── tests/
    ├── __init__.py
    └── test_pipeline.py
```

## Setup Instructions

### 1. Clone or Create the Project

```bash
cd "Resume Screening Agent"
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate      # On Linux/macOS
venv\Scripts\activate         # On Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Train the Ranking Model

The model is trained on synthetic data generated inside `src/train_model.py`.

```bash
python src/train_model.py
```

This saves the following artifacts to the `models/` directory:
- `ranking_model.json` — trained XGBoost model
- `scaler.joblib` — feature scaler
- `feature_names.json` — feature name list

Note: TF-IDF similarity is computed per resume-JD pair during both training and inference, so no global vectorizer artifact is required.

### 5. Generate Sample PDF/DOCX Files (Optional)

```bash
python data/generate_sample_files.py
```

### 6. Rank Candidates

From the command line:

```bash
python src/rank_candidates.py \
  --jd data/sample_job_descriptions/software_engineer_jd.txt \
  --resumes data/sample_resumes \
  --output ranked_candidates.csv
```

From Python:

```python
from src.rank_candidates import ResumeRanker

ranker = ResumeRanker(models_dir="models")

with open("data/sample_job_descriptions/software_engineer_jd.txt") as f:
    jd = f.read()

results = ranker.rank_from_folder(jd, "data/sample_resumes")
print(results[["rank", "filename", "score"]])
```

### 7. Run Tests

```bash
pytest tests/
```

## Model Performance

The XGBoost regressor is validated on a held-out test set. Example metrics on the generated synthetic data:

- **Mean Absolute Error (MAE)**: ~2.6 points
- **Mean Squared Error (MSE)**: ~11.8

Top predictive features are `experience_fit`, `skill_match_ratio`, and `education_fit`.

> Note: Because the training data is synthetic, real-world performance will depend on the quality and distribution of actual resumes and job descriptions. Replacing the synthetic generator with real labeled hiring data will improve generalization.

## Dependencies

Key libraries:
- `pdfplumber`, `python-docx` — resume parsing
- `sentence-transformers` — semantic embeddings
- `scikit-learn` — TF-IDF, scaling, metrics
- `xgboost` — gradient boosting ranking model
- `pandas`, `numpy` — data manipulation
- `pytest` — testing

## Public GitHub Repository

This project is ready to be pushed to GitHub. Create a new repository and run:

```bash
git init
git add .
git commit -m "Initial commit: Resume Screening Agent"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/resume-screening-agent.git
git push -u origin main
```

Replace `YOUR_USERNAME` with your GitHub username.

## Out of Scope

As per requirements, the following are intentionally excluded:
- User interface or front-end development
- Integration with external job boards or APIs
- Advanced features like interview scheduling or candidate communication

