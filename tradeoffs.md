# Model Choice Tradeoffs

This document explains the key modeling and implementation decisions in the Resume Screening Agent.

## 1. Resume Parsing

### Choices Considered
- **PyMuPDF** vs. **pdfplumber** vs. **PyPDF2**
- **python-docx** for DOCX files

### Decision
Use `pdfplumber` for PDFs and `python-docx` for DOCX files.

### Tradeoffs
- `pdfplumber` preserves layout better than `PyPDF2` and handles multi-column resumes more reliably.
- `PyMuPDF` is faster and sometimes more robust for scanned PDFs, but it is heavier and has a more restrictive license for some use cases.
- `python-docx` is the standard, lightweight choice for DOCX files.
- OCR for scanned/image-based resumes is not included to keep dependencies minimal.

## 2. NLP Similarity Scoring

### Choices Considered
- **TF-IDF + Cosine Similarity**
- **Sentence-BERT (all-MiniLM-L6-v2)**
- **BM25**
- **Hybrid combination**

### Decision
Combine TF-IDF lexical similarity with Sentence-BERT semantic similarity.

### Tradeoffs
- **TF-IDF** is fast, interpretable, and works well when resumes and JDs share exact keywords. It fails to capture synonyms or paraphrased concepts (e.g., "Python developer" vs. "programmer using Python").
- **Sentence-BERT** captures semantic meaning and synonyms, producing more human-like similarity scores. The downside is slower inference and higher memory usage, especially on CPU.
- **BM25** is strong for keyword matching but adds complexity without a clear benefit over TF-IDF for short documents like resumes.
- The **hybrid approach** balances speed and accuracy by feeding both scores (plus engineered features) into the ML model, letting it learn the optimal weighting.

## 3. Machine Learning Model for Ranking

### Choices Considered
- **XGBoost / LightGBM / Gradient Boosting**
- **Random Forest**
- **Logistic Regression / Linear Regression**
- **Neural Network**

### Decision
Use **XGBoost Regressor**.

### Tradeoffs
- **XGBoost** handles heterogeneous structured features well, is fast to train, and provides built-in feature importance. It is a strong baseline for tabular ranking tasks.
- **Random Forest** is simpler and less prone to overfitting, but usually slightly less accurate than gradient boosting.
- **Linear models** are very interpretable but cannot capture non-linear interactions between features (e.g., skill match ratio combined with experience fit).
- **Neural networks** could learn richer representations but require much more labeled data and longer training time. They are also harder to interpret and deploy.

## 4. Training Data

### Choices Considered
- Use a public labeled resume dataset
- Use a pre-trained zero-shot model
- Generate synthetic training data

### Decision
Generate **synthetic training data**.

### Tradeoffs
- Public labeled datasets for resume ranking are scarce, expensive, and often domain-specific.
- Zero-shot models (e.g., cross-encoders) are convenient but do not demonstrate the full ML pipeline and are slower at scale.
- Synthetic data lets us train a complete supervised model. The limitation is that the model may not generalize perfectly to real-world hiring signals. However, the pipeline is designed so that real labeled data can replace the synthetic generator with minimal changes.

## 5. Feature Engineering

### Decision
Engineer interpretable features rather than feeding raw text directly into the model.

### Tradeoffs
- Engineered features (skill match ratio, experience fit, education fit, section presence) are interpretable and robust.
- They require domain knowledge to design but make the model auditable.
- An alternative would be to use a cross-encoder or bi-encoder end-to-end, but that would reduce transparency and make debugging harder.

## 6. Scalability & Deployment

### Tradeoffs
- Sentence-BERT inference is the main bottleneck. For large resume volumes, batching (already implemented) and GPU acceleration would improve throughput.
- The current pipeline runs locally and stores artifacts on disk. A production version could wrap the ranker in a FastAPI service and store artifacts in object storage.
- The model is retrained from synthetic data on demand. In production, retraining should happen on real recruiter feedback (click data, hire outcomes).

## Summary

The chosen stack — `pdfplumber` + `python-docx` for parsing, TF-IDF + Sentence-BERT for similarity, and XGBoost on engineered features — provides a robust, interpretable, and easy-to-deploy baseline. It intentionally prioritizes code clarity and reproducibility over cutting-edge accuracy, while leaving clear extension points for production improvements.
