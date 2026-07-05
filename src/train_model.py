"""Synthetic data generation and ML model training for resume ranking."""

import json
import os
import pickle
import random
import re
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from extract_features import extract_all_features, get_feature_names, skill_match_features
from similarity import SimilarityScorer
from utils import (
    EDUCATION_LEVELS,
    ALL_SKILLS,
    extract_education_level,
    extract_experience_years,
    extract_required_experience,
    extract_skills,
)

random.seed(42)
np.random.seed(42)


def _sample_skills(pool: List[str], n: int) -> List[str]:
    return random.sample(pool, min(n, len(pool)))


def generate_synthetic_resume() -> Tuple[str, dict]:
    """Generate a synthetic resume text and its metadata."""
    first_names = ["Alice", "Bob", "Charlie", "Diana", "Evan", "Fiona", "George", "Hannah"]
    last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller"]
    roles = [
        "Software Engineer", "Data Scientist", "ML Engineer", "Product Manager",
        "DevOps Engineer", "Data Analyst", "Full Stack Developer", "Backend Developer",
    ]
    companies = ["TechCorp", "DataWorks", "CloudSys", "Innovate Inc", "NextGen Solutions"]
    degrees = ["Bachelor", "Master", "PhD"]
    fields = ["Computer Science", "Information Technology", "Data Science", "Engineering"]

    name = f"{random.choice(first_names)} {random.choice(last_names)}"
    role = random.choice(roles)
    years_exp = random.randint(0, 15)
    num_skills = random.randint(5, 20)
    skills = _sample_skills(list(ALL_SKILLS), num_skills)
    degree = random.choice(degrees)
    field = random.choice(fields)
    company = random.choice(companies)

    skill_text = ", ".join(skills)
    projects = "Projects: " + ", ".join(
        [f"{random.choice(['Built', 'Developed', 'Designed'])} a {random.choice(['web app', 'ML pipeline', 'data dashboard', 'API service'])}"]
    )

    resume_text = f"""
{name}
{role}

Skills:
{skill_text}

Experience:
{role} at {company} - {years_exp} years of experience.
Worked on multiple projects involving {random.choice(skills) if skills else 'software development'}.

Education:
{degree} in {field}

{projects}

Certifications:
{random.choice(['AWS Certified', 'Google Cloud Professional', 'Scrum Master', 'None'])}
"""

    metadata = {
        "name": name,
        "role": role,
        "years_experience": years_exp,
        "skills": set(skills),
        "degree": degree,
        "field": field,
    }
    return resume_text, metadata


def generate_synthetic_jd() -> Tuple[str, dict]:
    """Generate a synthetic job description and its metadata."""
    roles = [
        "Software Engineer", "Data Scientist", "ML Engineer", "Product Manager",
        "DevOps Engineer", "Data Analyst", "Full Stack Developer", "Backend Developer",
    ]
    degrees = ["Bachelor", "Master", "PhD"]
    fields = ["Computer Science", "Information Technology", "Data Science", "Engineering"]

    role = random.choice(roles)
    required_years = random.randint(1, 8)
    num_skills = random.randint(4, 12)
    required_skills = _sample_skills(list(ALL_SKILLS), num_skills)
    required_degree = random.choice(degrees)
    field = random.choice(fields)

    skill_text = ", ".join(required_skills)
    jd_text = f"""
Job Title: {role}

We are looking for a {role} with {required_years}+ years of experience.
Required skills: {skill_text}.
The candidate should have a {required_degree} in {field} or a related field.
Responsibilities include developing scalable solutions, collaborating with teams, and delivering high-quality software.
"""

    metadata = {
        "role": role,
        "required_years": required_years,
        "required_skills": set(required_skills),
        "required_degree": required_degree,
        "field": field,
    }
    return jd_text, metadata


def _compute_true_score(resume_meta: dict, jd_meta: dict) -> float:
    """Heuristic ground-truth relevance score based on metadata overlap."""
    skill_overlap = resume_meta["skills"] & jd_meta["required_skills"]
    required_skills = jd_meta["required_skills"]
    skill_score = len(skill_overlap) / max(len(required_skills), 1)

    years_score = 1.0 if resume_meta["years_experience"] >= jd_meta["required_years"] else \
        resume_meta["years_experience"] / max(jd_meta["required_years"], 1)

    edu_levels = {"Bachelor": 3, "Master": 4, "PhD": 5}
    resume_edu = edu_levels.get(resume_meta["degree"], 0)
    required_edu = edu_levels.get(jd_meta["required_degree"], 0)
    edu_score = 1.0 if resume_edu >= required_edu else resume_edu / max(required_edu, 1)

    score = (
        skill_score * 0.50
        + years_score * 0.30
        + edu_score * 0.20
    )
    score = min(max(score, 0.0), 1.0)
    # Scale to 0-100
    score = score * 100
    # Add small noise
    score += random.uniform(-3, 3)
    return min(max(score, 0.0), 100.0)


def generate_dataset(n_samples: int = 2000) -> Tuple[List[str], List[str], List[float]]:
    """Generate synthetic (resume, jd, score) triplets."""
    resumes = []
    jds = []
    scores = []
    for _ in range(n_samples):
        resume_text, resume_meta = generate_synthetic_resume()
        jd_text, jd_meta = generate_synthetic_jd()
        score = _compute_true_score(resume_meta, jd_meta)
        resumes.append(resume_text)
        jds.append(jd_text)
        scores.append(score)
    return resumes, jds, scores


def build_feature_matrix(
    resumes: List[str],
    jds: List[str],
    scorer: SimilarityScorer,
) -> np.ndarray:
    """Build feature matrix for all resume-JD pairs."""
    # Compute semantic similarities in batches for efficiency
    # Group by unique JD to reduce redundant encoding
    unique_jd_indices = {}
    for idx, jd in enumerate(jds):
        unique_jd_indices.setdefault(jd, []).append(idx)

    semantic_sims = np.zeros(len(resumes))
    for jd, indices in unique_jd_indices.items():
        batch_resumes = [resumes[i] for i in indices]
        sims = scorer.compute_batch_semantic_similarity(jd, batch_resumes)
        for i, sim in zip(indices, sims):
            semantic_sims[i] = sim

    features = []
    for resume_text, jd_text, sim in zip(resumes, jds, semantic_sims):
        # For each pair, fit a fresh TF-IDF on the pair to capture pair-specific terms
        # This mimics the inference behavior when ranking a single JD against multiple resumes
        vectorizer = scorer.tfidf_vectorizer.__class__(
            stop_words="english", ngram_range=(1, 2), max_features=5000
        )
        feature_vector = extract_all_features(resume_text, jd_text, sim)
        features.append(feature_vector)
    return np.array(features)


def train_model(
    n_samples: int = 2000,
    test_size: float = 0.2,
    output_dir: str = "models",
) -> dict:
    """Train the candidate ranking model and save artifacts."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating synthetic dataset...")
    resumes, jds, scores = generate_dataset(n_samples)

    print("Initializing similarity scorer...")
    scorer = SimilarityScorer()

    print("Building feature matrix...")
    X = build_feature_matrix(resumes, jds, scorer)
    y = np.array(scores)

    feature_names = get_feature_names()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Training XGBoost regressor...")
    model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(
        X_train_scaled,
        y_train,
        eval_set=[(X_test_scaled, y_test)],
        verbose=False,
    )

    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    print(f"Validation MSE: {mse:.2f}")
    print(f"Validation MAE: {mae:.2f}")

    # Save artifacts
    model_path = output_dir / "ranking_model.json"
    scaler_path = output_dir / "scaler.joblib"
    feature_names_path = output_dir / "feature_names.json"

    model.save_model(str(model_path))
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    with open(feature_names_path, "w") as f:
        json.dump(feature_names, f, indent=2)

    print(f"Artifacts saved to {output_dir}")

    # Feature importance
    importance = model.feature_importances_
    importance_df = pd.DataFrame(
        {"feature": feature_names, "importance": importance}
    ).sort_values("importance", ascending=False)
    print("\nTop 10 important features:")
    print(importance_df.head(10).to_string(index=False))

    return {
        "model_path": str(model_path),
        "scaler_path": str(scaler_path),

        "feature_names_path": str(feature_names_path),
        "mse": mse,
        "mae": mae,
        "feature_importance": importance_df.to_dict("records"),
    }


if __name__ == "__main__":
    train_model()
