"""Feature extraction module for resume-job description pairs."""

import re
from typing import Dict, List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from utils import (
    SECTION_KEYWORDS,
    extract_education_level,
    extract_experience_years,
    extract_required_experience,
    extract_skills,
    has_section,
)


def skill_match_features(resume_skills: set, jd_skills: set) -> Dict[str, float]:
    """Compute skill overlap features."""
    if not jd_skills:
        return {
            "matched_skills": 0,
            "missing_skills": 0,
            "skill_match_ratio": 0.0,
            "extra_skills": len(resume_skills),
        }
    matched = len(resume_skills & jd_skills)
    missing = len(jd_skills - resume_skills)
    ratio = matched / len(jd_skills)
    return {
        "matched_skills": matched,
        "missing_skills": missing,
        "skill_match_ratio": ratio,
        "extra_skills": len(resume_skills - jd_skills),
    }


def experience_match_feature(resume_years: int, required_years: int) -> float:
    """Score experience fit on a 0-1 scale."""
    if required_years <= 0:
        return 1.0 if resume_years > 0 else 0.0
    if resume_years >= required_years:
        return 1.0
    return resume_years / required_years


def education_match_feature(resume_level: int, required_level: int) -> float:
    """Score education fit on a 0-1 scale."""
    if required_level <= 0:
        return 1.0 if resume_level > 0 else 0.0
    if resume_level >= required_level:
        return 1.0
    return resume_level / required_level


def extract_all_features(
    resume_text: str,
    jd_text: str,
    semantic_similarity: float,
) -> np.ndarray:
    """Extract a numeric feature vector for a resume-JD pair."""
    resume_skills = extract_skills(resume_text)
    jd_skills = extract_skills(jd_text)
    skill_features = skill_match_features(resume_skills, jd_skills)

    resume_exp = extract_experience_years(resume_text)
    required_exp = extract_required_experience(jd_text)
    exp_fit = experience_match_feature(resume_exp, required_exp)

    resume_edu = extract_education_level(resume_text)
    required_edu = extract_education_level(jd_text)
    edu_fit = education_match_feature(resume_edu, required_edu)

    section_features = {
        f"has_{section}": has_section(resume_text, keywords)
        for section, keywords in SECTION_KEYWORDS.items()
    }

    # TF-IDF cosine similarity (fit fresh on the pair to mimic training behavior)
    tfidf_vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=5000)
    tfidf_matrix = tfidf_vectorizer.fit_transform([jd_text, resume_text])
    tfidf_sim = float((tfidf_matrix[0] @ tfidf_matrix[1].T).toarray()[0, 0])

    feature_values = [
        semantic_similarity,
        tfidf_sim,
        skill_features["matched_skills"],
        skill_features["missing_skills"],
        skill_features["skill_match_ratio"],
        skill_features["extra_skills"],
        resume_exp,
        required_exp,
        exp_fit,
        resume_edu,
        required_edu,
        edu_fit,
        *section_features.values(),
    ]
    return np.array(feature_values, dtype=np.float32)


def get_feature_names() -> List[str]:
    """Return human-readable feature names in the same order as extract_all_features."""
    base = [
        "semantic_similarity",
        "tfidf_similarity",
        "matched_skills",
        "missing_skills",
        "skill_match_ratio",
        "extra_skills",
        "resume_experience_years",
        "required_experience_years",
        "experience_fit",
        "resume_education_level",
        "required_education_level",
        "education_fit",
    ]
    section_names = [f"has_{section}" for section in SECTION_KEYWORDS]
    return base + section_names
