"""Unit tests for the resume screening pipeline."""

import sys
from pathlib import Path

import numpy as np
import pytest
# Ensure src is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from extract_features import extract_all_features, get_feature_names, skill_match_features
from parse_resume import extract_text_from_txt
from utils import clean_text, extract_experience_years, extract_skills


SAMPLE_RESUME = """
John Doe
Software Engineer

Skills:
Python, Django, React, SQL, Docker, Machine Learning

Experience:
Software Engineer at TechCorp - 5 years of experience.
Built web applications with Python and React.

Education:
Bachelor in Computer Science

Projects:
Developed a portfolio website.
"""

SAMPLE_JD = """
Software Engineer

We are looking for a Software Engineer with 3+ years of experience.
Required skills: Python, Django, React, Docker.
Bachelor's degree in Computer Science required.
"""


def test_clean_text():
    raw = "  Hello   \n\n  World  \t  "
    cleaned = clean_text(raw)
    assert cleaned == "Hello\nWorld"


def test_extract_skills():
    text = "Experienced in Python, Django, React, and Docker."
    skills = extract_skills(text)
    assert {"python", "django", "react", "docker"}.issubset(skills)


def test_extract_experience_years():
    assert extract_experience_years("5 years of experience") == 5
    assert extract_experience_years("No experience mentioned") == 0


def test_skill_match_features():
    resume_skills = {"python", "django", "react"}
    jd_skills = {"python", "django", "docker"}
    features = skill_match_features(resume_skills, jd_skills)
    assert features["matched_skills"] == 2
    assert features["missing_skills"] == 1
    assert abs(features["skill_match_ratio"] - 2 / 3) < 1e-6


def test_extract_all_features():
    features = extract_all_features(SAMPLE_RESUME, SAMPLE_JD, semantic_similarity=0.75)
    assert isinstance(features, np.ndarray)
    assert len(features) == len(get_feature_names())
    assert features[0] == 0.75  # semantic_similarity


def test_rank_candidates_ordering(tmp_path):
    """High-match resume should rank above low-match resume."""
    from rank_candidates import ResumeRanker

    models_dir = Path(__file__).resolve().parent.parent / "models"
    if not (models_dir / "ranking_model.json").exists():
        pytest.skip("Trained model artifacts not found; run src/train_model.py first.")

    ranker = ResumeRanker(models_dir=models_dir)

    high_match = "Python Django React Docker 5 years experience Bachelor Computer Science"
    low_match = "Graphic Designer Photoshop Creativity 2 years experience Bachelor Fine Arts"

    resumes = {"high_match.txt": high_match, "low_match.txt": low_match}
    results = ranker.rank(SAMPLE_JD, resumes)

    assert results.iloc[0]["filename"] == "high_match.txt"
    assert results.iloc[1]["filename"] == "low_match.txt"
    assert results.iloc[0]["score"] > results.iloc[1]["score"]


def test_parse_txt(tmp_path):
    resume_file = tmp_path / "resume.txt"
    resume_file.write_text(SAMPLE_RESUME, encoding="utf-8")
    text = extract_text_from_txt(resume_file)
    assert "Python" in text
    assert "Software Engineer" in text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
