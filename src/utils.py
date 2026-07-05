"""Utility helpers for text cleaning, skill taxonomy, and metadata extraction."""

import re
from typing import List, Set, Tuple


def clean_text(text: str) -> str:
    """Normalize whitespace and strip special characters from extracted text."""
    if not text:
        return ""
    text = text.replace("\r", "\n")
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" ?\n ?", "\n", text)
    text = re.sub(r"\n+", "\n", text)
    return text.strip()


TECH_SKILLS = {
    "python", "java", "javascript", "js", "typescript", "ts", "c++", "c#", "go", "golang",
    "rust", "ruby", "php", "swift", "kotlin", "scala", "r", "matlab", "sql", "bash",
    "shell", "html", "css", "react", "angular", "vue", "svelte", "node.js", "nodejs",
    "express", "django", "flask", "spring", "asp.net", "laravel", "rails",
    "tensorflow", "pytorch", "keras", "scikit-learn", "sklearn", "xgboost", "lightgbm",
    "pandas", "numpy", "scipy", "matplotlib", "seaborn", "plotly", "opencv", "nlp",
    "machine learning", "deep learning", "data science", "statistics", "aws", "azure",
    "gcp", "google cloud", "docker", "kubernetes", "jenkins", "git", "github", "gitlab",
    "terraform", "ansible", "prometheus", "grafana", "linux", "unix", "windows",
    "mongodb", "mysql", "postgresql", "sqlite", "redis", "elasticsearch", "cassandra",
    "kafka", "rabbitmq", "airflow", "spark", "hadoop", "tableau", "powerbi", "excel",
    "rest api", "graphql", "soap", "microservices", "ci/cd", "devops", "agile", "scrum",
}

SOFT_SKILLS = {
    "communication", "leadership", "teamwork", "problem solving", "critical thinking",
    "time management", "adaptability", "creativity", "collaboration", "project management",
    "mentoring", "analytical", "detail-oriented", "self-motivated", "interpersonal",
}

ALL_SKILLS = TECH_SKILLS | SOFT_SKILLS

EDUCATION_LEVELS = {
    "high school": 1,
    "associate": 2,
    "bachelor": 3,
    "master": 4,
    "phd": 5,
    "doctorate": 5,
}


def extract_skills(text: str) -> Set[str]:
    """Extract known skills from text using a curated taxonomy."""
    text_lower = text.lower()
    found = set()
    for skill in ALL_SKILLS:
        if re.search(r"\b" + re.escape(skill) + r"\b", text_lower):
            found.add(skill)
    return found


def extract_experience_years(text: str) -> int:
    """Estimate total years of professional experience from resume text."""
    text_lower = text.lower()
    patterns = [
        r"(\d+)\+?\s*years?\s*(?:of\s*)?experience",
        r"experience\s*:?\s*(\d+)\+?\s*years?",
        r"worked\s*(?:for\s*)?(\d+)\+?\s*years?",
    ]
    years = []
    for pattern in patterns:
        matches = re.findall(pattern, text_lower)
        years.extend([int(m) for m in matches])
    if years:
        return min(max(years), 40)
    return 0


def extract_required_experience(jd_text: str) -> int:
    """Extract required years of experience from a job description."""
    text_lower = jd_text.lower()
    patterns = [
        r"(\d+)\+?\s*years?\s*(?:of\s*)?experience",
        r"minimum\s*(?:of\s*)?(\d+)\+?\s*years?",
        r"at\s*least\s*(\d+)\+?\s*years?",
    ]
    years = []
    for pattern in patterns:
        matches = re.findall(pattern, text_lower)
        years.extend([int(m) for m in matches])
    if years:
        return min(years)
    return 0


def extract_education_level(text: str) -> int:
    """Return highest education level detected in text."""
    text_lower = text.lower()
    level = 0
    for keyword, value in EDUCATION_LEVELS.items():
        if keyword in text_lower:
            level = max(level, value)
    return level


def has_section(text: str, section_keywords: List[str]) -> int:
    """Check if a section is present in the resume."""
    text_lower = text.lower()
    return int(any(kw in text_lower for kw in section_keywords))


SECTION_KEYWORDS = {
    "experience": ["experience", "employment", "work history", "professional experience"],
    "education": ["education", "academic", "qualification", "degree"],
    "skills": ["skills", "technical skills", "core competencies"],
    "projects": ["projects", "personal projects", "portfolio"],
    "certifications": ["certifications", "certificates", "accreditations"],
}
