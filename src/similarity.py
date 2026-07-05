"""NLP similarity scoring: TF-IDF and Sentence-BERT semantic similarity."""

import os
import pickle
from pathlib import Path
from typing import List, Union

import numpy as np
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer


class SimilarityScorer:
    """Compute lexical and semantic similarity between job descriptions and resumes."""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        cache_dir: Union[str, Path] = "models",
    ):
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.embedding_model = self._load_embedding_model()
        self.tfidf_vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_features=5000,
        )

    def _load_embedding_model(self) -> SentenceTransformer:
        """Load or download the sentence transformer model."""
        try:
            model = SentenceTransformer(self.model_name, cache_folder=str(self.cache_dir))
            return model
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load sentence transformer '{self.model_name}': {exc}"
            ) from exc

    def compute_semantic_similarity(self, text1: str, text2: str) -> float:
        """Compute cosine similarity between two texts using Sentence-BERT."""
        embeddings = self.embedding_model.encode([text1, text2], convert_to_tensor=True)
        similarity = util.cos_sim(embeddings[0], embeddings[1])
        return float(similarity.item())

    def compute_tfidf_similarity(self, text1: str, text2: str, fit: bool = True) -> float:
        """Compute cosine similarity between two texts using TF-IDF."""
        if fit:
            matrix = self.tfidf_vectorizer.fit_transform([text1, text2])
        else:
            matrix = self.tfidf_vectorizer.transform([text1, text2])
        return float((matrix[0] @ matrix[1].T).toarray()[0, 0])

    def compute_batch_semantic_similarity(
        self, job_description: str, resumes: List[str]
    ) -> np.ndarray:
        """Compute semantic similarity between one JD and multiple resumes efficiently."""
        if not resumes:
            return np.array([])
        jd_embedding = self.embedding_model.encode(job_description, convert_to_tensor=True)
        resume_embeddings = self.embedding_model.encode(resumes, convert_to_tensor=True)
        similarities = util.cos_sim(jd_embedding, resume_embeddings)
        return similarities.cpu().numpy().flatten()

    def save(self, path: Union[str, Path]) -> None:
        """Save the TF-IDF vectorizer to disk."""
        path = Path(path)
        with open(path, "wb") as f:
            pickle.dump(self.tfidf_vectorizer, f)

    def load(self, path: Union[str, Path]) -> None:
        """Load a saved TF-IDF vectorizer from disk."""
        path = Path(path)
        with open(path, "rb") as f:
            self.tfidf_vectorizer = pickle.load(f)
