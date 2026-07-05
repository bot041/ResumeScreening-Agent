"""End-to-end candidate ranking pipeline."""

import json
import pickle
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import pandas as pd
import xgboost as xgb
from extract_features import extract_all_features, get_feature_names
from parse_resume import parse_resumes_from_folder
from similarity import SimilarityScorer


class ResumeRanker:
    """Load trained artifacts and rank candidates against a job description."""

    def __init__(self, models_dir: Union[str, Path] = "models"):
        self.models_dir = Path(models_dir)
        self.model = self._load_model()
        self.scaler = self._load_scaler()
        self.feature_names = self._load_feature_names()
        self.similarity_scorer = SimilarityScorer()

    def _load_model(self) -> xgb.XGBRegressor:
        model = xgb.XGBRegressor()
        model.load_model(str(self.models_dir / "ranking_model.json"))
        return model

    def _load_scaler(self):
        with open(self.models_dir / "scaler.joblib", "rb") as f:
            return pickle.load(f)

    def _load_feature_names(self) -> List[str]:
        with open(self.models_dir / "feature_names.json", "r") as f:
            return json.load(f)

    def rank(
        self,
        job_description: str,
        resumes: Dict[str, str],
    ) -> pd.DataFrame:
        """Rank parsed resumes against a job description."""
        if not resumes:
            return pd.DataFrame(columns=["filename", "score", "rank"] + self.feature_names)

        filenames = list(resumes.keys())
        texts = list(resumes.values())

        # Batch semantic similarity
        semantic_sims = self.similarity_scorer.compute_batch_semantic_similarity(
            job_description, texts
        )

        feature_matrix = []
        for resume_text, sim in zip(texts, semantic_sims):
            features = extract_all_features(resume_text, job_description, sim)
            feature_matrix.append(features)

        X = np.array(feature_matrix)
        X_scaled = self.scaler.transform(X)
        scores = self.model.predict(X_scaled)
        scores = np.clip(scores, 0, 100)

        results = pd.DataFrame({
            "filename": filenames,
            "score": scores,
        })

        # Add feature breakdown
        feature_df = pd.DataFrame(X, columns=self.feature_names)
        results = pd.concat([results, feature_df], axis=1)

        results = results.sort_values("score", ascending=False).reset_index(drop=True)
        results["rank"] = results.index + 1
        return results

    def rank_from_folder(
        self,
        job_description: str,
        folder_path: Union[str, Path],
    ) -> pd.DataFrame:
        """Parse resumes from a folder and rank them."""
        resumes = parse_resumes_from_folder(folder_path)
        return self.rank(job_description, resumes)


def main():
    """CLI entry point for ranking candidates."""
    import argparse

    parser = argparse.ArgumentParser(description="Rank resumes against a job description.")
    parser.add_argument("--jd", required=True, help="Path to job description text file")
    parser.add_argument("--resumes", required=True, help="Path to folder containing resumes")
    parser.add_argument("--models", default="models", help="Path to trained model artifacts")
    parser.add_argument("--output", default="ranked_candidates.csv", help="Output CSV path")
    args = parser.parse_args()

    with open(args.jd, "r", encoding="utf-8") as f:
        job_description = f.read()

    ranker = ResumeRanker(models_dir=args.models)
    results = ranker.rank_from_folder(job_description, args.resumes)

    results.to_csv(args.output, index=False)
    print(f"Ranked {len(results)} candidates. Results saved to {args.output}")
    print(results[["rank", "filename", "score"]].to_string(index=False))


if __name__ == "__main__":
    main()
