import json
import pickle
from pathlib import Path
from typing import Dict, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field


class TemperatureScalerFit:
    def __init__(self):
        self.temperature = 1.0

    def fit(self, probs, true_labels):
        return self

    def predict_proba(self, probs):
        probs = np.asarray(probs, dtype=float)
        eps = 1e-12
        logits = np.log((probs + eps) / (1 - probs + eps))
        scaled = 1 / (1 + np.exp(-(logits / self.temperature)))
        return np.clip(scaled, 0.0, 1.0)


class ArtifactUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "__main__" and name == "TemperatureScalerFit":
            return TemperatureScalerFit
        return super().find_class(module, name)


class FinalDetectionOutput(BaseModel):
    review_text: str = Field(description="Original review text")
    model_used: Literal["logistic_regression", "random_forest"] = Field(
        description="Classifier used"
    )
    calibrated: bool = Field(description="Whether probability calibration was applied")
    ai_probability: float = Field(ge=0.0, le=1.0)
    ai_likeness_score: int = Field(ge=0, le=100)
    uncertainty_band: Literal[
        "likely human-written",
        "uncertain",
        "likely AI-generated",
    ]
    predicted_label: Literal["Human", "AI"]
    top_features: Dict[str, float] = Field(
        description="Top stylometric feature values used for explanation"
    )
    explanation: str = Field(
        description="Short explanation grounded only in the provided model output and top features"
    )


class FinalReviewDetector:
    def __init__(self, artifact_path: str | None = None):
        root_dir = Path(__file__).resolve().parents[1]
        default_artifact = root_dir / "models" / "diverse_week5_artifacts.pkl"

        self.artifact_path = Path(artifact_path) if artifact_path else default_artifact
        if not self.artifact_path.exists():
            raise FileNotFoundError(
                f"Artifact file not found: {self.artifact_path}. "
                f"Expected to load models/diverse_week5_artifacts.pkl"
            )

        with open(self.artifact_path, "rb") as f:
            artifacts = ArtifactUnpickler(f).load()

        self.feature_columns = artifacts["feature_columns"]
        self.lr_classifier = artifacts["lr_classifier"]
        self.rf_classifier = artifacts["rf_classifier"]
        self.lr_temp_scaler = artifacts["lr_temp_scaler"]
        self.rf_temp_scaler = artifacts["rf_temp_scaler"]
        self.selected_model_name = artifacts["selected_model_name"]
        self.comparison_df = artifacts.get("comparison_df", None)

    @staticmethod
    def get_uncertainty_band(score: int) -> str:
        if score <= 39:
            return "likely human-written"
        if score <= 69:
            return "uncertain"
        return "likely AI-generated"

    @staticmethod
    def get_prediction_label(ai_prob: float) -> str:
        return "AI" if ai_prob >= 0.5 else "Human"

    @staticmethod
    def apply_temperature_scaling(prob: float, scaler) -> float:
        scaled = scaler.predict_proba(np.array([prob], dtype=float))
        return float(np.clip(scaled[0], 0.0, 1.0))

    @staticmethod
    def _rank_features_for_model(model_name: str, clf, feature_columns: list[str]) -> list[str]:
        if model_name == "logistic_regression":
            if not hasattr(clf, "coef_"):
                return feature_columns
            importances = np.abs(clf.coef_[0])
        else:
            if not hasattr(clf, "feature_importances_"):
                return feature_columns
            importances = np.asarray(clf.feature_importances_)

        ranked_idx = np.argsort(importances)[::-1]
        return [feature_columns[i] for i in ranked_idx]

    def get_top_features_for_explanation(
        self,
        features: Dict[str, float],
        model_name: str,
        top_k: int = 3,
    ) -> Dict[str, float]:
        clf = self.lr_classifier if model_name == "logistic_regression" else self.rf_classifier
        ranked = self._rank_features_for_model(model_name, clf, self.feature_columns)

        selected = {}
        for feat in ranked[:top_k]:
            selected[feat] = float(features.get(feat, 0.0))
        return selected

    def run_classifier_on_features(
        self,
        features: Dict[str, float],
        model_name: str | None = None,
        use_calibration: bool = True,
    ) -> Dict:
        if model_name is None:
            model_name = self.selected_model_name

        feature_vector = pd.DataFrame([features])[self.feature_columns]

        if model_name == "logistic_regression":
            clf = self.lr_classifier
            scaler = self.lr_temp_scaler
        elif model_name == "random_forest":
            clf = self.rf_classifier
            scaler = self.rf_temp_scaler
        else:
            raise ValueError("model_name must be 'logistic_regression' or 'random_forest'")

        raw_prob = float(clf.predict_proba(feature_vector)[0, 1])

        if use_calibration:
            ai_prob = self.apply_temperature_scaling(raw_prob, scaler)
        else:
            ai_prob = raw_prob

        score = int(round(100 * ai_prob))
        band = self.get_uncertainty_band(score)
        label = self.get_prediction_label(ai_prob)
        top_features = self.get_top_features_for_explanation(
            features=features,
            model_name=model_name,
            top_k=3,
        )

        return {
            "model_used": model_name,
            "calibrated": use_calibration,
            "ai_probability": ai_prob,
            "ai_likeness_score": score,
            "uncertainty_band": band,
            "predicted_label": label,
            "top_features": top_features,
        }

    @staticmethod
    def generate_grounded_explanation(classifier_output: Dict) -> str:
        ai_prob = classifier_output["ai_probability"]
        score = classifier_output["ai_likeness_score"]
        band = classifier_output["uncertainty_band"]
        label = classifier_output["predicted_label"]
        top_features = classifier_output["top_features"]

        feature_str = ", ".join(
            [f"{k}={round(v, 4)}" for k, v in top_features.items()]
        )

        if label == "AI":
            return (
                f"The detector assigns an AI probability of {ai_prob:.4f} "
                f"with an AI-likeness score of {score}, which falls in the "
                f"'{band}' band. The prediction is driven by the top stylometric "
                f"features {feature_str}."
            )

        return (
            f"The detector assigns an AI probability of {ai_prob:.4f} "
            f"with an AI-likeness score of {score}, which falls in the "
            f"'{band}' band. The prediction is supported by the top stylometric "
            f"features {feature_str}."
        )

    def detect_review(
        self,
        review_text: str,
        extracted_features: Dict[str, float],
        model_name: str | None = None,
        use_calibration: bool = True,
    ) -> FinalDetectionOutput:
        filtered_features = {
            col: float(extracted_features.get(col, 0.0))
            for col in self.feature_columns
        }

        classifier_output = self.run_classifier_on_features(
            features=filtered_features,
            model_name=model_name,
            use_calibration=use_calibration,
        )

        explanation = self.generate_grounded_explanation(classifier_output)

        return FinalDetectionOutput(
            review_text=review_text,
            model_used=classifier_output["model_used"],
            calibrated=classifier_output["calibrated"],
            ai_probability=round(classifier_output["ai_probability"], 4),
            ai_likeness_score=classifier_output["ai_likeness_score"],
            uncertainty_band=classifier_output["uncertainty_band"],
            predicted_label=classifier_output["predicted_label"],
            top_features=classifier_output["top_features"],
            explanation=explanation,
        )

    def detect_review_dict(
        self,
        review_text: str,
        extracted_features: Dict[str, float],
        model_name: str | None = None,
        use_calibration: bool = True,
    ) -> Dict:
        result = self.detect_review(
            review_text=review_text,
            extracted_features=extracted_features,
            model_name=model_name,
            use_calibration=use_calibration,
        )
        return json.loads(result.model_dump_json())


detector = FinalReviewDetector()
