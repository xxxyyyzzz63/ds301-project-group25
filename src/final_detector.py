from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

try:
    from .stylometry_features import extract_stylometry_features
except ImportError:
    from stylometry_features import extract_stylometry_features

__all__ = [
    "FinalReviewDetector",
    "FinalDetectionOutput",
    "TemperatureScalerFit",
    "ArtifactUnpickler",
    "detector",
]


class TemperatureScalerFit:
    def __init__(self) -> None:
        self.temperature: float = 1.0

    def fit(self, probs, true_labels) -> "TemperatureScalerFit":
        return self

    def predict_proba(self, probs) -> np.ndarray:
        probs = np.asarray(probs, dtype=float)
        eps = 1e-12
        logits = np.log((probs + eps) / (1 - probs + eps))
        scaled = 1.0 / (1.0 + np.exp(-(logits / self.temperature)))
        return np.clip(scaled, 0.0, 1.0)


class ArtifactUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        if module == "__main__" and name == "TemperatureScalerFit":
            return TemperatureScalerFit
        return super().find_class(module, name)


class FinalDetectionOutput(BaseModel):
    review_text: str
    model_used: str
    calibrated: bool
    ai_probability: float = Field(ge=0.0, le=1.0)
    ai_likeness_score: int = Field(ge=0, le=100)
    uncertainty_band: str
    predicted_label: str
    top_features: Dict[str, float]
    explanation: str


def _find_artifact_path() -> Path:
    here = Path(__file__).resolve()
    target = "diverse_week5_artifacts.pkl"

    for parent in [here.parent, *here.parents]:
        candidate = parent / "models" / target
        if candidate.exists():
            return candidate

    return here.parents[1] / "models" / target


class FinalReviewDetector:
    HUMAN_BAND_MAX: int = 39
    UNCERTAIN_BAND_MAX: int = 69

    def __init__(self, artifact_path: Optional[str | Path] = None) -> None:
        if artifact_path is None:
            self.artifact_path = _find_artifact_path()
        else:
            self.artifact_path = Path(artifact_path)

        if not self.artifact_path.exists():
            raise FileNotFoundError(
                f"Artifact file not found: {self.artifact_path}. "
                "Expected to load models/diverse_week5_artifacts.pkl."
            )

        with open(self.artifact_path, "rb") as f:
            artifacts = ArtifactUnpickler(f).load()

        self.feature_columns: List[str] = list(artifacts["feature_columns"])
        self.lr_classifier = artifacts["lr_classifier"]
        self.rf_classifier = artifacts["rf_classifier"]
        self.lr_temp_scaler = artifacts["lr_temp_scaler"]
        self.rf_temp_scaler = artifacts["rf_temp_scaler"]
        self.selected_model_name: str = artifacts["selected_model_name"]
        self.comparison_df = artifacts.get("comparison_df", None)

    @staticmethod
    def get_uncertainty_band(score: int) -> str:
        if score <= FinalReviewDetector.HUMAN_BAND_MAX:
            return "likely human-written"
        if score <= FinalReviewDetector.UNCERTAIN_BAND_MAX:
            return "uncertain"
        return "likely AI-generated"

    @staticmethod
    def get_prediction_label(ai_prob: float) -> str:
        return "AI" if ai_prob >= 0.5 else "Human"

    @staticmethod
    def apply_temperature_scaling(prob: float, scaler: TemperatureScalerFit) -> float:
        scaled = scaler.predict_proba(np.array([prob], dtype=float))
        return float(np.clip(scaled[0], 0.0, 1.0))

    @staticmethod
    def _rank_features_for_model(
        model_name: str, clf: Any, feature_columns: List[str]
    ) -> List[str]:
        if model_name == "logistic_regression":
            if not hasattr(clf, "coef_"):
                return list(feature_columns)
            importances = np.abs(clf.coef_[0])
        else:
            if not hasattr(clf, "feature_importances_"):
                return list(feature_columns)
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
        return {feat: float(features.get(feat, 0.0)) for feat in ranked[:top_k]}

    def run_classifier_on_features(
        self,
        features: Dict[str, float],
        model_name: Optional[str] = None,
        use_calibration: bool = True,
    ) -> Dict[str, Any]:
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
            raise ValueError(
                "model_name must be 'logistic_regression' or 'random_forest', "
                f"got {model_name!r}"
            )

        raw_prob = float(clf.predict_proba(feature_vector)[0, 1])
        ai_prob = (
            self.apply_temperature_scaling(raw_prob, scaler) if use_calibration else raw_prob
        )

        score = int(round(100 * ai_prob))
        return {
            "model_used": model_name,
            "calibrated": use_calibration,
            "ai_probability": ai_prob,
            "ai_likeness_score": score,
            "uncertainty_band": self.get_uncertainty_band(score),
            "predicted_label": self.get_prediction_label(ai_prob),
            "top_features": self.get_top_features_for_explanation(
                features=features, model_name=model_name, top_k=3
            ),
        }

    @staticmethod
    def generate_grounded_explanation(classifier_output: Dict[str, Any]) -> str:
        ai_prob = classifier_output["ai_probability"]
        score = classifier_output["ai_likeness_score"]
        band = classifier_output["uncertainty_band"]
        label = classifier_output["predicted_label"]
        top_features = classifier_output["top_features"]

        feature_str = ", ".join(f"{k}={round(v, 4)}" for k, v in top_features.items())
        verb = "driven by" if label == "AI" else "supported by"
        return (
            f"The detector assigns an AI probability of {ai_prob:.4f} "
            f"with an AI-likeness score of {score}, which falls in the "
            f"'{band}' band. The prediction is {verb} the top stylometric "
            f"features {feature_str}."
        )

    def detect_review(
        self,
        review_text: str,
        extracted_features: Dict[str, float],
        model_name: Optional[str] = None,
        use_calibration: bool = True,
    ) -> FinalDetectionOutput:
        filtered = {col: float(extracted_features.get(col, 0.0)) for col in self.feature_columns}

        classifier_output = self.run_classifier_on_features(
            features=filtered, model_name=model_name, use_calibration=use_calibration
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
        model_name: Optional[str] = None,
        use_calibration: bool = True,
    ) -> Dict[str, Any]:
        result = self.detect_review(
            review_text=review_text,
            extracted_features=extracted_features,
            model_name=model_name,
            use_calibration=use_calibration,
        )
        return json.loads(result.model_dump_json())

    def detect(
        self,
        review_text: str,
        model_name: Optional[str] = None,
        use_calibration: bool = True,
    ) -> Dict[str, Any]:
        features = extract_stylometry_features(review_text)
        return self.detect_review_dict(
            review_text=review_text,
            extracted_features=features,
            model_name=model_name,
            use_calibration=use_calibration,
        )


_DETECTOR_INSTANCE: Optional[FinalReviewDetector] = None


def _get_detector() -> FinalReviewDetector:
    global _DETECTOR_INSTANCE
    if _DETECTOR_INSTANCE is None:
        _DETECTOR_INSTANCE = FinalReviewDetector()
    return _DETECTOR_INSTANCE


def __getattr__(name: str) -> Any:
    if name == "detector":
        return _get_detector()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _format_result(result: Dict[str, Any]) -> str:
    lines = [
        "=" * 80,
        f"REVIEW: {result['review_text']}",
        "-" * 80,
        f"Model used:        {result['model_used']}",
        f"Calibrated:        {result['calibrated']}",
        f"AI probability:    {result['ai_probability']}",
        f"AI-likeness score: {result['ai_likeness_score']}",
        f"Uncertainty band:  {result['uncertainty_band']}",
        f"Predicted label:   {result['predicted_label']}",
        f"Top features:      {result['top_features']}",
        f"Explanation:       {result['explanation']}",
        "=" * 80,
    ]
    return "\n".join(lines)


def _main(argv: List[str]) -> int:
    if len(argv) < 2:
        print(
            "Usage: python src/final_detector.py \"<review text>\"",
            file=sys.stderr,
        )
        return 1

    review_text = " ".join(argv[1:])
    d = _get_detector()
    result = d.detect(review_text)
    print(_format_result(result))
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv))
