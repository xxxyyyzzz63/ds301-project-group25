from src.final_detector import FinalReviewDetector
from src.stylometry_features import extract_stylometry_features


def run_one(detector, review_text: str):
    features = extract_stylometry_features(review_text)
    result = detector.detect_review(
        review_text=review_text,
        extracted_features=features,
        use_calibration=True,
    )

    print("=" * 80)
    print("REVIEW:")
    print(review_text)
    print("-" * 80)
    print("MODEL USED:", result.model_used)
    print("CALIBRATED:", result.calibrated)
    print("AI PROBABILITY:", result.ai_probability)
    print("AI-LIKENESS SCORE:", result.ai_likeness_score)
    print("UNCERTAINTY BAND:", result.uncertainty_band)
    print("PREDICTED LABEL:", result.predicted_label)
    print("TOP FEATURES:", result.top_features)
    print("EXPLANATION:", result.explanation)
    print()


def main():
    detector = FinalReviewDetector()

    test_reviews = [
        "The room was clean and the location was convenient, but the breakfast was disappointing and the walls were thin.",
        "This hotel exceeded all expectations. Every amenity was thoughtfully arranged, the staff delivered impeccable service, and the stay felt seamless from check-in to checkout.",
        "We stayed for two nights. The bed was comfortable and the bathroom was clean, but the parking fee felt too high for what we got.",
        "Absolutely loved our stay. The pool area was spotless, the breakfast buffet had lots of options, and the staff were incredibly friendly throughout the weekend.",
    ]

    for review in test_reviews:
        run_one(detector, review)


if __name__ == "__main__":
    main()
