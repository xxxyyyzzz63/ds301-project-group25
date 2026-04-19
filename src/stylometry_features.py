"""
Stylometry feature extractor for AI-generated text detection.
"""

from typing import Dict
import numpy as np
import re


def extract_stylometry_features(text: str) -> Dict[str, float]:
    text = str(text).strip()

    words = re.findall(r"\b\w+\b", text.lower())
    word_count = len(words)

    if word_count == 0:
        return {
            "avg_word_length": 0.0,
            "type_token_ratio": 0.0,
            "hapax_ratio": 0.0,
            "stopword_ratio": 0.0,
            "avg_words_per_sentence": 0.0,
            "avg_sentence_length": 0.0,
            "sentence_length_variance": 0.0,
            "paragraph_count": 0.0,
            "period_ratio": 0.0,
            "capital_letter_ratio": 0.0,
            "comma_ratio": 0.0,
            "exclamation_ratio": 0.0,
            "question_ratio": 0.0,
            "long_word_ratio": 0.0,
        }

    avg_word_length = np.mean([len(w) for w in words])

    unique_words = set(words)
    type_token_ratio = len(unique_words) / word_count

    word_freq = {}
    for w in words:
        word_freq[w] = word_freq.get(w, 0) + 1
    hapax_ratio = sum(1 for w in word_freq if word_freq[w] == 1) / word_count

    stopwords = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "as", "is", "was", "are", "were", "be",
        "been", "being", "have", "has", "had", "do", "does", "did", "will",
        "would", "could", "should", "may", "might", "must", "can", "this",
        "that", "these", "those", "i", "you", "he", "she", "it", "we", "they"
    }
    stopword_ratio = sum(1 for w in words if w in stopwords) / word_count

    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    sentence_lengths = [len(re.findall(r"\b\w+\b", s)) for s in sentences]

    avg_words_per_sentence = np.mean(sentence_lengths) if sentence_lengths else 0.0
    avg_sentence_length = np.mean([len(s) for s in sentences]) if sentences else 0.0
    sentence_length_variance = np.var(sentence_lengths) if len(sentence_lengths) > 1 else 0.0

    char_count = len(text) if len(text) > 0 else 1
    period_ratio = text.count(".") / char_count
    capital_letter_ratio = sum(1 for c in text if c.isupper()) / char_count
    comma_ratio = text.count(",") / char_count
    exclamation_ratio = text.count("!") / char_count
    question_ratio = text.count("?") / char_count

    paragraphs = [p for p in text.split("\n\n") if p.strip()]
    paragraph_count = len(paragraphs) if paragraphs else 1

    long_word_ratio = sum(1 for w in words if len(w) >= 7) / word_count

    return {
        "avg_word_length": round(float(avg_word_length), 3),
        "type_token_ratio": round(float(type_token_ratio), 3),
        "hapax_ratio": round(float(hapax_ratio), 3),
        "stopword_ratio": round(float(stopword_ratio), 3),
        "avg_words_per_sentence": round(float(avg_words_per_sentence), 3),
        "avg_sentence_length": round(float(avg_sentence_length), 3),
        "sentence_length_variance": round(float(sentence_length_variance), 3),
        "paragraph_count": float(paragraph_count),
        "period_ratio": round(float(period_ratio), 4),
        "capital_letter_ratio": round(float(capital_letter_ratio), 4),
        "comma_ratio": round(float(comma_ratio), 4),
        "exclamation_ratio": round(float(exclamation_ratio), 4),
        "question_ratio": round(float(question_ratio), 4),
        "long_word_ratio": round(float(long_word_ratio), 4),
    }
