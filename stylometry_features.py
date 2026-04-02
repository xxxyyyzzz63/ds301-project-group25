"""
Stylometry Feature Extractor for AI-generated Text Detection
Extracts 15 stylometric features from hotel reviews
"""

from langchain.tools import Tool
from typing import Dict
import numpy as np
import re


def extract_stylometry_features(text: str) -> Dict[str, float]:
    """
    Extract 15 stylometric features from text.
    
    Features:
    - Word-level: avg_word_length, long_word_ratio, unique_word_ratio
    - Sentence-level: avg_sentence_length, sentence_length_variance
    - Punctuation: exclamation_ratio, question_ratio, comma_ratio, period_ratio
    - Lexical diversity: type_token_ratio, hapax_ratio
    - Syntactic: stopword_ratio, avg_words_per_sentence
    - Structural: paragraph_count, capital_letter_ratio
    """
    
    # Clean text
    text = text.strip()
    
    # Word-level features
    words = re.findall(r'\b\w+\b', text.lower())
    word_count = len(words)
    
    if word_count == 0:
        return {f"feature_{i}": 0.0 for i in range(15)}
    
    # 1. Average word length
    avg_word_length = np.mean([len(w) for w in words]) if words else 0
    
    # 2. Long word ratio (words > 6 characters)
    long_words = [w for w in words if len(w) > 6]
    long_word_ratio = len(long_words) / word_count
    
    # 3. Unique word ratio
    unique_words = set(words)
    unique_word_ratio = len(unique_words) / word_count
    
    # Sentence-level features
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    sentence_count = len(sentences)
    
    # 4. Average sentence length (in words)
    sentence_lengths = [len(re.findall(r'\b\w+\b', s)) for s in sentences]
    avg_sentence_length = np.mean(sentence_lengths) if sentence_lengths else 0
    
    # 5. Sentence length variance
    sentence_length_variance = np.var(sentence_lengths) if len(sentence_lengths) > 1 else 0
    
    # Punctuation features
    char_count = len(text)
    
    # 6. Exclamation ratio
    exclamation_count = text.count('!')
    exclamation_ratio = exclamation_count / char_count if char_count > 0 else 0
    
    # 7. Question ratio
    question_count = text.count('?')
    question_ratio = question_count / char_count if char_count > 0 else 0
    
    # 8. Comma ratio
    comma_count = text.count(',')
    comma_ratio = comma_count / char_count if char_count > 0 else 0
    
    # 9. Period ratio
    period_count = text.count('.')
    period_ratio = period_count / char_count if char_count > 0 else 0
    
    # Lexical diversity
    # 10. Type-token ratio (unique words / total words)
    type_token_ratio = len(unique_words) / word_count
    
    # 11. Hapax legomena ratio (words appearing once / total words)
    word_freq = {}
    for word in words:
        word_freq[word] = word_freq.get(word, 0) + 1
    hapax_words = [w for w, count in word_freq.items() if count == 1]
    hapax_ratio = len(hapax_words) / word_count
    
    # Syntactic features
    # 12. Stopword ratio (common English stopwords)
    stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                 'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be',
                 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this',
                 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they'}
    stopword_count = sum(1 for w in words if w in stopwords)
    stopword_ratio = stopword_count / word_count
    
    # 13. Average words per sentence
    avg_words_per_sentence = word_count / sentence_count if sentence_count > 0 else 0
    
    # Structural features
    # 14. Paragraph count (approximated by double newlines)
    paragraphs = text.split('\n\n')
    paragraph_count = len([p for p in paragraphs if p.strip()])
    
    # 15. Capital letter ratio
    capital_count = sum(1 for c in text if c.isupper())
    capital_letter_ratio = capital_count / char_count if char_count > 0 else 0
    
    return {
        'avg_word_length': round(avg_word_length, 3),
        'long_word_ratio': round(long_word_ratio, 3),
        'unique_word_ratio': round(unique_word_ratio, 3),
        'avg_sentence_length': round(avg_sentence_length, 3),
        'sentence_length_variance': round(sentence_length_variance, 3),
        'exclamation_ratio': round(exclamation_ratio, 4),
        'question_ratio': round(question_ratio, 4),
        'comma_ratio': round(comma_ratio, 4),
        'period_ratio': round(period_ratio, 4),
        'type_token_ratio': round(type_token_ratio, 3),
        'hapax_ratio': round(hapax_ratio, 3),
        'stopword_ratio': round(stopword_ratio, 3),
        'avg_words_per_sentence': round(avg_words_per_sentence, 3),
        'paragraph_count': paragraph_count,
        'capital_letter_ratio': round(capital_letter_ratio, 4)
    }


# Create LangChain tool
stylometry_tool = Tool(
    name="extract_stylometry_features",
    func=extract_stylometry_features,
    description="Extracts 15 stylometric features from text for AI detection"
)


if __name__ == "__main__":
    # Test the feature extractor
    sample_text = """
    This hotel was absolutely amazing! The staff were incredibly friendly and helpful.
    The room was spacious and clean. I would definitely recommend this place to anyone
    visiting the area. The location is perfect for exploring the city.
    """
    
    features = extract_stylometry_features(sample_text)
    print("Extracted features:")
    for feature, value in features.items():
        print(f"  {feature}: {value}")
