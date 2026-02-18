# =============================================================================
# AI-Powered Learning Management System — Plagiarism Detection Module
# =============================================================================
# Copyright (c) 2025 Suyash Vishwas Jadhav. All Rights Reserved.
# Author: Suyash Vishwas Jadhav
# Contributor: Nidhi Pawar (Plagiarism scoring engine)
# Unauthorized use or distribution is strictly prohibited.
# =============================================================================
"""
Plagiarism Detection Module.
Detects plagiarism in learner submissions using rule-based analysis.
"""

import re
import nltk
from typing import Dict, List

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    try:
        nltk.download('punkt', quiet=True)
    except:
        pass

def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences using NLTK"""
    try:
        sentences = nltk.sent_tokenize(text)
        return [s.strip() for s in sentences if s.strip()]
    except:
        # Fallback to simple splitting
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]

def check_plagiarism(text: str) -> Dict:
    """
    Comprehensive plagiarism detection
    Returns: {
        'plagiarism_detected': bool,
        'score': float (0-100),
        'details': str,
        'matches': List[Dict]
    }
    """
    if not text or len(text.strip()) < 20:
        return {
            "plagiarism_detected": False,
            "score": 0.0,
            "details": "Text too short for analysis",
            "matches": []
        }
    
    sentences = split_into_sentences(text)
    
    if len(sentences) < 2:
        return {
            "plagiarism_detected": False,
            "score": 0.0,
            "details": "Insufficient content for analysis",
            "matches": []
        }
    
    plagiarism_score = 0
    matches = []
    text_lower = text.lower()
    
    # Formal/Professional Writing (50 points)
    formal_score = 0
    definition_patterns = [
        (r'\bis the ability\b', 30, "Formal definition pattern"),
        (r'\bis defined as\b', 25, "Definition phrase"),
        (r'\brefers to\b', 20, "Reference phrase"),
        (r'\bis a branch of\b', 25, "Academic classification"),
        (r'\bis the process\b', 25, "Process definition"),
        (r'\bcan be defined\b', 22, "Definition structure")
    ]
    
    for pattern, points, description in definition_patterns:
        if re.search(pattern, text_lower):
            formal_score += points
            matches.append({
                "text": pattern.replace(r'\b', ''),
                "status": description,
                "confidence": "High"
            })
            break
    
    encyclopedic_phrases = ['such as', 'for example', 'including', 'consists of', 'characterized by']
    encyc_count = sum(1 for phrase in encyclopedic_phrases if phrase in text_lower)
    if encyc_count >= 1:
        formal_score += min(encyc_count * 4, 25)
        if encyc_count >= 2:
            matches.append({
                "text": "Encyclopedic phrases",
                "status": f"{encyc_count} formal phrases detected",
                "confidence": "Medium"
            })
    
    formal_score = min(formal_score, 50)
    plagiarism_score += formal_score
    
    # Publication markers (25 points)
    pub_keywords = {
        'chapter': 8,
        'references': 10,
        'bibliography': 10,
        'published': 8,
        'isbn': 10
    }
    pub_score = 0
    for keyword, points in pub_keywords.items():
        if keyword in text_lower:
            pub_score += points
            matches.append({
                "text": keyword,
                "status": "Publication marker",
                "confidence": "High"
            })
    pub_score = min(pub_score, 25)
    plagiarism_score += pub_score
    
    # Citations (20 points)
    citation_score = 0
    if re.search(r'\[\d+\]', text):
        citation_score += 10
        matches.append({
            "text": "Numbered citations",
            "status": "Academic citation format",
            "confidence": "High"
        })
    if re.search(r'\(\d{4}\)', text):
        citation_score += 8
        matches.append({
            "text": "Year citations",
            "status": "Academic reference",
            "confidence": "Medium"
        })
    citation_score = min(citation_score, 20)
    plagiarism_score += citation_score
    
    # Academic style (25 points)
    style_score = 0
    long_sentences = [s for s in sentences if len(s.split()) > 15]
    if len(long_sentences) > len(sentences) * 0.3:
        style_score += 12
        matches.append({
            "text": "Complex sentence structure",
            "status": "Academic writing style",
            "confidence": "Medium"
        })
    
    formal_transitions = ['however', 'moreover', 'furthermore', 'nevertheless', 'consequently']
    formal_count = sum(1 for word in formal_transitions if f' {word} ' in text_lower)
    if formal_count >= 2:
        style_score += 10
        matches.append({
            "text": "Formal transitions",
            "status": f"{formal_count} academic transition words",
            "confidence": "Medium"
        })
    
    style_score = min(style_score, 25)
    plagiarism_score += style_score
    
    # Vocabulary complexity (20 points)
    words = text_lower.split()
    if len(words) > 30:
        complex_words = [w for w in words if len(w) > 10]
        complex_ratio = len(complex_words) / len(words)
        if complex_ratio > 0.12:
            plagiarism_score += 12
            matches.append({
                "text": "Complex vocabulary",
                "status": "Advanced academic language",
                "confidence": "Low"
            })
    
    # Calculate final score with multiplier
    base_percentage = (plagiarism_score / 140) * 130
    multiplier = 1.4 if formal_score >= 20 else 1.0
    final_score = min(base_percentage * multiplier, 100)
    
    if plagiarism_score >= 30:
        final_score = max(final_score, 60)
    
    return {
        "plagiarism_detected": final_score >= 25,
        "score": round(final_score, 2),
        "details": f"Analyzed {len(sentences)} sentences | {len(matches)} indicators found",
        "matches": matches[:5]  # Limit to top 5 matches
    }

def detect_ai_content_rule_based(text: str) -> Dict:
    """
    Rule-based AI detection
    Returns: {
        'ai_detected': bool,
        'confidence': float (0-100),
        'indicators': Dict
    }
    """
    if not text or len(text.strip()) < 20:
        return {
            "ai_detected": False,
            "confidence": 0.0,
            "indicators": {}
        }
    
    sentences = split_into_sentences(text)
    if len(sentences) == 0:
        return {
            "ai_detected": False,
            "confidence": 0.0,
            "indicators": {}
        }
    
    ai_score = 0
    text_lower = text.lower()
    
    # ChatGPT phrases
    chatgpt_phrases = [
        'it is important to note', 'keep in mind', 'in conclusion',
        'delve into', 'comprehensive guide', 'best practices',
        'leverage', 'it\'s worth noting', 'it is worth noting'
    ]
    ai_phrase_count = sum(1 for phrase in chatgpt_phrases if phrase in text_lower)
    if ai_phrase_count > 0:
        ai_score += min(ai_phrase_count * 12, 35)
    
    # Transition words
    transitions = [
        'however', 'moreover', 'furthermore', 'additionally',
        'consequently', 'therefore', 'nevertheless'
    ]
    transition_count = sum(1 for word in transitions if f' {word} ' in text_lower)
    if transition_count >= 2:
        ai_score += min(transition_count * 6, 25)
    
    # Formal language
    formal_words = [
        'utilize', 'implement', 'facilitate', 'demonstrate',
        'comprehensive', 'leverage', 'substantial'
    ]
    formal_count = sum(1 for word in formal_words if word in text_lower)
    if formal_count >= 2:
        ai_score += min(formal_count * 5, 20)
    
    # Sentence uniformity
    if len(sentences) > 3:
        lengths = [len(s.split()) for s in sentences]
        avg_length = sum(lengths) / len(lengths)
        variance = sum((l - avg_length) ** 2 for l in lengths) / len(lengths)
        if variance < 35:
            ai_score += 20
    
    ai_score = min(ai_score, 100)
    
    return {
        "ai_detected": ai_score >= 35,
        "confidence": round(ai_score, 2),
        "indicators": {
            "ai_phrases": ai_phrase_count,
            "transitions": transition_count,
            "formal_language": formal_count
        }
    }

def analyze_submission_text(text: str) -> Dict:
    """
    Complete analysis of submission text
    Returns combined plagiarism and AI detection results
    """
    if not text or len(text.strip()) < 10:
        return {
            "success": False,
            "error": "Text too short for analysis"
        }
    
    plagiarism_results = check_plagiarism(text)
    ai_results = detect_ai_content_rule_based(text)
    
    return {
        "success": True,
        "plagiarism": plagiarism_results,
        "ai_detection": ai_results,
        "text_length": len(text),
        "sentence_count": len(split_into_sentences(text))
    }

