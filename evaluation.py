# =============================================================================
# AI-Powered Learning Management System — Evaluation Module
# =============================================================================
# Copyright (c) 2025 Suyash Vishwas Jadhav. All Rights Reserved.
# Author: Suyash Vishwas Jadhav
# Unauthorized use or distribution is strictly prohibited.
# =============================================================================
"""
AI-powered evaluation module with OCR support.
Supports Gemini, DeepSeek, and Groq models.
"""

import os
import re
from datetime import datetime
from difflib import SequenceMatcher
from google.cloud import vision
from google.oauth2 import service_account

# OCR Setup (Google Cloud Vision)
try:
    import json
    # Try to read from environment variable first (for GCP deployment)
    credentials_json = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS_JSON')
    
    if credentials_json:
        # Parse JSON from environment variable
        credentials_info = json.loads(credentials_json)
        credentials = service_account.Credentials.from_service_account_info(credentials_info)
        vision_client = vision.ImageAnnotatorClient(credentials=credentials)
        print("✅ OCR initialized from environment variable")
    else:
        # Fallback to file (for local development)
        credentials = service_account.Credentials.from_service_account_file("ocr.json")
        vision_client = vision.ImageAnnotatorClient(credentials=credentials)
        print("✅ OCR initialized from ocr.json file")
except Exception as e:
    print(f"⚠️ OCR initialization error: {e}")
    vision_client = None

def extract_handwritten_text(image_path):
    """Extract text from handwritten images using Google Cloud Vision OCR
    
    Uses document_text_detection which is optimized for handwritten text,
    documents, and dense text. This method provides better accuracy for
    handwritten content compared to text_detection.
    """
    if not vision_client:
        return "OCR service not available"
    
    try:
        with open(image_path, "rb") as f:
            content = f.read()
        
        # Use document_text_detection for handwritten text (better than text_detection)
        image = vision.Image(content=content)
        
        # Configure image context for better handwritten text recognition
        image_context = vision.ImageContext(
            language_hints=['en']  # Hint that text is in English
        )
        
        # Use document_text_detection - optimized for handwritten and dense text
        response = vision_client.document_text_detection(
            image=image,
            image_context=image_context
        )
        
        if response.error.message:
            raise Exception(f"Google Vision API Error: {response.error.message}")
        
        # Extract full text annotation (best for handwritten)
        if response.full_text_annotation and response.full_text_annotation.text:
            extracted_text = response.full_text_annotation.text.strip()
            
            # Also try to get text from individual blocks if full text is empty
            if not extracted_text and response.text_annotations:
                # Fallback to first text annotation
                extracted_text = response.text_annotations[0].description.strip() if response.text_annotations else ""
            
            return extracted_text if extracted_text else ""
        else:
            # Fallback: try text_annotations if full_text_annotation is empty
            if response.text_annotations:
                return response.text_annotations[0].description.strip() if response.text_annotations else ""
            return ""
            
    except Exception as e:
        print(f"Google Vision OCR extraction error: {e}")
        import traceback
        traceback.print_exc()
        return f"Error extracting text: {str(e)}"

def check_name_mismatch(extracted_text, learner_name):
    """Check if first three words match learner's name"""
    if not extracted_text or not learner_name:
        return False
    
    words = extracted_text.strip().split()[:3]
    name_words = learner_name.lower().split()
    
    # Check if any name word appears in first three words
    for name_word in name_words:
        if name_word and any(name_word in word.lower() for word in words):
            return False
    
    return True  # Flag if name not found

def check_plagiarism(current_text, all_submissions):
    """Check for exact matches with other submissions"""
    if not current_text:
        return []
    
    suspicious = []
    current_clean = ' '.join(current_text.lower().split())
    
    for submission in all_submissions:
        other_text = submission.get('text', '')
        other_clean = ' '.join(other_text.lower().split())
        
        if other_clean and current_clean and current_clean == other_clean:
            suspicious.append({
                'learner_id': submission.get('learner_id'),
                'learner_name': submission.get('learner_name'),
                'similarity': 100
            })
    
    return suspicious


def calculate_text_similarity(text1, text2):
    """Calculate similarity percentage between two texts"""
    from difflib import SequenceMatcher

    if not text1 or not text2:
        return 0

    # Normalize texts
    text1_clean = ' '.join(text1.lower().split())
    text2_clean = ' '.join(text2.lower().split())

    # Calculate similarity
    similarity = SequenceMatcher(None, text1_clean, text2_clean).ratio()
    return round(similarity * 100, 2)


def detect_plagiarism_realtime(current_answer_id, question_id, extracted_text, db_session):
    """
    Real-time plagiarism detection comparing with all other submissions
    for the same question
    """
    from sqlalchemy import and_

    # Import models lazily to avoid circular import at module import time
    try:
        from app import AssignmentAnswer, AssignmentQuestion
    except Exception:
        # If import fails, try to access by name via globals (best-effort)
        AssignmentAnswer = None
        AssignmentQuestion = None

    # Get all other answers for the same question
    query = db_session.query(AssignmentAnswer).join(
        AssignmentQuestion
    ).filter(
        and_(
            AssignmentQuestion.id == question_id,
            AssignmentAnswer.id != current_answer_id,
            AssignmentAnswer.extracted_text.isnot(None),
            AssignmentAnswer.extracted_text != ''
        )
    )

    other_answers = query.all()

    plagiarism_matches = []

    for other_answer in other_answers:
        similarity = calculate_text_similarity(extracted_text, other_answer.extracted_text)

        # Flag if similarity >= 70%
        if similarity >= 70:
            plagiarism_matches.append({
                'matched_answer_id': other_answer.id,
                'matched_learner': other_answer.submission.learner,
                'similarity_score': similarity
            })

    return plagiarism_matches

def calculate_size_score(word_count, expected_min=50, expected_max=200):
    """Calculate size score based on word count (30% weight)"""
    if word_count < expected_min:
        return (word_count / expected_min) * 30
    elif word_count > expected_max:
        penalty = min((word_count - expected_max) / expected_max, 0.5)
        return max(30 * (1 - penalty), 15)
    else:
        return 30

def evaluate_with_gemini(question, answer_text, marks_per_question):
    """Evaluate using Gemini Flash 2.5"""
    try:
        import google.generativeai as genai
        
        api_key = os.environ.get('GEMINI_API_KEY')
        if not api_key or not api_key.strip():
            return {
                'relevance_score': 0,
                'grammar_score': 0,
                'uniqueness_score': 0,
                'feedback': 'Gemini API key not configured. Please add GEMINI_API_KEY to your .env file.',
                'covered_points': [],
                'missing_points': []
            }
        
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        prompt = f"""
Evaluate this answer strictly based on the question asked.

Question: {question}

Student's Answer: {answer_text}

Provide evaluation in this exact JSON format:
{{
    "relevance_score": <0-50, measures if answer addresses the question>,
    "grammar_score": <0-10, grammar and language quality>,
    "uniqueness_score": <0-10, creativity and originality>,
    "feedback": "<brief constructive feedback>",
    "covered_points": ["point1", "point2"],
    "missing_points": ["missing1", "missing2"]
}}

Important:
- If answer is completely irrelevant, set relevance_score to 0
- Be strict but fair
- Focus on content quality
"""
        
        response = model.generate_content(prompt)
        result_text = response.text.strip()
        
        # Parse JSON response
        import json
        result_text = result_text.replace('```json', '').replace('```', '').strip()
        result = json.loads(result_text)
        
        return result
        
    except Exception as e:
        print(f"Gemini evaluation error: {e}")
        return {
            'relevance_score': 0,
            'grammar_score': 0,
            'uniqueness_score': 0,
            'feedback': f'Evaluation error: {str(e)}',
            'covered_points': [],
            'missing_points': []
        }

def evaluate_with_deepseek(question, answer_text, marks_per_question):
    """Evaluate using DeepSeek V3.2"""
    try:
        from openai import OpenAI
        
        api_key = os.environ.get('DEEPSEEK_API_KEY')
        if not api_key:
            return {
                'relevance_score': 0,
                'grammar_score': 0,
                'uniqueness_score': 0,
                'feedback': 'DeepSeek API key not configured',
                'covered_points': [],
                'missing_points': []
            }
        
        client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com"
        )
        
        prompt = f"""
Evaluate this answer strictly based on the question asked.

Question: {question}

Student's Answer: {answer_text}

Provide evaluation in this exact JSON format:
{{
    "relevance_score": <0-50, measures if answer addresses the question>,
    "grammar_score": <0-10, grammar and language quality>,
    "uniqueness_score": <0-10, creativity and originality>,
    "feedback": "<brief constructive feedback>",
    "covered_points": ["point1", "point2"],
    "missing_points": ["missing1", "missing2"]
}}
"""
        
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        
        result_text = response.choices[0].message.content.strip()
        
        # Parse JSON response
        import json
        result_text = result_text.replace('```json', '').replace('```', '').strip()
        result = json.loads(result_text)
        
        return result
        
    except Exception as e:
        print(f"DeepSeek evaluation error: {e}")
        return {
            'relevance_score': 0,
            'grammar_score': 0,
            'uniqueness_score': 0,
            'feedback': f'Evaluation error: {str(e)}',
            'covered_points': [],
            'missing_points': []
        }

def evaluate_with_groq(question, answer_text, marks_per_question):
    """Evaluate using Groq LLM"""
    try:
        from groq import Groq
        
        api_key = os.environ.get('GROQ_API_KEY')
        if not api_key:
            return {
                'relevance_score': 0,
                'grammar_score': 0,
                'uniqueness_score': 0,
                'feedback': 'Groq API key not configured',
                'covered_points': [],
                'missing_points': []
            }
        
        client = Groq(api_key=api_key)
        
        prompt = f"""
Evaluate this answer strictly based on the question asked.

Question: {question}

Student's Answer: {answer_text}

Provide evaluation in this exact JSON format:
{{
    "relevance_score": <0-50, measures if answer addresses the question>,
    "grammar_score": <0-10, grammar and language quality>,
    "uniqueness_score": <0-10, creativity and originality>,
    "feedback": "<brief constructive feedback>",
    "covered_points": ["point1", "point2"],
    "missing_points": ["missing1", "missing2"]
}}
"""
        
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        
        result_text = response.choices[0].message.content.strip()
        
        # Parse JSON response
        import json
        result_text = result_text.replace('```json', '').replace('```', '').strip()
        result = json.loads(result_text)
        
        return result
        
    except Exception as e:
        print(f"Groq evaluation error: {e}")
        return {
            'relevance_score': 0,
            'grammar_score': 0,
            'uniqueness_score': 0,
            'feedback': f'Evaluation error: {str(e)}',
            'covered_points': [],
            'missing_points': []
        }

def evaluate_answer(question, answer_text, marks_per_question, model='gemini'):
    """
    Main evaluation function
    Returns weighted scores:
    - Relevance: 50% (if 0, total = 0)
    - Grammar: 10%
    - Size: 30%
    - Uniqueness: 10%
    """
    if not answer_text or len(answer_text.strip()) < 10:
        return {
            'relevance_score': 0,
            'grammar_score': 0,
            'size_score': 0,
            'uniqueness_score': 0,
            'total_score': 0,
            'feedback': 'Answer too short or empty',
            'covered_points': [],
            'missing_points': ['Complete answer required']
        }
    
    # Choose evaluation model
    if model == 'deepseek':
        ai_result = evaluate_with_deepseek(question, answer_text, marks_per_question)
    elif model == 'groq':
        ai_result = evaluate_with_groq(question, answer_text, marks_per_question)
    else:  # default: gemini
        ai_result = evaluate_with_gemini(question, answer_text, marks_per_question)
    
    # Calculate size score
    word_count = len(answer_text.split())
    size_score = calculate_size_score(word_count)
    
    # Get AI scores
    relevance = ai_result.get('relevance_score', 0)
    grammar = ai_result.get('grammar_score', 0)
    uniqueness = ai_result.get('uniqueness_score', 0)
    
    # If relevance is 0, total is 0
    if relevance == 0:
        total = 0
    else:
        total = relevance + grammar + size_score + uniqueness
    
    # Scale to marks_per_question
    scaled_total = (total / 100) * marks_per_question
    
    return {
        'relevance_score': relevance,
        'grammar_score': grammar,
        'size_score': size_score,
        'uniqueness_score': uniqueness,
        'total_score': round(scaled_total, 2),
        'word_count': word_count,
        'feedback': ai_result.get('feedback', ''),
        'covered_points': ai_result.get('covered_points', []),
        'missing_points': ai_result.get('missing_points', [])
    }


def generate_question_mcq(question_text, model='gemini'):
    """Generate MCQ about the question topic"""
    try:
        prompt = f"""
Create a multiple choice question that tests understanding of this question's topic:

Question: {question_text}

Generate ONE MCQ in this EXACT JSON format:
{{
    "question": "What is the main concept being asked about?",
    "option_a": "First option",
    "option_b": "Second option",
    "option_c": "Third option",
    "option_d": "Fourth option",
    "correct": "A"
}}

Rules:
- Question should test conceptual understanding
- Make it challenging but fair
- Only one correct answer
- Return ONLY valid JSON, no extra text
"""
        
        if model == 'gemini':
            import google.generativeai as genai
            api_key = os.environ.get('GEMINI_API_KEY')
            if not api_key or not api_key.strip():
                print("Warning: GEMINI_API_KEY not configured for question MCQ generation")
                return None
            
            genai.configure(api_key=api_key)
            model_obj = genai.GenerativeModel('gemini-2.5-flash')
            response = model_obj.generate_content(prompt)
            result_text = response.text.strip()
            
        elif model == 'deepseek':
            from openai import OpenAI
            api_key = os.environ.get('DEEPSEEK_API_KEY')
            if not api_key or not api_key.strip():
                print("Warning: DEEPSEEK_API_KEY not configured for answer MCQ generation")
                return None
            
            client = OpenAI(
                api_key=api_key,
                base_url="https://api.deepseek.com"
            )
            response = client.chat.completions.create(
                model="deepseek-chat",  # Free model
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
            result_text = response.choices[0].message.content.strip()
            
        elif model == 'groq':
            from openai import OpenAI
            api_key = os.environ.get('GROQ_API_KEY')
            if not api_key or not api_key.strip():
                print("Warning: GROQ_API_KEY not configured for answer MCQ generation")
                return None
            
            client = OpenAI(
                api_key=api_key,
                base_url="https://api.groq.com/openai/v1"
            )
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",  # Free model
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
            result_text = response.choices[0].message.content.strip()
        else:
            return None
        
        import json
        result_text = result_text.replace('```json', '').replace('```', '').strip()
        return json.loads(result_text)
        
    except Exception as e:
        print(f"MCQ generation error ({model}): {e}")
        return None


def generate_answer_mcq(question_text, answer_text, model='gemini'):
    """Generate MCQ about what learner wrote"""
    try:
        prompt = f"""
Create a multiple choice question that tests if the learner understands what they wrote:

Original Question: {question_text}

Learner's Answer: {answer_text}

Generate ONE MCQ in this EXACT JSON format:
{{
    "question": "According to your answer, what did you explain about...?",
    "option_a": "First option",
    "option_b": "Second option", 
    "option_c": "Third option",
    "option_d": "Fourth option",
    "correct": "A"
}}

Rules:
- Question must be about THEIR answer content
- Test if they understand what they wrote
- Make options relevant to their answer
- Only one correct answer
- Return ONLY valid JSON, no extra text
"""
        
        if model == 'gemini':
            import google.generativeai as genai
            api_key = os.environ.get('GEMINI_API_KEY')
            if not api_key or not api_key.strip():
                print("Warning: GEMINI_API_KEY not configured for answer MCQ generation")
                return None
            
            genai.configure(api_key=api_key)
            model_obj = genai.GenerativeModel('gemini-2.5-flash')
            response = model_obj.generate_content(prompt)
            result_text = response.text.strip()
            
        elif model == 'deepseek':
            from openai import OpenAI
            api_key = os.environ.get('DEEPSEEK_API_KEY')
            if not api_key or not api_key.strip():
                print("Warning: DEEPSEEK_API_KEY not configured for answer MCQ generation")
                return None
            
            client = OpenAI(
                api_key=api_key,
                base_url="https://api.deepseek.com"
            )
            response = client.chat.completions.create(
                model="deepseek-chat",  # Free model
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
            result_text = response.choices[0].message.content.strip()
            
        elif model == 'groq':
            from openai import OpenAI
            api_key = os.environ.get('GROQ_API_KEY')
            if not api_key or not api_key.strip():
                print("Warning: GROQ_API_KEY not configured for answer MCQ generation")
                return None
            
            client = OpenAI(
                api_key=api_key,
                base_url="https://api.groq.com/openai/v1"
            )
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",  # Free model
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
            result_text = response.choices[0].message.content.strip()
        else:
            return None
        
        import json
        result_text = result_text.replace('```json', '').replace('```', '').strip()
        return json.loads(result_text)
        
    except Exception as e:
        print(f"Answer MCQ generation error ({model}): {e}")
        return None