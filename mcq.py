# =============================================================================
# AI-Powered Learning Management System — MCQ Generation Module
# =============================================================================
# Copyright (c) 2025 Suyash Vishwas Jadhav. All Rights Reserved.
# Author: Suyash Vishwas Jadhav
# Unauthorized use or distribution is strictly prohibited.
# =============================================================================
"""
MCQ Generation Module using Gemini AI and Groq API.
Generates MCQs from topics or documents.
"""

import os
import json
import PyPDF2
import csv
import io
from werkzeug.utils import secure_filename

# Load environment variables first
try:
    from dotenv import load_dotenv
    load_dotenv()
except:
    pass

# Configure Gemini API using new google.genai client
gemini_client = None
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', '')
GEMINI_AVAILABLE = False

def initialize_gemini_client():
    """Initialize Gemini client - can be called multiple times"""
    global gemini_client, GEMINI_API_KEY, GEMINI_AVAILABLE
    
    # Reload env if needed
    try:
        from dotenv import load_dotenv
        load_dotenv()
        GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', '')
    except:
        pass
    
    if not GEMINI_API_KEY or not GEMINI_API_KEY.strip():
        return False, "GEMINI_API_KEY not set in environment variables"
    
    try:
        from google import genai
        from google.genai import types
        GEMINI_AVAILABLE = True
        
        try:
            gemini_client = genai.Client(api_key=GEMINI_API_KEY)
            print(f"✅ Gemini client initialized successfully")
            return True, "Success"
        except Exception as e:
            error_msg = f"Error initializing Gemini client: {str(e)}"
            print(f"❌ {error_msg}")
            import traceback
            traceback.print_exc()
            return False, error_msg
    except ImportError:
        error_msg = "google.genai not available. Please install: pip install google-genai"
        print(f"⚠️ {error_msg}")
        return False, error_msg
    except Exception as e:
        error_msg = f"Unexpected error initializing Gemini: {str(e)}"
        print(f"❌ {error_msg}")
        import traceback
        traceback.print_exc()
        return False, error_msg

# Try to initialize on module load
initialize_gemini_client()

# Configure Groq API
groq_client = None
GROQ_API_KEY = os.environ.get('GROQ_API_KEY', '')
GROQ_AVAILABLE = False

try:
    from groq import Groq
    GROQ_AVAILABLE = True
    if GROQ_API_KEY:
        try:
            groq_client = Groq(api_key=GROQ_API_KEY)
            print(f"✅ Groq client initialized successfully")
        except Exception as e:
            print(f"❌ Error initializing Groq client: {str(e)}")
    else:
        print("⚠️ GROQ_API_KEY not set in environment variables")
except ImportError:
    print("⚠️ groq package not available. Please install: pip install groq")

# Model names
GEMINI_2_5_FLASH = "gemini-2.5-flash"  # Gemini 2.5 Flash
GROQ_MODEL = "llama-3.1-70b-versatile"  # Fast and capable Groq model

ALLOWED_EXTENSIONS = {'txt', 'pdf', 'csv', 'docx'}

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_pdf(file):
    """Extract text from PDF file"""
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text.strip()
    except Exception as e:
        raise Exception(f"Error reading PDF: {str(e)}")

def extract_text_from_csv(file):
    """Extract text from CSV file"""
    try:
        text = ""
        content = file.read().decode('utf-8')
        csv_reader = csv.reader(io.StringIO(content))
        for row in csv_reader:
            text += " ".join(row) + "\n"
        return text.strip()
    except Exception as e:
        raise Exception(f"Error reading CSV: {str(e)}")

def extract_text_from_txt(file):
    """Extract text from TXT file"""
    try:
        return file.read().decode('utf-8').strip()
    except Exception as e:
        raise Exception(f"Error reading TXT: {str(e)}")

def extract_text_from_docx(file):
    """Extract text from DOCX file"""
    try:
        from docx import Document
        doc = Document(file)
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        return text.strip()
    except ImportError:
        raise Exception("python-docx library not installed. Install with: pip install python-docx")
    except Exception as e:
        raise Exception(f"Error reading DOCX: {str(e)}")

def extract_text_from_file(file, filename):
    """Extract text from any supported file format"""
    filename_lower = filename.lower()
    
    if filename_lower.endswith('.pdf'):
        return extract_text_from_pdf(file)
    elif filename_lower.endswith('.csv'):
        return extract_text_from_csv(file)
    elif filename_lower.endswith('.txt'):
        return extract_text_from_txt(file)
    elif filename_lower.endswith('.docx'):
        return extract_text_from_docx(file)
    else:
        raise Exception(f"Unsupported file type: {filename}")

def generate_mcqs_from_topic(topic, num_questions, subcategory="", difficulty="medium", option_difficulty="identifiable", provider="gemini"):
    """Generate MCQs from a topic using Gemini or Groq"""
    if provider == "gemini":
        # Try to initialize/reinitialize client
        if not gemini_client:
            success, error_msg = initialize_gemini_client()
            if not success:
                raise Exception(error_msg)
        
        if not gemini_client:
            raise Exception("Gemini client initialization failed. Please check your GEMINI_API_KEY.")
        
        gemini_model = GEMINI_2_5_FLASH
    elif provider == "groq":
        if not GROQ_AVAILABLE:
            raise Exception("groq package not installed. Please install: pip install groq")
        if not GROQ_API_KEY or not GROQ_API_KEY.strip():
            raise Exception("GROQ_API_KEY not set in environment variables. Please set it in your .env file or environment.")
        if not groq_client:
            raise Exception("Groq client not initialized")
    
    subcategory_text = f"Focus on subcategory: {subcategory}. " if subcategory else ""
    
    prompt = f"""Create exactly {num_questions} multiple choice questions (MCQs) on the topic: {topic}

Specifications:
- Difficulty level: {difficulty}
- {subcategory_text}
- Options should be: {option_difficulty} (make options {'clearly distinguishable' if option_difficulty == 'identifiable' else 'tricky and similar to confuse students'})

Format each question EXACTLY as follows (this format is critical):

Question 1: [question text here]
A) [option A text]
B) [option B text]
C) [option C text]
D) [option D text]
Correct Answer: [A or B or C or D]
Explanation: [brief explanation of why this answer is correct]

Question 2: [question text here]
A) [option A text]
B) [option B text]
C) [option C text]
D) [option D text]
Correct Answer: [A or B or C or D]
Explanation: [brief explanation]

Continue this format for all {num_questions} questions.

Ensure:
- Questions are {difficulty} difficulty level
- Options are {option_difficulty}
- Each question has exactly 4 options (A, B, C, D)
- Only one correct answer per question
- Questions are relevant to the topic: {topic}
"""
    
    try:
        if provider == "gemini":
            # Use Gemini API
            response = gemini_client.models.generate_content(
                model=gemini_model,
                contents=prompt
            )
            if not response or not hasattr(response, 'text') or not response.text:
                raise Exception("No response from Gemini API")
            mcqs_text = response.text
        else:  # groq
            # Use Groq API
            chat_completion = groq_client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                model=GROQ_MODEL,
                temperature=0.7
            )
            mcqs_text = chat_completion.choices[0].message.content
            if not mcqs_text:
                raise Exception("No response from Groq API")
        
        return parse_mcqs(mcqs_text)
    except Exception as e:
        error_msg = str(e)
        if "pattern" in error_msg.lower() or "invalid" in error_msg.lower() or "model" in error_msg.lower():
            raise Exception(f"API error: Please check your API key and model availability. Error: {error_msg}")
        raise Exception(f"Error generating MCQs: {error_msg}")

def generate_mcqs_from_document(content, num_questions=None, max_questions=False, provider="gemini"):
    """Generate MCQs from document content using Gemini or Groq"""
    if provider == "gemini":
        # Try to initialize/reinitialize client
        if not gemini_client:
            success, error_msg = initialize_gemini_client()
            if not success:
                raise Exception(error_msg)
        
        if not gemini_client:
            raise Exception("Gemini client initialization failed. Please check your GEMINI_API_KEY.")
        
        gemini_model = GEMINI_2_5_FLASH
    elif provider == "groq":
        if not GROQ_AVAILABLE:
            raise Exception("groq package not installed. Please install: pip install groq")
        if not GROQ_API_KEY or not GROQ_API_KEY.strip():
            raise Exception("GROQ_API_KEY not set in environment variables. Please set it in your .env file or environment.")
        if not groq_client:
            raise Exception("Groq client not initialized")
    
    # Limit content length to avoid token limits
    content_preview = content[:5000] if len(content) > 5000 else content
    
    if max_questions:
        prompt = f"""Based on the following content, create as many relevant multiple choice questions (MCQs) as possible. 
Cover all important concepts and topics from the content.

Content:
{content_preview}

Format each question EXACTLY as follows (this format is critical):

Question 1: [question text here]
A) [option A text]
B) [option B text]
C) [option C text]
D) [option D text]
Correct Answer: [A or B or C or D]
Explanation: [brief explanation of why this answer is correct]

Question 2: [question text here]
A) [option A text]
B) [option B text]
C) [option C text]
D) [option D text]
Correct Answer: [A or B or C or D]
Explanation: [brief explanation]

Continue this format for all questions. Create comprehensive questions covering all major topics in the content.
"""
    else:
        prompt = f"""Based on the following content, create exactly {num_questions} multiple choice questions (MCQs).

Content:
{content_preview}

Format each question EXACTLY as follows (this format is critical):

Question 1: [question text here]
A) [option A text]
B) [option B text]
C) [option C text]
D) [option D text]
Correct Answer: [A or B or C or D]
Explanation: [brief explanation of why this answer is correct]

Question 2: [question text here]
A) [option A text]
B) [option B text]
C) [option C text]
D) [option D text]
Correct Answer: [A or B or C or D]
Explanation: [brief explanation]

Continue this format for all {num_questions} questions. Cover the most important concepts from the content.
"""
    
    try:
        if provider == "gemini":
            # Use Gemini API
            response = gemini_client.models.generate_content(
                model=gemini_model,
                contents=prompt
            )
            if not response or not hasattr(response, 'text') or not response.text:
                raise Exception("No response from Gemini API")
            mcqs_text = response.text
        else:  # groq
            # Use Groq API
            chat_completion = groq_client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                model=GROQ_MODEL,
                temperature=0.7
            )
            mcqs_text = chat_completion.choices[0].message.content
            if not mcqs_text:
                raise Exception("No response from Groq API")
        
        return parse_mcqs(mcqs_text)
    except Exception as e:
        error_msg = str(e)
        if "pattern" in error_msg.lower() or "invalid" in error_msg.lower() or "model" in error_msg.lower():
            raise Exception(f"API error: Please check your API key and model availability. Error: {error_msg}")
        raise Exception(f"Error generating MCQs: {error_msg}")

def parse_mcqs(text):
    """Parse the generated MCQ text into structured format"""
    mcqs = []
    lines = text.strip().split('\n')
    
    current_mcq = {}
    current_question_num = 0
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Detect question start
        if line.lower().startswith('question') and ':' in line:
            # Save previous MCQ if exists
            if current_mcq and current_mcq.get('question'):
                mcqs.append(current_mcq)
            
            # Start new MCQ
            parts = line.split(':', 1)
            if len(parts) > 1:
                current_mcq = {
                    'question': parts[1].strip(),
                    'options': [],
                    'answer': '',
                    'explanation': ''
                }
                # Extract question number
                try:
                    num_part = parts[0].lower().replace('question', '').strip()
                    current_question_num = int(num_part) if num_part.isdigit() else len(mcqs) + 1
                except:
                    current_question_num = len(mcqs) + 1
            else:
                current_mcq = {
                    'question': line.replace('Question', '').replace('question', '').strip(),
                    'options': [],
                    'answer': '',
                    'explanation': ''
                }
        
        # Detect options (A), B), C), D))
        elif line.startswith(('A)', 'B)', 'C)', 'D)')):
            option_text = line[2:].strip() if len(line) > 2 else line.strip()
            if option_text:
                current_mcq['options'].append(line)
        
        # Detect correct answer
        elif line.lower().startswith('correct answer:'):
            answer_part = line.split(':', 1)[1].strip() if ':' in line else line.replace('Correct Answer', '').strip()
            # Extract just the letter (A, B, C, or D)
            answer_letter = answer_part[0].upper() if answer_part else ''
            if answer_letter in ['A', 'B', 'C', 'D']:
                current_mcq['answer'] = answer_letter
        
        # Detect explanation
        elif line.lower().startswith('explanation:'):
            explanation_part = line.split(':', 1)[1].strip() if ':' in line else line.replace('Explanation', '').strip()
            current_mcq['explanation'] = explanation_part
    
    # Add last MCQ if exists
    if current_mcq and current_mcq.get('question'):
        mcqs.append(current_mcq)
    
    # Validate and clean MCQs
    validated_mcqs = []
    for mcq in mcqs:
        if mcq.get('question') and len(mcq.get('options', [])) >= 4 and mcq.get('answer'):
            validated_mcqs.append({
                'question': mcq['question'],
                'options': mcq['options'][:4],  # Ensure only 4 options
                'answer': mcq['answer'],
                'explanation': mcq.get('explanation', '')
            })
    
    return validated_mcqs

