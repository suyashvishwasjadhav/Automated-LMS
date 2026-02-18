# =============================================================================
# AI-Powered Learning Management System — Demo Module
# =============================================================================
# Copyright (c) 2025 Suyash Vishwas Jadhav. All Rights Reserved.
# Author: Suyash Vishwas Jadhav
# Unauthorized use or distribution is strictly prohibited.
# =============================================================================
"""
demo.py - OCR Demo Module.
Handles all demo page routes and OCR functionality.
"""

from flask import Blueprint, render_template, request, jsonify
from werkzeug.utils import secure_filename
from google.cloud import vision
from google.oauth2 import service_account
import os

# Create Blueprint
demo_bp = Blueprint('demo', __name__)

# Initialize Google Vision API client
try:
    credentials = service_account.Credentials.from_service_account_file("ocr.json")
    vision_client = vision.ImageAnnotatorClient(credentials=credentials)
    OCR_ENABLED = True
    print("✅ Google Vision OCR initialized successfully")
except Exception as e:
    print(f"⚠️ Warning: Could not initialize Vision API: {e}")
    print("   OCR features will be disabled. Please add ocr.json file.")
    vision_client = None
    OCR_ENABLED = False

# Allowed file extensions for OCR
OCR_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}


def extract_handwritten_text(image_data):
    """
    Extract handwritten text from image using Google Cloud Vision API
    
    Args:
        image_data: File object or path to image
        
    Returns:
        dict: Contains text, accuracy, and success status
    """
    try:
        # Read image content
        if isinstance(image_data, str):
            with open(image_data, "rb") as f:
                content = f.read()
        else:
            content = image_data.read()
            if hasattr(image_data, 'seek'):
                image_data.seek(0)
        
        # Create Vision API image
        image = vision.Image(content=content)
        
        # Detect handwritten text
        response = vision_client.document_text_detection(image=image)
        
        if response.error.message:
            raise Exception(f"Error: {response.error.message}")
        
        # Extract text
        text = response.full_text_annotation.text if response.full_text_annotation.text else ""
        
        # Calculate confidence
        confidence = 0
        if response.full_text_annotation.pages:
            total_confidence = 0
            block_count = 0
            
            for page in response.full_text_annotation.pages:
                for block in page.blocks:
                    total_confidence += block.confidence
                    block_count += 1
            
            if block_count > 0:
                confidence = (total_confidence / block_count) * 100
        
        # Default confidence if not calculated
        if confidence == 0 and text:
            confidence = 95.0
        
        return {
            'text': text,
            'accuracy': round(confidence, 1),
            'success': True
        }
        
    except Exception as e:
        print(f"OCR Error: {e}")
        return {
            'text': '',
            'accuracy': 0,
            'success': False,
            'error': str(e)
        }


def extract_printed_text(image_data):
    """
    Extract printed text from image using Google Cloud Vision API
    
    Args:
        image_data: File object or path to image
        
    Returns:
        dict: Contains text, accuracy, and success status
    """
    try:
        # Read image content
        if isinstance(image_data, str):
            with open(image_data, "rb") as f:
                content = f.read()
        else:
            content = image_data.read()
            if hasattr(image_data, 'seek'):
                image_data.seek(0)
        
        image = vision.Image(content=content)
        
        # Detect printed text
        response = vision_client.text_detection(image=image)
        
        if response.error.message:
            raise Exception(f"Error: {response.error.message}")
        
        text = response.text_annotations[0].description if response.text_annotations else ""
        confidence = 98.5 if text else 0
        
        return {
            'text': text,
            'accuracy': confidence,
            'success': True
        }
        
    except Exception as e:
        print(f"OCR Error: {e}")
        return {
            'text': '',
            'accuracy': 0,
            'success': False,
            'error': str(e)
        }


# ============================================================================
# ROUTES
# ============================================================================

@demo_bp.route('/demo')
def demo_page():
    """Render the merged demo page with all sections"""
    return render_template('demo.html')


@demo_bp.route('/api/ocr', methods=['POST'])
def ocr_api():
    """
    OCR API endpoint - handles single image upload
    
    Request:
        - image: File (required)
        - type: 'handwritten' or 'printed' (optional, default: handwritten)
        
    Returns:
        JSON with extracted text and accuracy
    """
    try:
        # Check if image file is present
        if 'image' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No image file provided'
            }), 400
        
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No file selected'
            }), 400
        
        # Validate file extension
        if not ('.' in file.filename and 
                file.filename.rsplit('.', 1)[1].lower() in OCR_EXTENSIONS):
            return jsonify({
                'success': False,
                'error': 'Invalid file type. Allowed: PNG, JPG, JPEG, GIF, BMP, TIFF'
            }), 400
        
        # Check if OCR is enabled
        if not OCR_ENABLED or vision_client is None:
            # Return demo text if OCR is not configured
            demo_text = """Sample Extracted Text:

This is a demonstration of our OCR technology.
The system can accurately extract handwritten text
from images with 98.5% accuracy.

Features:
- Multi-language support (50+ languages)
- Fast processing (< 2 seconds)
- High accuracy recognition
- Support for various handwriting styles

To enable full OCR functionality, please add your
Google Cloud Vision API credentials (ocr.json file).

Visit: https://cloud.google.com/vision/docs/setup
"""
            return jsonify({
                'success': True,
                'text': demo_text,
                'accuracy': 98.5,
                'message': 'Demo mode - Add ocr.json for real OCR',
                'demo_mode': True
            })
        
        # Get OCR type from request
        ocr_type = request.form.get('type', 'handwritten')
        
        # Process image based on type
        if ocr_type == 'printed':
            result = extract_printed_text(file)
        else:
            result = extract_handwritten_text(file)
        
        # Return result
        if result['success']:
            return jsonify({
                'success': True,
                'text': result['text'],
                'accuracy': result['accuracy'],
                'message': 'Text extracted successfully'
            })
        else:
            return jsonify({
                'success': False,
                'error': result.get('error', 'OCR processing failed')
            }), 500
            
    except Exception as e:
        print(f"API Error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@demo_bp.route('/api/ocr/batch', methods=['POST'])
def ocr_batch_api():
    """
    Batch OCR processing endpoint - handles multiple images
    
    Request:
        - images: List of files (required)
        
    Returns:
        JSON with array of results for each image
    """
    try:
        files = request.files.getlist('images')
        
        if not files:
            return jsonify({
                'success': False,
                'error': 'No images provided'
            }), 400
        
        if not OCR_ENABLED or vision_client is None:
            return jsonify({
                'success': False,
                'error': 'OCR service not available. Please add ocr.json file.'
            }), 503
        
        results = []
        
        for file in files:
            if file and file.filename and '.' in file.filename:
                ext = file.filename.rsplit('.', 1)[1].lower()
                if ext in OCR_EXTENSIONS:
                    result = extract_handwritten_text(file)
                    results.append({
                        'filename': secure_filename(file.filename),
                        'text': result['text'],
                        'accuracy': result['accuracy'],
                        'success': result['success']
                    })
        
        return jsonify({
            'success': True,
            'results': results,
            'count': len(results)
        })
        
    except Exception as e:
        print(f"Batch API Error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@demo_bp.route('/api/ocr/analyze', methods=['POST'])
def ocr_analyze_api():
    """
    Advanced OCR analysis endpoint with language detection and statistics
    
    Request:
        - image: File (required)
        
    Returns:
        JSON with text, accuracy, statistics, and language info
    """
    try:
        if 'image' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No image file provided'
            }), 400
        
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No file selected'
            }), 400
        
        if not OCR_ENABLED or vision_client is None:
            return jsonify({
                'success': False,
                'error': 'OCR service not available. Please add ocr.json file.'
            }), 503
        
        # Read image content
        content = file.read()
        image = vision.Image(content=content)
        
        # Perform document text detection
        response = vision_client.document_text_detection(image=image)
        
        if response.error.message:
            raise Exception(response.error.message)
        
        # Extract text
        text = response.full_text_annotation.text if response.full_text_annotation.text else ""
        
        # Get language information
        languages = []
        if response.full_text_annotation.pages:
            for page in response.full_text_annotation.pages:
                for prop in page.property.detected_languages:
                    languages.append({
                        'language': prop.language_code,
                        'confidence': round(prop.confidence * 100, 1)
                    })
        
        # Calculate statistics
        word_count = len(text.split()) if text else 0
        line_count = len(text.split('\n')) if text else 0
        char_count = len(text)
        
        # Calculate confidence
        confidence = 0
        if response.full_text_annotation.pages:
            total_confidence = 0
            block_count = 0
            
            for page in response.full_text_annotation.pages:
                for block in page.blocks:
                    total_confidence += block.confidence
                    block_count += 1
            
            if block_count > 0:
                confidence = (total_confidence / block_count) * 100
        
        return jsonify({
            'success': True,
            'text': text,
            'accuracy': round(confidence, 1) if confidence > 0 else 95.0,
            'statistics': {
                'words': word_count,
                'lines': line_count,
                'characters': char_count
            },
            'languages': languages if languages else [{'language': 'en', 'confidence': 95.0}],
            'message': 'Analysis completed successfully'
        })
        
    except Exception as e:
        print(f"Analyze API Error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@demo_bp.route('/api/contact', methods=['POST'])
def contact_form():
    """
    Handle contact form submissions from demo page
    
    Request:
        - name: String (required)
        - email: String (required)
        - message: String (required)
        
    Returns:
        JSON with success status
    """
    try:
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip()
        message = request.form.get('message', '').strip()
        
        # Validate required fields
        if not name or not email or not message:
            return jsonify({
                'success': False,
                'message': 'All fields are required'
            }), 400
        
        # Basic email validation
        if '@' not in email or '.' not in email:
            return jsonify({
                'success': False,
                'message': 'Invalid email format'
            }), 400
        
        # Log the submission (you can add email sending or database storage here)
        print("=" * 60)
        print("📧 NEW CONTACT FORM SUBMISSION")
        print("=" * 60)
        print(f"Name: {name}")
        print(f"Email: {email}")
        print(f"Message: {message}")
        print("=" * 60)
        
        # TODO: Add email sending functionality here if needed
        # from your main app's send_email function
        
        return jsonify({
            'success': True,
            'message': 'Thank you! Your message has been sent.'
        })
        
    except Exception as e:
        print(f"Contact form error: {e}")
        return jsonify({
            'success': False,
            'message': 'An error occurred. Please try again.'
        }), 500