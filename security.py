# =============================================================================
# AI-Powered Learning Management System — Security Module
# =============================================================================
# Copyright (c) 2025 Suyash Vishwas Jadhav. All Rights Reserved.
# Author: Suyash Vishwas Jadhav
# Unauthorized use or distribution is strictly prohibited.
# =============================================================================
"""
Comprehensive Security Module for the Application.
Provides protection against SQL injection, XSS, CSRF, and other threats.
"""

import re
import html
import json
import base64
import hashlib
import secrets
from typing import Any, Optional, Union, Dict, List
from functools import wraps
from datetime import datetime, timedelta
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from werkzeug.security import generate_password_hash, check_password_hash
import bleach
from bleach.css_sanitizer import CSSSanitizer


class SecurityConfig:
    """Security configuration constants"""
    # Encryption key derivation - should be stored in environment variable
    ENCRYPTION_KEY = None  # Will be set from environment or generated
    
    # Rate limiting
    MAX_LOGIN_ATTEMPTS = 5
    LOGIN_LOCKOUT_TIME = 900  # 15 minutes in seconds
    
    # Input validation
    MAX_STRING_LENGTH = 10000
    MAX_EMAIL_LENGTH = 255
    MAX_PASSWORD_LENGTH = 128
    
    # SQL injection patterns
    SQL_INJECTION_PATTERNS = [
        r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|EXECUTE|UNION|SCRIPT)\b)",
        r"(--|#|/\*|\*/|;|\||&)",
        r"(\b(OR|AND)\s+\d+\s*=\s*\d+)",
        r"(\b(OR|AND)\s+['\"].*?['\"]\s*=\s*['\"].*?['\"])",
        r"(\bUNION\s+(ALL\s+)?SELECT)",
        r"(xp_|sp_|cmdshell)",
        r"(\bLOAD_FILE\b|\bINTO\s+OUTFILE\b)",
    ]
    
    # XSS patterns
    XSS_PATTERNS = [
        r"<script[^>]*>.*?</script>",
        r"javascript:",
        r"on\w+\s*=",
        r"<iframe[^>]*>",
        r"<object[^>]*>",
        r"<embed[^>]*>",
        r"<link[^>]*>",
        r"<meta[^>]*>",
        r"<style[^>]*>.*?</style>",
        r"expression\s*\(",
    ]
    
    # Command injection patterns
    COMMAND_INJECTION_PATTERNS = [
        r"[;&|`$(){}[\]<>]",
        r"\b(cat|ls|pwd|whoami|id|uname|ps|kill|rm|mv|cp|chmod|chown)\b",
        r"(\$\{|\$\(|`|&&|\|\|)",
    ]


class EncryptionService:
    """Service for encrypting and decrypting sensitive data"""
    
    _instance = None
    _fernet = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EncryptionService, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Initialize encryption with key from environment or generate one"""
        import os
        key_env = os.environ.get('ENCRYPTION_KEY')
        if key_env:
            # Use provided key
            key = key_env.encode()
        else:
            # Generate and store key (in production, this should be set via environment)
            key_file = '.encryption_key'
            if os.path.exists(key_file):
                with open(key_file, 'rb') as f:
                    key = f.read()
            else:
                key = Fernet.generate_key()
                with open(key_file, 'wb') as f:
                    f.write(key)
                print("⚠️  Generated new encryption key. Store this securely!")
        
        # Ensure key is 32 bytes for Fernet
        if len(key) != 44:  # Fernet keys are base64 encoded, 44 chars
            # Derive key from password
            password = key if isinstance(key, bytes) else key.encode()
            salt = b'secure_salt_12345678'  # In production, use random salt
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
            )
            key = base64.urlsafe_b64encode(kdf.derive(password))
        
        self._fernet = Fernet(key)
    
    def encrypt(self, data: str) -> str:
        """Encrypt sensitive data"""
        if not data:
            return ""
        try:
            encrypted = self._fernet.encrypt(data.encode())
            return base64.urlsafe_b64encode(encrypted).decode()
        except Exception as e:
            print(f"Encryption error: {e}")
            return data
    
    def decrypt(self, encrypted_data: str) -> str:
        """Decrypt sensitive data"""
        if not encrypted_data:
            return ""
        try:
            decoded = base64.urlsafe_b64decode(encrypted_data.encode())
            decrypted = self._fernet.decrypt(decoded)
            return decrypted.decode()
        except Exception as e:
            print(f"Decryption error: {e}")
            return encrypted_data


class InputValidator:
    """Validates and sanitizes user input"""
    
    @staticmethod
    def sanitize_string(value: Any, max_length: int = SecurityConfig.MAX_STRING_LENGTH) -> str:
        """Sanitize string input"""
        if value is None:
            return ""
        
        # Convert to string
        value = str(value)
        
        # Trim whitespace
        value = value.strip()
        
        # Limit length
        if len(value) > max_length:
            value = value[:max_length]
        
        # Remove null bytes
        value = value.replace('\x00', '')
        
        return value
    
    @staticmethod
    def sanitize_html(value: str, allowed_tags: Optional[List[str]] = None) -> str:
        """Sanitize HTML input to prevent XSS"""
        if not value:
            return ""
        
        # Default allowed tags (very restrictive)
        if allowed_tags is None:
            allowed_tags = ['p', 'br', 'strong', 'em', 'u', 'ol', 'ul', 'li']
        
        # Use bleach to sanitize HTML
        css_sanitizer = CSSSanitizer(allowed_css_properties=[])
        cleaned = bleach.clean(
            value,
            tags=allowed_tags,
            attributes={},
            css_sanitizer=css_sanitizer,
            strip=True
        )
        
        return cleaned
    
    @staticmethod
    def validate_email(email: str) -> bool:
        """Validate email format"""
        if not email:
            return False
        
        email = email.strip().lower()
        
        # Basic email regex
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(pattern, email):
            return False
        
        # Check length
        if len(email) > SecurityConfig.MAX_EMAIL_LENGTH:
            return False
        
        return True
    
    @staticmethod
    def validate_password(password: str) -> tuple[bool, str]:
        """Validate password strength"""
        if not password:
            return False, "Password is required"
        
        if len(password) < 8:
            return False, "Password must be at least 8 characters long"
        
        if len(password) > SecurityConfig.MAX_PASSWORD_LENGTH:
            return False, f"Password must be less than {SecurityConfig.MAX_PASSWORD_LENGTH} characters"
        
        # Check for at least one uppercase, one lowercase, one digit
        if not re.search(r'[A-Z]', password):
            return False, "Password must contain at least one uppercase letter"
        
        if not re.search(r'[a-z]', password):
            return False, "Password must contain at least one lowercase letter"
        
        if not re.search(r'\d', password):
            return False, "Password must contain at least one digit"
        
        return True, "Password is valid"
    
    @staticmethod
    def check_sql_injection(value: str) -> bool:
        """Check if input contains SQL injection patterns"""
        if not value:
            return False
        
        value_upper = value.upper()
        for pattern in SecurityConfig.SQL_INJECTION_PATTERNS:
            if re.search(pattern, value_upper, re.IGNORECASE):
                return True
        return False
    
    @staticmethod
    def check_xss(value: str) -> bool:
        """Check if input contains XSS patterns"""
        if not value:
            return False
        
        for pattern in SecurityConfig.XSS_PATTERNS:
            if re.search(pattern, value, re.IGNORECASE):
                return True
        return False
    
    @staticmethod
    def check_command_injection(value: str) -> bool:
        """Check if input contains command injection patterns"""
        if not value:
            return False
        
        for pattern in SecurityConfig.COMMAND_INJECTION_PATTERNS:
            if re.search(pattern, value):
                return True
        return False
    
    @staticmethod
    def sanitize_input(value: Any, input_type: str = "string") -> str:
        """Comprehensive input sanitization"""
        if value is None:
            return ""
        
        # Convert to string
        sanitized = InputValidator.sanitize_string(value)
        
        # Check for injection attacks
        if InputValidator.check_sql_injection(sanitized):
            raise ValueError("Invalid input detected: potential SQL injection")
        
        if InputValidator.check_xss(sanitized):
            # Sanitize HTML instead of raising error for XSS
            sanitized = InputValidator.sanitize_html(sanitized)
        
        if InputValidator.check_command_injection(sanitized):
            raise ValueError("Invalid input detected: potential command injection")
        
        return sanitized


class RateLimiter:
    """Simple in-memory rate limiter (for production, use Redis)"""
    
    _attempts: Dict[str, List[datetime]] = {}
    
    @classmethod
    def check_rate_limit(cls, identifier: str, max_attempts: int = SecurityConfig.MAX_LOGIN_ATTEMPTS,
                        window_seconds: int = SecurityConfig.LOGIN_LOCKOUT_TIME) -> tuple[bool, Optional[int]]:
        """Check if rate limit is exceeded"""
        now = datetime.now()
        key = f"rate_limit_{identifier}"
        
        # Clean old attempts
        if key in cls._attempts:
            cls._attempts[key] = [
                attempt for attempt in cls._attempts[key]
                if (now - attempt).total_seconds() < window_seconds
            ]
        else:
            cls._attempts[key] = []
        
        # Check if limit exceeded
        if len(cls._attempts[key]) >= max_attempts:
            oldest_attempt = cls._attempts[key][0]
            remaining = int(window_seconds - (now - oldest_attempt).total_seconds())
            return False, remaining
        
        return True, None
    
    @classmethod
    def record_attempt(cls, identifier: str):
        """Record an attempt"""
        key = f"rate_limit_{identifier}"
        if key not in cls._attempts:
            cls._attempts[key] = []
        cls._attempts[key].append(datetime.now())
    
    @classmethod
    def reset_attempts(cls, identifier: str):
        """Reset attempts for an identifier"""
        key = f"rate_limit_{identifier}"
        if key in cls._attempts:
            del cls._attempts[key]


class SecurityHeaders:
    """Middleware for adding security headers"""
    
    @staticmethod
    def get_headers() -> Dict[str, str]:
        """Get security headers dictionary"""
        return {
            'X-Content-Type-Options': 'nosniff',
            'X-Frame-Options': 'DENY',
            'X-XSS-Protection': '1; mode=block',
            'Strict-Transport-Security': 'max-age=31536000; includeSubDomains',
            # Comprehensive CSP allowing all necessary APIs and resources
            'Content-Security-Policy': (
                "default-src 'self'; "
                "script-src 'self' 'unsafe-inline' 'unsafe-eval' https://cdnjs.cloudflare.com https://cdn.jsdelivr.net https://cdn.socket.io; "
                "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com https://cdnjs.cloudflare.com https://cdn.jsdelivr.net; "
                "font-src 'self' data: https://fonts.gstatic.com https://cdnjs.cloudflare.com; "
                "img-src 'self' data: https: blob:; "
                # Allow all Google APIs and services
                "connect-src 'self' "
                "https://fonts.googleapis.com https://cdnjs.cloudflare.com "
                # Google APIs
                "https://oauth2.googleapis.com "
                "https://www.googleapis.com "
                "https://accounts.google.com "
                "https://generativelanguage.googleapis.com "
                "https://vision.googleapis.com "
                "https://*.googleapis.com "
                # Other AI/ML APIs
                "https://api.deepseek.com "
                "https://api.groq.com "
                # Socket.IO CDN and WebSocket connections
                "https://cdn.socket.io wss://* ws://* "
                # News APIs
                "https://newsapi.org "
                "https://gnews.io "
                "https://api.gnews.io "
                # SMTP (for email)
                "https://smtp.gmail.com; "
                "frame-src 'self' https://accounts.google.com; "
                "object-src 'none'; "
                "base-uri 'self';"
            ),
            'Referrer-Policy': 'strict-origin-when-cross-origin',
            'Permissions-Policy': 'geolocation=(), microphone=(), camera=()'
        }


# Decorator for secure input handling
def secure_input(input_type: str = "string", required: bool = False, max_length: int = None):
    """Decorator to secure route inputs"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            from flask import request, jsonify
            
            # Process form data
            if request.method == 'POST':
                form_data = {}
                for key, value in request.form.items():
                    try:
                        if max_length:
                            sanitized = InputValidator.sanitize_input(value, input_type)[:max_length]
                        else:
                            sanitized = InputValidator.sanitize_input(value, input_type)
                        form_data[key] = sanitized
                    except ValueError as e:
                        return jsonify({'success': False, 'message': str(e)}), 400
                
                # Replace request.form with sanitized version
                request._form_data = form_data
                
                # Process JSON data
                if request.is_json:
                    try:
                        json_data = request.get_json()
                        if json_data:
                            sanitized_json = {}
                            for key, value in json_data.items():
                                if isinstance(value, str):
                                    try:
                                        sanitized_json[key] = InputValidator.sanitize_input(value, input_type)
                                    except ValueError as e:
                                        return jsonify({'success': False, 'message': str(e)}), 400
                                else:
                                    sanitized_json[key] = value
                            request._json_data = sanitized_json
                    except Exception as e:
                        return jsonify({'success': False, 'message': 'Invalid JSON data'}), 400
                
                # Process query parameters
                args_data = {}
                for key, value in request.args.items():
                    try:
                        args_data[key] = InputValidator.sanitize_input(value, input_type)
                    except ValueError as e:
                        return jsonify({'success': False, 'message': str(e)}), 400
                request._args_data = args_data
            
            return f(*args, **kwargs)
        return decorated_function
    return decorator


# Helper function to get sanitized form data
def get_sanitized_form(key: str, default: Any = None) -> str:
    """Get sanitized form data"""
    from flask import request
    if hasattr(request, '_form_data'):
        return request._form_data.get(key, default)
    value = request.form.get(key, default)
    if value:
        return InputValidator.sanitize_input(value)
    return default or ""


# Helper function to get sanitized JSON data
def get_sanitized_json(key: str = None, default: Any = None) -> Any:
    """Get sanitized JSON data"""
    from flask import request
    if hasattr(request, '_json_data'):
        data = request._json_data
    elif request.is_json:
        data = request.get_json() or {}
        # Sanitize string values
        sanitized = {}
        for k, v in data.items():
            if isinstance(v, str):
                try:
                    sanitized[k] = InputValidator.sanitize_input(v)
                except ValueError:
                    sanitized[k] = v
            else:
                sanitized[k] = v
        data = sanitized
    else:
        data = {}
    
    if key:
        return data.get(key, default)
    return data


# Helper function to get sanitized args
def get_sanitized_args(key: str, default: Any = None) -> str:
    """Get sanitized query arguments"""
    from flask import request
    if hasattr(request, '_args_data'):
        return request._args_data.get(key, default)
    value = request.args.get(key, default)
    if value:
        return InputValidator.sanitize_input(value)
    return default or ""

