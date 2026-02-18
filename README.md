# AI-Powered Learning Management System

**An enterprise-grade, AI-integrated Learning Management System built with Flask, PostgreSQL, and multiple AI providers including Google Gemini, Groq (LLaMA), and DeepSeek. Designed for organizations that need intelligent course management, automated assignment evaluation, real-time communication, and advanced plagiarism detection.**

---

## Team & Contributors

### Lead Developer

**Suyash Vishwas Jadhav**

> Developer · Designer · Architect · Idea Creator · Full Stack Integration · Database Design · Frontend Development · Backend Development · Security Implementation · Cloud Deployment (GCP) · Testing & QA

### Team Members

| Name | Contribution |
|------|--------------|
| **Harsh Gawande** | Diagram Evaluation — built the AI-powered system that evaluates hand-drawn and digital diagrams submitted by learners, scoring them across topic match, accuracy, structure, visual quality, and technical correctness |
| **Nidhi Pawar** | Plagiarism Detection — built the rule-based plagiarism scoring engine that analyzes student text submissions, detects copied or AI-generated content, and returns a plagiarism confidence score |

---

## Problem Statement

Traditional Learning Management Systems (LMS) are passive repositories — they store content and collect submissions, but they do not evaluate, analyse, or adapt. Educators spend enormous amounts of time manually grading handwritten assignments, checking for plagiarism, generating quiz questions, and tracking student progress across multiple tools. This creates three critical problems:

**1. Evaluation Bottleneck**
In large cohorts, a single guide may receive hundreds of handwritten answer sheets per assignment. Manual grading is slow, inconsistent, and subjective. There is no standardized feedback mechanism, and learners often receive grades without understanding where they went wrong.

**2. Academic Integrity Gaps**
Existing plagiarism tools are expensive, require internet access to third-party databases, and cannot detect AI-generated content. Organizations running internal training programs have no affordable way to verify the originality of learner submissions.

**3. Fragmented Learning Infrastructure**
Organizations use separate tools for content delivery, communication, assessment, and reporting. This fragmentation increases administrative overhead, creates data silos, and makes it impossible to get a unified view of learner performance.

**Our Solution**
This AI-powered LMS solves all three problems in a single, integrated platform:
- Handwritten assignments are automatically graded using Google Cloud Vision OCR and evaluated by large language models (Gemini, Groq, DeepSeek) with detailed, per-criterion feedback.
- Plagiarism and AI-content detection runs automatically on every submission using a rule-based engine — no third-party subscription required.
- A unified platform handles course management, material delivery, real-time messaging, MCQ test generation, document Q&A, diagram evaluation, and scoreboard reporting — all in one place, deployable on Google Cloud Run for any organization.

---

## Quick Navigation

> Click any section below to jump directly to it.

| # | Section | Description |
|---|---------|-------------|
| 1 | [Project Overview](#project-overview) | What this platform does and achieves |
| 2 | [Architecture Diagram](#architecture-diagram) | Full system architecture (ASCII diagram) |
| 3 | [Technology Stack](#technology-stack) | Backend, AI/ML, Frontend, GCP services |
| 4 | [Project Structure](#project-structure) | Complete file and folder tree |
| 5 | [Core Modules](#core-modules) | Deep-dive into each Python module |
| 6 | [Database Schema](#database-schema) | All 20+ tables and relationships |
| 7 | [User Roles and Workflows](#user-roles-and-workflows) | Executive, Guide, Learner flows |
| 8 | [Feature Breakdown](#feature-breakdown) | Evaluation pipeline, RAG, MCQ, Diagrams |
| 9 | [API Reference](#api-reference) | All 80+ endpoints with methods |
| 10 | [Security Implementation](#security-implementation) | CSRF, rate limiting, session security |
| 11 | [AI Integration Details](#ai-integration-details) | Gemini, Groq, DeepSeek, Vision API |
| 12 | [Local Development Setup](#local-development-setup) | Step-by-step local setup guide |
| 13 | [Environment Variables](#environment-variables) | All required env vars with examples |
| 14 | [GCP Deployment Guide](#gcp-deployment-guide) | 5-phase GCP deployment walkthrough |
| 15 | [Final Deployment Steps](#final-deployment-steps) | Checklist and deployment commands |
| 16 | [Cost Monitoring](#cost-monitoring) | GCP cost estimates and controls |
| 17 | [Future Scope](#future-scope) | Short, medium, and long-term roadmap |

---

---

## Project Overview

This platform is a full-stack, AI-powered Learning Management System designed for corporate and academic organizations. It supports three distinct user roles: **Executive** (organization administrator), **Guide** (instructor/teacher), and **Learner** (student). The system automates the most time-consuming aspects of education management, including assignment grading, plagiarism detection, MCQ generation, and document-based question answering, all powered by state-of-the-art large language models.

### What This Platform Achieves

- Executives create courses, assign guides, enroll learners, and monitor organizational analytics from a single dashboard.
- Guides upload learning materials, create and manage assignments, evaluate handwritten submissions using OCR, detect plagiarism, generate MCQ tests from topics or documents, and export scoreboards.
- Learners access course materials, submit handwritten assignments as image uploads, take AI-generated MCQ tests, interact with an AI chatbot, and query uploaded documents using natural language.
- The entire system is secured with CSRF protection, rate limiting, session fixation prevention, input sanitization, and security headers.

---

## Architecture Diagram

```
+-----------------------------------------------------------------------------------+
|                          CLIENT BROWSER                                           |
|   Login / Register / Demo / Executive Dashboard / Guide Dashboard / Learner       |
+---------------------------+-------------------------------------------------------+
                            |  HTTP / WebSocket (Socket.IO)
                            v
+-----------------------------------------------------------------------------------+
|                        FLASK APPLICATION SERVER (app.py)                          |
|                                                                                   |
|  +------------------+  +------------------+  +------------------+                |
|  |  Auth Module     |  |  Executive       |  |  Guide Module    |                |
|  |  - Register      |  |  Module          |  |  - Upload        |                |
|  |  - Login         |  |  - Create Course |  |  - Assignments   |                |
|  |  - Google OAuth  |  |  - Add Learners  |  |  - Scoreboard    |                |
|  |  - OTP Reset     |  |  - Announcements |  |  - MCQ Tests     |                |
|  +------------------+  +------------------+  +------------------+                |
|                                                                                   |
|  +------------------+  +------------------+  +------------------+                |
|  |  Learner Module  |  |  Messaging       |  |  Demo Module     |                |
|  |  - Materials     |  |  Module          |  |  - OCR Demo      |                |
|  |  - Submit Assign |  |  - Real-time     |  |  - Contact Form  |                |
|  |  - MCQ Tests     |  |  - Socket.IO     |  |                  |                |
|  |  - Document QA   |  |                  |  |                  |                |
|  +------------------+  +------------------+  +------------------+                |
|                                                                                   |
|  +------------------+  +------------------+  +------------------+                |
|  |  security.py     |  |  evaluation.py   |  |  mcq.py          |                |
|  |  - Input Valid.  |  |  - OCR Extract   |  |  - Topic MCQs    |                |
|  |  - XSS/SQLi      |  |  - Gemini Eval   |  |  - Doc MCQs      |                |
|  |  - Rate Limit    |  |  - Groq Eval     |  |  - Parse MCQs    |                |
|  |  - Encryption    |  |  - DeepSeek Eval |  |                  |                |
|  +------------------+  +------------------+  +------------------+                |
|                                                                                   |
|  +------------------+                                                             |
|  | plagiarism_      |                                                             |
|  | detector.py      |                                                             |
|  | - Rule-based     |                                                             |
|  | - AI Detection   |                                                             |
|  +------------------+                                                             |
+---------------------------+-------------------------------------------------------+
                            |
          +-----------------+------------------+
          |                 |                  |
          v                 v                  v
+------------------+ +-------------+  +------------------+
| PostgreSQL DB    | | Google APIs |  | AI APIs          |
| (Cloud SQL /     | | - Vision    |  | - Gemini 2.5     |
|  Local)          | |   (OCR)     |  |   Flash          |
| - users          | | - OAuth 2.0 |  | - Groq LLaMA     |
| - courses        | | - Generative|  |   3.3-70B        |
| - assignments    | |   AI        |  | - DeepSeek Chat  |
| - submissions    | +-------------+  +------------------+
| - messages       |
| - document_qa    |
| - test_mcqs      |
| - notifications  |
+------------------+
          |
          v
+------------------+
| Cloud Storage /  |
| Local Filesystem |
| - uploads/       |
| - logos/         |
| - photos/        |
| - assignment_    |
|   uploads/       |
| - diagram_       |
|   uploads/       |
+------------------+
```

---

## Technology Stack

### Backend

| Component | Technology |
|-----------|-----------|
| Web Framework | Flask 3.x |
| ORM | SQLAlchemy with scoped sessions |
| Database | PostgreSQL (psycopg2-binary) |
| Authentication | Flask-Login, Authlib (Google OAuth 2.0) |
| Real-time Messaging | Flask-SocketIO (threading mode) |
| Email | Flask-Mail, smtplib (SMTP/Gmail) |
| Security | Flask-WTF (CSRF), Flask-Limiter, bleach, cryptography |
| PDF Generation | ReportLab |
| Document Parsing | pypdf, PyPDF2, python-docx, python-pptx |
| OCR | Google Cloud Vision API |
| Text-to-Speech | gTTS |
| Speech-to-Text | OpenAI Whisper (optional), Groq Whisper API |
| Production Server | Gunicorn + Eventlet |

### AI and Machine Learning

| Provider | Models Used | Purpose |
|----------|-------------|---------|
| Google Gemini | gemini-2.5-flash | Assignment evaluation, MCQ generation, document QA |
| Groq | llama-3.3-70b-versatile | Assignment evaluation, MCQ generation, chatbot |
| DeepSeek | deepseek-chat | Assignment evaluation, MCQ generation |
| Google Cloud Vision | document_text_detection | Handwritten text OCR |
| Sentence Transformers | all-MiniLM-L6-v2 | Document embeddings for RAG |
| FAISS | faiss-cpu | Vector similarity search |

### Frontend

| Component | Technology |
|-----------|-----------|
| Templating | Jinja2 (Flask) |
| Styling | Vanilla CSS (glassmorphism design) |
| Real-time | Socket.IO client |
| Charts | Chart.js |
| Icons | Font Awesome |
| Fonts | Google Fonts |

### Infrastructure (GCP)

| Service | Purpose |
|---------|---------|
| Cloud Run | Serverless container hosting |
| Cloud SQL (PostgreSQL 15) | Managed database |
| Cloud Storage | File uploads |
| Secret Manager | Secure credential storage |
| Cloud Build | Container image building |
| Cloud Vision API | OCR processing |

---

## Project Structure

```
AI LMS/
|
|-- app.py                        # Main Flask application (7,951 lines)
|   |-- Database Models (20+)
|   |-- Authentication Routes
|   |-- Executive Routes
|   |-- Guide Routes
|   |-- Learner Routes
|   |-- Messaging Routes
|   |-- Assignment Routes
|   |-- MCQ Test Routes
|   |-- Document QA Routes
|   |-- Diagram Evaluation Routes
|   |-- Plagiarism Routes
|   |-- Scoreboard Routes
|   |-- SocketIO Event Handlers
|
|-- evaluation.py                 # AI evaluation engine
|   |-- extract_handwritten_text()
|   |-- evaluate_with_gemini()
|   |-- evaluate_with_deepseek()
|   |-- evaluate_with_groq()
|   |-- evaluate_answer()
|   |-- generate_question_mcq()
|   |-- generate_answer_mcq()
|   |-- detect_plagiarism_realtime()
|
|-- mcq.py                        # MCQ generation module
|   |-- generate_mcqs_from_topic()
|   |-- generate_mcqs_from_document()
|   |-- parse_mcqs()
|   |-- extract_text_from_file()
|
|-- plagiarism_detector.py        # Plagiarism and AI detection
|   |-- check_plagiarism()
|   |-- detect_ai_content_rule_based()
|   |-- analyze_submission_text()
|
|-- security.py                   # Security module
|   |-- InputValidator
|   |-- EncryptionService
|   |-- RateLimiter
|   |-- SecurityHeaders
|
|-- demo.py                       # OCR demo blueprint
|   |-- /demo
|   |-- /api/ocr
|   |-- /api/ocr/batch
|   |-- /api/ocr/analyze
|
|-- requirements.txt              # Python dependencies
|-- Dockerfile                    # Container definition
|-- deploy.sh                     # GCP deployment script
|-- .env.example                  # Environment variable template
|-- .gitignore                    # Git ignore rules
|-- .dockerignore                 # Docker ignore rules
|
|-- templates/                    # Jinja2 HTML templates
|   |-- login.html
|   |-- register.html
|   |-- complete_profile.html
|   |-- forgot_password.html
|   |-- executive_dashboard.html
|   |-- guide_dashboard.html
|   |-- learner_dashboard.html
|   |-- messages.html
|   |-- mcq_test.html
|   |-- demo.html
|
|-- static/
    |-- css/
    |   |-- loginglass.css        # Login/register glassmorphism styles
    |   |-- glass.css             # Shared glass UI components
    |   |-- executive_glass.css   # Executive dashboard styles
    |   |-- guide.css             # Guide dashboard styles
    |   |-- guidebot.css          # AI chatbot widget styles
    |   |-- messages_style.css    # Messaging interface styles
    |   |-- messages_animations.css
    |   |-- register.css
    |   |-- demo.css              # Demo page styles
    |
    |-- js/
    |   |-- executive_dashboard.js
    |   |-- guidebot.js           # Guide AI chatbot
    |   |-- guidebot_guide.js     # Guide-side bot logic
    |   |-- guidebot_learner.js   # Learner-side bot logic
    |   |-- messages_script.js    # Real-time messaging
    |   |-- session_manager.js    # Session timeout handling
    |   |-- theme-toggle.js       # Dark/light mode toggle
    |   |-- demo.js               # Demo page interactions
    |
    |-- uploads/                  # Course material uploads
    |-- assignment_uploads/       # Assignment submission images
    |-- diagram_uploads/          # Diagram submission images
    |-- document_qa_uploads/      # Document QA file uploads
    |-- logos/                    # Organization logos
    |-- photos/                   # User profile photos
```

---

## Core Modules

### app.py (7,951 lines)

The central application file containing all route handlers, database models, middleware, and business logic. It is organized into the following logical sections:

**Imports and Configuration (lines 1-135)**
Loads all dependencies, configures AI clients (Gemini, Groq, DeepSeek), sets up Whisper language support for 60+ languages, and initializes Flask extensions.

**Flask App Setup (lines 163-425)**
Configures session security, CSRF protection, rate limiting, security headers middleware, session timeout middleware, and database connection.

**Database Models (lines 428-911)**
Defines 20+ SQLAlchemy ORM models covering users, courses, assignments, submissions, messages, notifications, MCQ tests, document QA, diagram evaluations, plagiarism tracking, and scoreboards.

**Authentication Routes (lines 987-1531)**
Handles registration, login, Google OAuth 2.0 flow, profile completion, and password reset via OTP email.

**Executive Routes (lines 1532-2602)**
Course creation, guide assignment, learner enrollment, announcement posting, user deletion with OTP verification, and analytics APIs.

**Guide Routes (lines 2603-4900+)**
Material uploads, assignment creation and management, submission review, AI-powered evaluation, scoreboard management, PDF/CSV export, MCQ test creation, diagram assignment management, and plagiarism review.

**Learner Routes (lines 2670-4600+)**
Dashboard, material access, assignment submission (image upload), MCQ test taking, document QA, AI chatbot interaction, and progress tracking.

**Messaging and SocketIO (lines 2783-3460)**
Real-time bidirectional messaging between all user roles using Socket.IO with room-based isolation.

**MCQ Test System (lines 4900-6000+)**
Complete test lifecycle: creation, assignment to courses, learner attempts with time limits, scoring, and result reporting.

**Document QA System (lines 6000-7000+)**
RAG-based document question answering using Gemini embeddings, FAISS vector search, and chunked document storage.

**Diagram Evaluation (lines 7000-7908)**
AI-powered evaluation of hand-drawn or digital diagrams submitted as images, with multi-dimensional scoring.

---

### evaluation.py (612 lines)

The AI evaluation engine that powers assignment grading. It supports three AI providers and uses a weighted scoring model:

```
Total Score = Relevance (50%) + Grammar (10%) + Size (30%) + Uniqueness (10%)
```

- **Relevance Score (0-50):** Measures whether the answer addresses the question. If relevance is 0, the total score is forced to 0 regardless of other scores.
- **Grammar Score (0-10):** Evaluates language quality and correctness.
- **Size Score (0-30):** Calculated based on word count relative to expected range (50-200 words default). Penalizes both very short and excessively long answers.
- **Uniqueness Score (0-10):** Rewards original thinking and creativity.

The module also generates two types of MCQs per submission: one testing the question topic, and one testing whether the learner understood what they wrote.

---

### mcq.py (435 lines)

Generates multiple-choice questions from two sources:

1. **Topic-based generation:** Given a topic, subcategory, difficulty level (easy/medium/hard), and option difficulty (identifiable/tricky), generates N questions using Gemini or Groq.
2. **Document-based generation:** Extracts text from PDF, DOCX, TXT, or CSV files and generates questions covering all key concepts.

The parser handles structured output from LLMs and validates that each MCQ has exactly 4 options and one correct answer.

---

### plagiarism_detector.py (284 lines)

A rule-based plagiarism and AI content detection system with two analysis modes:

**Plagiarism Detection** scores text across four dimensions:
- Formal/professional writing patterns (definition phrases, encyclopedic language)
- Publication markers (chapter, references, bibliography, ISBN)
- Academic citations (numbered references, year citations)
- Academic style (complex sentence structure, formal transition words)

**AI Content Detection** identifies AI-generated text by looking for:
- ChatGPT-specific phrases ("it is important to note", "delve into", "comprehensive guide")
- Overuse of formal transition words
- Uniform sentence length (low variance, characteristic of LLM output)
- Formal vocabulary density

---

### security.py (485 lines)

A comprehensive security module providing:

- **InputValidator:** Sanitizes strings, validates emails and passwords, detects SQL injection patterns, XSS patterns, and command injection patterns using regex.
- **EncryptionService:** Singleton Fernet-based symmetric encryption for sensitive data at rest.
- **RateLimiter:** In-memory rate limiter tracking login attempts per IP address with configurable lockout windows.
- **SecurityHeaders:** Returns a complete set of HTTP security headers including CSP, HSTS, X-Frame-Options, and Permissions-Policy.

---

### demo.py (452 lines)

A Flask Blueprint providing a public-facing OCR demonstration page with three API endpoints:

- `POST /api/ocr` - Single image OCR (handwritten or printed)
- `POST /api/ocr/batch` - Batch processing of multiple images
- `POST /api/ocr/analyze` - Advanced analysis with language detection and statistics

Falls back to demo mode with sample text when OCR credentials are not configured.

---

## Database Schema

The system uses 20+ PostgreSQL tables. Key relationships are described below:

```
users
  |-- id, username, email, password_hash, full_name
  |-- role (Executive | Guide | Learner)
  |-- organization_name, organization_logo, profile_photo
  |-- google_id, is_profile_complete, is_online
  |-- session_id, last_activity (session management)

courses
  |-- id, course_name, executive_id (FK users), guide_id (FK users)
  |-- organization, created_at

course_learners
  |-- course_id (FK courses), learner_id (FK users)

assignments
  |-- course_id, guide_id, name, instructions, deadline
  |-- total_marks, evaluation_model (gemini|deepseek|groq)
  |-- scores_visible

assignment_questions
  |-- assignment_id, question_text, marks, order_num

assignment_submissions
  |-- assignment_id, learner_id, total_score, is_flagged, flag_reason

assignment_answers
  |-- submission_id, question_id, image_paths (JSON)
  |-- extracted_text, relevance_score, grammar_score
  |-- size_score, uniqueness_score, total_score, feedback
  |-- covered_points (JSON), missing_points (JSON)

test_mcqs
  |-- guide_id, course_id, title, source_type (topic|document)
  |-- num_questions, marks_per_question, total_marks
  |-- time_limit_minutes, difficulty, mcqs_data (JSON)

test_mcq_attempts
  |-- test_mcq_id, learner_id, answers (JSON)
  |-- score, marks_obtained, correct_answers

document_qa
  |-- user_id, course_id, filename, extracted_text
  |-- summary, topics (JSON)

document_qa_chunks
  |-- document_id, chunk_text, chunk_index, embedding (JSON)

document_qa_chat
  |-- session_id, document_id, user_message, bot_response

diagram_assignments
  |-- course_id, topic, description, created_by

diagram_submissions
  |-- assignment_id, learner_id, diagram_path
  |-- topic_match_score, accuracy_score, structure_score
  |-- visual_score, technical_score, total_score
  |-- is_hand_drawn, drawing_type

plagiarism_matches
  |-- answer_id, matched_answer_id, similarity_score
  |-- status (pending|ignored|penalized), penalty_marks

messages
  |-- sender_id, receiver_id, content, is_read

notifications
  |-- user_id, title, message, notification_type, is_read

otps
  |-- user_id, otp_code, purpose, expires_at, is_used

custom_scoreboard_columns
  |-- course_id, name, total_marks

custom_scoreboard_column_values
  |-- column_id, learner_id, mark
```

---

## User Roles and Workflows

### Executive Workflow

```
Register (with organization details and logo)
        |
        v
Executive Dashboard
        |
        +-- Create Course
        |       |-- Specify course name
        |       |-- Assign Guide (creates account, sends credentials via email)
        |       |-- Add Learners (creates accounts, sends credentials via email)
        |
        +-- Manage Existing Courses
        |       |-- View enrolled learners
        |       |-- Add more learners
        |       |-- Remove users (OTP-verified deletion)
        |
        +-- Post Announcements (with optional file attachments)
        |
        +-- Analytics
        |       |-- Learner enrollment trends (by date)
        |       |-- Course statistics
        |
        +-- Contact Support (urgent issues, complaints, suggestions)
```

### Guide Workflow

```
Login (credentials emailed by Executive)
        |
        v
Guide Dashboard
        |
        +-- Upload Course Materials
        |       |-- PDF, DOCX, XLS, CSV, images, videos
        |       |-- Materials visible to enrolled learners
        |
        +-- Create Assignments
        |       |-- Add questions with individual marks
        |       |-- Set deadline and total marks
        |       |-- Choose AI evaluation model (Gemini/Groq/DeepSeek)
        |
        +-- Review Submissions
        |       |-- View OCR-extracted text from handwritten images
        |       |-- See AI evaluation scores per question
        |       |-- Review plagiarism flags
        |       |-- Override scores manually
        |       |-- Toggle score visibility for learners
        |
        +-- Scoreboard Management
        |       |-- View all learner scores
        |       |-- Add custom columns (attendance, participation, etc.)
        |       |-- Export scoreboard as PDF or CSV
        |
        +-- MCQ Test Creation
        |       |-- Generate from topic (with difficulty settings)
        |       |-- Generate from uploaded document
        |       |-- Set time limits and marks per question
        |       |-- Assign to courses
        |
        +-- Diagram Assignments
        |       |-- Create diagram topics
        |       |-- Review AI-evaluated diagram submissions
        |
        +-- Plagiarism Management
        |       |-- View similarity matches between submissions
        |       |-- Ignore or penalize flagged submissions
        |
        +-- AI Chatbot (GuideBot)
                |-- Ask questions about course content
                |-- Get teaching suggestions
```

### Learner Workflow

```
Login (credentials emailed by Executive)
        |
        v
Complete Profile (name, photo, bio)
        |
        v
Learner Dashboard
        |
        +-- Access Course Materials
        |       |-- Download files uploaded by Guide
        |       |-- Mark materials as read
        |
        +-- Submit Assignments
        |       |-- Upload handwritten answer images per question
        |       |-- System extracts text via Google Vision OCR
        |       |-- AI evaluates and scores automatically
        |       |-- View scores when Guide makes them visible
        |
        +-- Take MCQ Tests
        |       |-- Timed tests with countdown timer
        |       |-- Immediate scoring and feedback
        |       |-- View correct answers after submission
        |
        +-- Document Q&A
        |       |-- Upload any document (PDF, DOCX, TXT)
        |       |-- Ask natural language questions
        |       |-- RAG-powered answers with source context
        |
        +-- AI Chatbot (LearnerBot)
        |       |-- Ask questions about course topics
        |       |-- Get explanations and study help
        |
        +-- Real-time Messaging
                |-- Chat with Guide and other course members
```

---

## Feature Breakdown

### Authentication and Session Management

- Email and password registration with organization details
- Google OAuth 2.0 login and registration
- OTP-based password reset via email
- Session fixation protection (session ID regeneration on login)
- Single active session enforcement (new login invalidates previous sessions)
- 30-minute inactivity timeout with automatic logout
- Role-based access control with decorators

### Assignment Evaluation Pipeline

```
Learner uploads image(s) for each question
        |
        v
Google Cloud Vision API (document_text_detection)
        |-- Optimized for handwritten text
        |-- Returns full_text_annotation
        |-- Fallback to text_annotations if needed
        |
        v
Name Mismatch Check
        |-- First 3 words of extracted text checked against learner name
        |-- Flags if name not found (potential identity fraud)
        |
        v
AI Evaluation (Gemini / Groq / DeepSeek)
        |-- Relevance Score (0-50)
        |-- Grammar Score (0-10)
        |-- Uniqueness Score (0-10)
        |-- Covered points list
        |-- Missing points list
        |-- Constructive feedback text
        |
        v
Size Score Calculation
        |-- Word count vs expected range (50-200 words)
        |-- Penalty for very short or excessively long answers
        |
        v
Plagiarism Detection
        |-- Compare with all other submissions for same question
        |-- Flag if similarity >= 70%
        |-- Store match records with similarity scores
        |
        v
MCQ Generation (per answer)
        |-- Question MCQ: tests understanding of the topic
        |-- Answer MCQ: tests if learner understands what they wrote
        |
        v
Final Score = (Relevance + Grammar + Size + Uniqueness) / 100 * marks_per_question
```

### Real-time Messaging System

Built on Flask-SocketIO with threading async mode:

- Room-based isolation (each conversation is a private room)
- Online/offline status tracking
- Message read receipts
- Persistent message history in PostgreSQL
- Supports Executive-Guide, Executive-Learner, and Guide-Learner conversations

### Document Question Answering (RAG System)

```
Document Upload (PDF/DOCX/TXT/PPTX)
        |
        v
Text Extraction
        |
        v
Chunking (overlapping chunks for context preservation)
        |
        v
Embedding Generation (Google Gemini embedding-001)
        |
        v
FAISS Vector Index Storage
        |
        v
User Question
        |
        v
Question Embedding
        |
        v
Similarity Search (top-k chunks)
        |
        v
Context Assembly
        |
        v
Gemini 2.5 Flash Answer Generation
        |
        v
Response with Source Context
```

### MCQ Test System

- Tests can be created from a topic (with difficulty, subcategory, option difficulty settings) or from an uploaded document
- Configurable time limits: 20, 30, 60, 120, 180, 240, or 300 minutes
- Marks per question configurable independently
- Learner attempts tracked with start and completion timestamps
- Immediate scoring with per-question feedback
- Results stored for guide review

### Diagram Evaluation

Guides create diagram assignments with a topic. Learners upload images of hand-drawn or digital diagrams. The AI evaluates across five dimensions:

- Topic Match Score
- Accuracy Score
- Structure Score
- Visual Quality Score
- Technical Correctness Score

The system also detects whether the diagram is hand-drawn or digital.

### Scoreboard and Reporting

- Live scoreboard per course showing all assignment scores
- Custom columns for non-assignment scores (attendance, participation, viva, etc.)
- PDF export using ReportLab with formatted tables
- CSV export for spreadsheet analysis
- Per-submission detailed PDF reports with OCR text, AI feedback, and plagiarism status

---

## API Reference

### Authentication

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET/POST | `/register` | User registration |
| GET/POST | `/login` | User login |
| GET | `/auth/google` | Initiate Google OAuth |
| GET | `/auth/google/callback` | Google OAuth callback |
| GET/POST | `/complete_profile` | Profile completion |
| POST | `/forgot-password/send-otp` | Send OTP to email |
| POST | `/forgot-password/verify-otp` | Verify OTP |
| POST | `/forgot-password/reset` | Reset password |
| GET | `/logout` | Logout |

### Executive APIs

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/executive/dashboard` | Executive dashboard |
| POST | `/executive/create_course` | Create a new course |
| POST | `/executive/add_learners` | Add learners to course |
| GET | `/api/course/<id>/details` | Get course details |
| POST | `/executive/delete_user/<id>` | Delete user (OTP required) |
| POST | `/executive/announcements` | Post announcement |
| GET | `/api/executive/learners-by-date` | Enrollment analytics |
| POST | `/api/executive/contact` | Submit contact/support request |

### Guide APIs

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/guide/dashboard` | Guide dashboard |
| POST | `/guide/upload` | Upload course material |
| POST | `/guide/assignments/create` | Create assignment |
| GET | `/guide/assignments/list` | List assignments |
| GET | `/guide/assignments/<id>/submissions` | View submissions |
| POST | `/guide/assignments/answers/<id>/score` | Override score |
| GET | `/guide/assignments/<id>/scores` | Get all scores |
| POST | `/guide/assignments/<id>/visibility` | Toggle score visibility |
| GET | `/guide/assignments/<id>/reports/pdf` | Export PDF report |
| GET | `/guide/assignments/<id>/scores/csv` | Export CSV scores |
| GET | `/guide/scoreboard/course/<id>/data` | Get scoreboard data |
| POST | `/guide/scoreboard/columns` | Add custom column |
| POST | `/guide/scoreboard/columns/<id>/value` | Set column value |
| GET | `/guide/scoreboard/course/<id>/pdf` | Export scoreboard PDF |
| POST | `/api/guide/feedback` | Submit guide feedback |
| POST | `/api/guide/suggestion` | Submit suggestion |

### Learner APIs

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/learner/dashboard` | Learner dashboard |
| GET | `/api/course/<id>/materials` | Get course materials |
| POST | `/api/material/<id>/mark-read` | Mark material as read |
| GET | `/learner/assignments/mcq/<id>` | Get MCQ for answer |
| POST | `/learner/assignments/mcq/submit` | Submit MCQ response |
| POST | `/api/learner/suggestion` | Submit suggestion |

### Messaging APIs

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/messages` | Messages page |
| GET | `/api/messages/<user_id>` | Get conversation |

### MCQ Test APIs

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/mcq-test` | MCQ test page |

### Demo APIs

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/demo` | Demo page |
| POST | `/api/ocr` | Single image OCR |
| POST | `/api/ocr/batch` | Batch OCR |
| POST | `/api/ocr/analyze` | Advanced OCR analysis |
| POST | `/api/contact` | Contact form |

---

## Security Implementation

### Input Validation and Sanitization

Every user input passes through `InputValidator` in `security.py`:

- SQL injection detection using 7 regex patterns covering UNION SELECT, DROP, EXEC, and boolean-based injections
- XSS detection and HTML sanitization using the `bleach` library with a restrictive allowed-tags whitelist
- Command injection detection for shell metacharacters and common Unix commands
- String length limits enforced at the application layer

### CSRF Protection

Flask-WTF CSRFProtect is applied globally. CSRF tokens are injected into all Jinja2 templates via a context processor and accepted in both form submissions and JSON request headers (`X-CSRFToken`, `X-CSRF-Token`).

### Rate Limiting

Flask-Limiter applies rate limits at the route level:
- Login: 5 attempts per minute
- Registration: 10 per minute
- Course creation: 10 per minute
- Learner addition: 20 per minute

A custom in-memory `RateLimiter` class in `security.py` provides additional per-IP login attempt tracking with 15-minute lockout after 5 failures.

### Session Security

- Session cookies are HttpOnly (no JavaScript access) and SameSite=Lax (CSRF protection)
- Session fixation prevention: session is cleared and regenerated on every login
- Single active session: the current session ID is stored in the database; any new login invalidates the previous session
- 30-minute inactivity timeout enforced server-side via `last_activity` timestamp

### Security Headers

Applied to all non-static responses:

```
X-Content-Type-Options: nosniff
X-Frame-Options: DENY
X-XSS-Protection: 1; mode=block
Strict-Transport-Security: max-age=31536000; includeSubDomains
Content-Security-Policy: (comprehensive policy allowing required APIs)
Referrer-Policy: strict-origin-when-cross-origin
Permissions-Policy: geolocation=(), microphone=(), camera=()
```

### Password Security

- Passwords hashed with PBKDF2-SHA256 via Werkzeug
- Minimum 8 characters, requires uppercase, lowercase, and digit
- Maximum 128 characters to prevent denial-of-service via hash computation

---

## AI Integration Details

### Google Gemini 2.5 Flash

Used for:
- Assignment answer evaluation (relevance, grammar, uniqueness scoring)
- MCQ generation from topics and documents
- Document Q&A (RAG-based answers)
- Diagram evaluation
- Question and answer MCQ generation

Configured via `GEMINI_API_KEY` environment variable. The `google-generativeai` and `google-genai` packages are both used for different features.

### Groq (LLaMA 3.3-70B Versatile)

Used as a fallback or alternative for:
- Assignment evaluation
- MCQ generation
- AI chatbot responses

Configured via `GROQ_API_KEY`. Uses the OpenAI-compatible API endpoint at `https://api.groq.com/openai/v1`.

### DeepSeek Chat

Used as a third evaluation option:
- Assignment evaluation
- MCQ generation

Configured via `DEEPSEEK_API_KEY`. Uses the OpenAI-compatible API at `https://api.deepseek.com`.

### Google Cloud Vision

Used for OCR of handwritten assignment submissions. The `document_text_detection` method is used (optimized for handwritten and dense text) rather than `text_detection`. Credentials are loaded from `ocr.json` locally or from the `GOOGLE_APPLICATION_CREDENTIALS_JSON` environment variable in production.

### Whisper (Optional)

OpenAI Whisper is imported conditionally. The model loading is disabled by default to reduce memory usage in production. The Groq Whisper API endpoint can be used as a production alternative. Supports 60+ languages via ISO 639-1 codes.

---

## Local Development Setup

### Prerequisites

- Python 3.12
- PostgreSQL 14+ running locally
- A Google Cloud project with Vision API enabled
- API keys for Gemini, Groq, and DeepSeek

### Step 1: Clone and Create Virtual Environment

```bash
git clone <your-repo-url>
cd "AI LMS"
python3.12 -m venv venv
source venv/bin/activate
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Set Up PostgreSQL

```bash
psql -U postgres
CREATE DATABASE automate;
\q
```

### Step 4: Configure Environment Variables

```bash
cp .env.example .env
# Edit .env with your actual values
```

### Step 5: Add OCR Credentials

Place your Google Cloud service account JSON file as `ocr.json` in the project root.

### Step 6: Run the Application

```bash
python app.py
```

The application will be available at `http://127.0.0.1:5000`. The database tables are created automatically on first run.

---

## Environment Variables

Copy `.env.example` to `.env` and fill in all values:

```env
# AI API Keys
GROQ_API_KEY=your_groq_api_key
DEEPSEEK_API_KEY=your_deepseek_api_key
GEMINI_API_KEY=your_gemini_api_key

# Database
DATABASE_URL=postgresql://postgres:yourpassword@localhost:5432/automate

# Session Security
SESSION_SECRET=your_random_32_character_secret_key

# Email (Gmail SMTP)
SMTP_EMAIL=your_gmail_address@gmail.com
SMTP_PASSWORD=your_gmail_app_password

# Google OAuth
GOOGLE_CLIENT_ID=your_google_client_id
GOOGLE_CLIENT_SECRET=your_google_client_secret

# GCP Storage (production)
GCP_BUCKET_NAME=your_gcs_bucket_name

# OCR (production - paste entire ocr.json content as single line)
GOOGLE_APPLICATION_CREDENTIALS_JSON={"type":"service_account",...}
```

**Note:** The Gmail SMTP password must be an App Password, not your regular Gmail password. Enable 2FA on your Google account and generate an App Password at https://myaccount.google.com/apppasswords.

---

## GCP Deployment Guide

### Phase 1: GCP Project Setup

1. Go to https://console.cloud.google.com/
2. Create a new project (note the Project ID)
3. Enable the following APIs:
   - Cloud SQL Admin API
   - Cloud Storage API
   - Secret Manager API
   - Cloud Run API
   - Cloud Build API
   - Cloud Vision API
   - Compute Engine API

### Phase 2: Create Cloud SQL Instance

1. Navigate to SQL in the GCP Console
2. Create a PostgreSQL 15 instance
   - Instance ID: `learning-platform-db`
   - Region: `asia-south1` (Mumbai)
   - Machine type: `db-f1-micro` for testing, `db-custom-2-8192` for production
   - Enable Public IP and add `0.0.0.0/0` network temporarily
3. Create a database named `automate`
4. Note the instance connection name and public IP

### Phase 3: Create Cloud Storage Bucket

1. Navigate to Cloud Storage
2. Create a bucket named `capstone-learning-uploads` (or your preferred name)
3. Region: `asia-south1`
4. Storage class: Standard
5. Access control: Uniform

### Phase 4: Configure Secret Manager

Create the following secrets in Secret Manager:

| Secret Name | Value |
|-------------|-------|
| `GROQ_API_KEY` | Your Groq API key |
| `DEEPSEEK_API_KEY` | Your DeepSeek API key |
| `GEMINI_API_KEY` | Your Gemini API key |
| `SESSION_SECRET` | Random 32+ character string |
| `SMTP_EMAIL` | Gmail address |
| `SMTP_PASSWORD` | Gmail App Password |
| `GOOGLE_CLIENT_ID` | OAuth client ID |
| `GOOGLE_CLIENT_SECRET` | OAuth client secret |
| `DATABASE_URL` | `postgresql://postgres:PASSWORD@PUBLIC_IP:5432/automate` |
| `GCP_BUCKET_NAME` | Your bucket name |
| `GOOGLE_APPLICATION_CREDENTIALS_JSON` | Entire content of ocr.json |

### Phase 5: Deploy via Cloud Shell

Open Cloud Shell in the GCP Console and run:

```bash
# Upload your project (or clone from GitHub)
git clone YOUR_GITHUB_REPO_URL
cd YOUR_REPO_NAME

# Set project ID
export PROJECT_ID=your-project-id

# Build container
gcloud builds submit --tag gcr.io/$PROJECT_ID/learning-platform

# Deploy to Cloud Run
gcloud run deploy learning-platform \
  --image gcr.io/$PROJECT_ID/learning-platform \
  --platform managed \
  --region asia-south1 \
  --allow-unauthenticated \
  --memory 4Gi \
  --cpu 2 \
  --timeout 300 \
  --max-instances 10 \
  --set-env-vars="PORT=8080" \
  --set-secrets="GROQ_API_KEY=GROQ_API_KEY:latest,\
DEEPSEEK_API_KEY=DEEPSEEK_API_KEY:latest,\
GEMINI_API_KEY=GEMINI_API_KEY:latest,\
SESSION_SECRET=SESSION_SECRET:latest,\
SMTP_EMAIL=SMTP_EMAIL:latest,\
SMTP_PASSWORD=SMTP_PASSWORD:latest,\
GOOGLE_CLIENT_ID=GOOGLE_CLIENT_ID:latest,\
GOOGLE_CLIENT_SECRET=GOOGLE_CLIENT_SECRET:latest,\
DATABASE_URL=DATABASE_URL:latest,\
GCP_BUCKET_NAME=GCP_BUCKET_NAME:latest,\
GOOGLE_APPLICATION_CREDENTIALS_JSON=GOOGLE_APPLICATION_CREDENTIALS_JSON:latest"
```

---

## Final Deployment Steps

### What Has Been Completed

1. Security cleanup (removed hardcoded secrets)
2. Created GCP Project
3. Enabled 7 APIs
4. Created Cloud SQL Database (2 vCPU, 8 GB)
5. Created Database named `automate`
6. Created Cloud Storage Bucket
7. Created 11 Secrets in Secret Manager
8. Created Dockerfile
9. Created deployment script

### Secrets Checklist

Verify all 11 secrets exist in Secret Manager before deploying:

1. `GROQ_API_KEY`
2. `DEEPSEEK_API_KEY`
3. `GEMINI_API_KEY`
4. `SESSION_SECRET`
5. `SMTP_EMAIL`
6. `SMTP_PASSWORD`
7. `GOOGLE_CLIENT_ID`
8. `GOOGLE_CLIENT_SECRET`
9. `DATABASE_URL` = `postgresql://postgres:MySecurePass2026!@35.200.207.252:5432/automate`
10. `GCP_BUCKET_NAME` = `capstone-learning-uploads`
11. `GOOGLE_APPLICATION_CREDENTIALS_JSON` = (entire ocr.json content)

### Deployment Steps

**Step 1: Open Cloud Shell**

Go to https://console.cloud.google.com/ and click the Cloud Shell icon (top right corner).

**Step 2: Upload Your Code**

```bash
mkdir learning-platform
cd learning-platform
# Upload via Cloud Shell upload button, or:
git clone YOUR_GITHUB_REPO_URL
cd YOUR_REPO_NAME
```

**Step 3: Make Deploy Script Executable**

```bash
chmod +x deploy.sh
```

**Step 4: Run Deployment**

```bash
./deploy.sh
```

This will build the Docker container, deploy to Cloud Run, configure all secrets, and output the app URL. Expected time: 10-15 minutes.

**Step 5: Get Your App URL**

After deployment completes:
```
Service URL: https://learning-platform-xxxxx-uc.a.run.app
```

### Post-Deployment Testing Checklist

- Can you access the homepage?
- Can you register a new user?
- Can you login?
- Can you create a course?
- Can you upload files?
- Do email notifications work?
- Does Google OAuth work?
- Can learners submit assignments?
- Does OCR extract text correctly?

### Troubleshooting

**If deployment fails:**
```bash
gcloud run logs read learning-platform --region asia-south1
```

**Common issues:**
- Missing secrets: Add them in Secret Manager
- Database connection failure: Verify DATABASE_URL secret and Cloud SQL instance status
- Build errors: Check Dockerfile and requirements.txt

**If the app crashes:**
- Go to Cloud Run in GCP Console
- Click your service name
- Click the LOGS tab
- Look for Python tracebacks

---

## Cost Monitoring

### Expected Costs for 7-Day Testing Period

| Service | Configuration | Estimated Cost (INR) |
|---------|--------------|---------------------|
| Cloud SQL | 2 vCPU, 8 GB RAM | 2,800 - 3,500 |
| Cloud Run | 4 GB RAM, 2 CPU | 1,000 - 2,000 |
| Cloud Storage | Standard, Mumbai | 50 |
| Cloud Vision API | OCR calls | 100 - 200 |
| Secret Manager | 11 secrets | Free |
| **Total** | | **3,950 - 5,750** |

### Monitoring Spending

1. Go to https://console.cloud.google.com/billing
2. Click your billing account
3. View the Reports tab

### To Stop All Costs

1. Delete the Cloud SQL instance (largest cost driver)
2. Delete the Cloud Run service
3. Delete the Cloud Storage bucket

### Custom Domain (Optional)

1. Purchase a domain from Google Domains or Namecheap (500-1500 INR/year)
2. In Cloud Run, click your service
3. Click "Manage Custom Domains"
4. Follow the verification wizard

---

## Future Scope

### Short-Term Enhancements (3-6 months)

**Video Lecture Integration**
Add support for video uploads with automatic transcription using the Groq Whisper API. Generate chapter markers and searchable transcripts. Allow learners to ask questions about specific video segments.

**Advanced Analytics Dashboard**
Build an executive-level analytics dashboard with learning velocity metrics, assignment completion rates, score distribution charts, and learner engagement heatmaps using Chart.js or D3.js.

**Mobile Application**
Develop a React Native or Flutter mobile application using the existing REST API. Add push notifications for assignment deadlines, new materials, and messages.

**Peer Review System**
Allow learners to anonymously review each other's submissions before the guide's final evaluation. Implement a calibration mechanism to weight peer scores.

**Adaptive Learning Paths**
Based on MCQ test performance and assignment scores, automatically recommend additional study materials or remedial content to learners who are struggling.

### Medium-Term Enhancements (6-12 months)

**Live Virtual Classroom**
Integrate WebRTC for live video sessions between guides and learners. Add screen sharing, virtual whiteboard, and session recording with automatic transcription.

**Gamification Layer**
Add badges, leaderboards, streaks, and achievement systems to increase learner engagement. Implement a points economy where learners earn points for completing materials, submitting assignments on time, and achieving high scores.

**Multi-Language Support**
Extend the UI to support multiple languages using Flask-Babel. The OCR system already supports 60+ languages; extend the AI evaluation prompts to evaluate answers in the learner's native language.

**Advanced Plagiarism Detection**
Integrate with external plagiarism databases (Turnitin API or Copyleaks) for internet-wide plagiarism checking. Add code plagiarism detection for programming assignments using AST comparison.

**LTI Integration**
Implement the Learning Tools Interoperability (LTI) standard to allow this platform to be embedded within existing LMS platforms like Moodle or Canvas, or to embed external tools.

**Automated Curriculum Generation**
Given a learning objective and target audience, use AI to automatically generate a complete course curriculum, assignment questions, MCQ banks, and recommended reading lists.

### Long-Term Vision (12+ months)

**Multi-Tenant SaaS Architecture**
Refactor the application into a true multi-tenant SaaS platform with isolated database schemas per organization, custom branding per tenant, and a subscription billing system.

**AI Tutor Agent**
Build a persistent AI tutor agent that tracks each learner's knowledge state over time, identifies gaps, and proactively suggests study sessions. The agent would maintain a knowledge graph of the curriculum and map each learner's progress onto it.

**Proctoring System**
Add AI-powered exam proctoring using webcam feeds, tab-switch detection, and keystroke dynamics analysis to ensure academic integrity during online tests.

**Content Marketplace**
Allow organizations to publish and sell course content to other organizations on the platform. Implement a revenue-sharing model and content quality rating system.

**Offline-First Mobile App**
Build a mobile application that works offline, syncing content when connectivity is available. This is particularly valuable for learners in regions with unreliable internet access.

**Blockchain Certificates**
Issue tamper-proof course completion certificates as NFTs or on a permissioned blockchain. Allow employers to verify certificates without contacting the issuing organization.

**Integration Ecosystem**
Build integrations with popular enterprise tools: Slack/Teams for notifications, Google Workspace for document collaboration, Zoom for scheduling, and HR systems for automatic learner enrollment based on employee onboarding.

---

## License

Copyright © 2026 **Suyash Vishwas Jadhav**. All rights reserved.

This software and its source code are the exclusive intellectual property of Suyash Vishwas Jadhav. Unauthorized copying, reproduction, modification, distribution, or use of this software, in whole or in part, without the express written permission of the copyright holder is strictly prohibited.

For licensing inquiries, contact the author directly.

---

*Built with Flask, PostgreSQL, Google Cloud, and multiple AI providers. Designed for enterprise-grade educational management.*

*© 2026 Suyash Vishwas Jadhav — Lead Developer, Designer, Architect & Creator*
