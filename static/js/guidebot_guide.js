// ==================== GUIDEBOT FOR GUIDE DASHBOARD ====================

class GuideGuidebot {
    constructor() {
        this.toggleBtn = document.getElementById('guidebot-toggle');
        this.window = document.getElementById('guidebot-window');
        this.closeBtn = document.getElementById('guidebot-close');
        this.messagesContainer = document.getElementById('guidebot-messages');
        this.input = document.getElementById('guidebot-input');
        this.sendBtn = document.getElementById('guidebot-send');
        this.isOpen = false;
        
        this.init();
    }
    
    init() {
        // Toggle chatbot
        this.toggleBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            this.toggle();
        });
        this.closeBtn.addEventListener('click', () => this.close());
        
        // Close when clicking outside
        document.addEventListener('click', (e) => {
            if (this.isOpen && !this.window.contains(e.target) && !this.toggleBtn.contains(e.target)) {
                this.close();
            }
        });
        
        // Prevent closing when clicking inside the window
        this.window.addEventListener('click', (e) => {
            e.stopPropagation();
        });
        
        // Prevent closing when clicking toggle button
        this.toggleBtn.addEventListener('click', (e) => {
            e.stopPropagation();
        });
        
        // Send message
        this.sendBtn.addEventListener('click', () => this.sendMessage());
        this.input.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });
        
        // Handle option button clicks
        document.querySelectorAll('.guidebot-option-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const option = e.currentTarget.dataset.option;
                this.handleOptionClick(option, e.currentTarget);
            });
        });
    }
    
    toggle() {
        this.isOpen = !this.isOpen;
        if (this.isOpen) {
            this.open();
        } else {
            this.close();
        }
    }
    
    open() {
        this.window.classList.add('active');
        this.input.focus();
        this.scrollToBottom();
        const badge = document.querySelector('.guidebot-badge');
        if (badge) badge.style.display = 'none';
    }
    
    close() {
        this.window.classList.remove('active');
        this.isOpen = false;
    }
    
    sendMessage() {
        const message = this.input.value.trim();
        if (!message) return;
        
        this.addMessage(message, 'user');
        this.input.value = '';
        
        this.showTyping();
        
        setTimeout(() => {
            this.hideTyping();
            const response = this.processMessage(message);
            this.addMessage(response.text, 'bot', response.actions, response.showForm, response.showRating);
        }, 800 + Math.random() * 500);
    }
    
    handleOptionClick(option, buttonElement) {
        const optionText = buttonElement.querySelector('span').textContent;
        this.addMessage(`I want to know: ${optionText}`, 'user');
        
        this.showTyping();
        
        setTimeout(() => {
            this.hideTyping();
            const response = this.getOptionResponse(option);
            this.addMessage(response.text, 'bot', response.actions, response.showForm, response.showRating, true);
        }, 600);
    }
    
    getOptionResponse(option) {
        const responses = {
            'upload-materials': {
                text: `**How to Upload Materials:**\n\n1. Go to the **"Uploads"** tab in the sidebar\n2. Select your course from the dropdown\n3. Click **"Choose File"** or drag and drop your file\n4. Supported formats: PDF, DOC, DOCX, PPT, PPTX, images\n5. Click **"Upload"** to add the material\n\n📚 Materials will be available to all learners in that course!`,
                actions: []
            },
            'view-assignments': {
                text: `**How to View Assignments:**\n\n1. Click on **"View Assignments"** in the sidebar\n2. Select a course from the dropdown\n3. You'll see all assignments for that course\n4. Click on any assignment to view:\n   • All submissions from learners\n   • Individual scores and feedback\n   • Plagiarism detection results\n   • MCQ results (if applicable)\n\n📊 You can also download reports and view detailed analytics!`,
                actions: []
            },
            'ai-grading': {
                text: `**How AI Grading Works:**\n\n🤖 **AI-Powered Evaluation:**\n\n• Uses advanced AI models (Gemini/Groq) to evaluate answers\n• Analyzes content quality, completeness, and accuracy\n• Provides detailed feedback on each answer\n• Identifies covered points and missing information\n• Assigns scores based on rubric and answer quality\n\n**Process:**\n1. Learner submits assignment with images\n2. OCR extracts text from images\n3. AI evaluates each answer against the question\n4. Scores are calculated automatically\n5. Feedback is generated for improvement\n\n✨ The system ensures fair and consistent grading!`,
                actions: []
            },
            'create-assignment': {
                text: `**How to Create Assignments:**\n\n1. Go to **"Create Assignment"** tab\n2. Select the course\n3. Enter assignment name and instructions\n4. Set the deadline date and time\n5. Choose evaluation model (Gemini/Groq)\n6. Add questions:\n   • Click **"Add Question"**\n   • Enter question text\n   • Set marks for each question\n7. Click **"Create Assignment"**\n\n📝 Learners can submit answers by uploading images of their work!`,
                actions: []
            },
            'scores': {
                text: `**How Scores are Shown:**\n\n📊 **Score Display:**\n\n• **Total Score:** Sum of all question scores\n• **Percentage:** Score out of total marks\n• **Per Question:** Individual scores for each answer\n• **Feedback:** Detailed AI-generated feedback\n• **Covered Points:** What the learner got right\n• **Missing Points:** What could be improved\n\n**Visibility:**\n• You control when scores are visible to learners\n• Toggle **"Scores Visible"** in assignment settings\n• Learners see their scores only when enabled\n\n📈 Scores help track learner progress and performance!`,
                actions: []
            },
            'cheating-detection': {
                text: `**How Cheating is Detected:**\n\n🛡️ **Multi-Layer Detection:**\n\n1. **Name Verification:**\n   • Checks if first words match learner's name\n   • Flags mismatches automatically\n\n2. **Plagiarism Detection:**\n   • Compares answers with other submissions\n   • Uses similarity algorithms\n   • Flags high similarity matches (≥85%)\n\n3. **Exact Match Detection:**\n   • Identifies identical submissions\n   • Flags suspicious duplicate content\n\n4. **Visual Analysis:**\n   • OCR text extraction from images\n   • Pattern recognition for copied content\n\n⚠️ **Flagged submissions** are marked for review with reasons!`,
                actions: []
            },
            'plagiarism-check': {
                text: `**What Does Plagiarism Check Do:**\n\n🔍 **Plagiarism Detection System:**\n\n• **Compares Answers:** Analyzes all submissions for similarity\n• **Similarity Score:** Calculates percentage match (0-100%)\n• **Identifies Matches:** Shows which learners have similar answers\n• **Penalty System:** Allows you to apply penalty marks\n• **Status Tracking:** Mark matches as reviewed/resolved\n• **Detailed Reports:** View side-by-side comparison\n\n**Features:**\n• Real-time detection during submission\n• Historical comparison with past submissions\n• Visual indicators for high-risk matches\n• Export reports for documentation\n\n📋 Helps maintain academic integrity!`,
                actions: []
            },
            'rate-feedback': {
                text: `**Rate & Feedback Service:**\n\nPlease rate our platform and share your feedback! Your input helps us improve. ⭐`,
                actions: [],
                showRating: true
            },
            'suggestions': {
                text: `**Submit Suggestions:**\n\nWe value your ideas! Please share any suggestions to improve the platform. 💡`,
                actions: [],
                showForm: true
            }
        };
        
        return responses[option] || {
            text: `I'm not sure about that option. Please try selecting another option from the menu.`,
            actions: []
        };
    }
    
    addMessage(text, type, quickActions = null, showForm = false, showRating = false, showCategorizedOptions = false) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `guidebot-message ${type}-message`;
        
        const avatar = document.createElement('div');
        avatar.className = 'message-avatar';
        avatar.innerHTML = type === 'bot' ? '<i class="fas fa-robot"></i>' : '<i class="fas fa-user"></i>';
        
        const content = document.createElement('div');
        content.className = 'message-content';
        
        if (typeof text === 'string') {
            content.innerHTML = this.formatMessage(text);
        } else {
            content.appendChild(text);
        }
        
        messageDiv.appendChild(avatar);
        messageDiv.appendChild(content);
        
        this.messagesContainer.appendChild(messageDiv);
        this.scrollToBottom();
        
        if (showForm) {
            this.addSuggestionForm(content);
        }
        
        if (showRating) {
            this.addRatingForm(content);
        }
        
        if (quickActions && quickActions.length > 0) {
            const actionsDiv = document.createElement('div');
            actionsDiv.className = 'quick-actions';
            quickActions.forEach(action => {
                const btn = document.createElement('button');
                btn.className = 'quick-action-btn';
                btn.textContent = action.text;
                btn.addEventListener('click', () => {
                    this.input.value = action.query;
                    this.sendMessage();
                });
                actionsDiv.appendChild(btn);
            });
            content.appendChild(actionsDiv);
        }
        
        // Show categorized options after response in a separate message
        if (showCategorizedOptions && type === 'bot') {
            setTimeout(() => {
                this.showCategorizedOptions();
            }, 500);
        }
    }
    
    showCategorizedOptions() {
        const optionsDiv = document.createElement('div');
        optionsDiv.className = 'guidebot-categorized-options';
        
        // Guide options organized by category (top options first)
        const categories = {
            'Know': [
                { option: 'upload-materials', icon: 'fas fa-upload', text: 'How to Upload Materials' },
                { option: 'view-assignments', icon: 'fas fa-list', text: 'How to View Assignments' },
                { option: 'ai-grading', icon: 'fas fa-brain', text: 'How AI Grading Works' },
                { option: 'create-assignment', icon: 'fas fa-plus-circle', text: 'How to Create Assignments' },
                { option: 'scores', icon: 'fas fa-chart-line', text: 'How Scores are Shown' },
                { option: 'cheating-detection', icon: 'fas fa-shield-alt', text: 'How Cheating is Detected' },
                { option: 'plagiarism-check', icon: 'fas fa-search', text: 'What Does Plagiarism Check Do' }
            ],
            'Feedback & Suggestions': [
                { option: 'rate-feedback', icon: 'fas fa-star', text: 'Rate & Feedback Service' },
                { option: 'suggestions', icon: 'fas fa-lightbulb', text: 'Submit Suggestions' }
            ]
        };
        
        const headerP = document.createElement('p');
        headerP.style.cssText = 'margin-bottom: 16px; font-size: 13px; font-weight: 600; color: var(--text-primary); opacity: 0.8;';
        headerP.textContent = 'What else can I help you with?';
        optionsDiv.appendChild(headerP);
        
        // Create category buttons (accordion style)
        Object.keys(categories).forEach(categoryName => {
            const categoryWrapper = document.createElement('div');
            categoryWrapper.className = 'guidebot-category-wrapper';
            
            const categoryBtn = document.createElement('button');
            categoryBtn.className = 'guidebot-category-btn';
            categoryBtn.innerHTML = `
                <span>${categoryName}</span>
                <i class="fas fa-chevron-down guidebot-category-icon"></i>
            `;
            
            const optionsList = document.createElement('div');
            optionsList.className = 'guidebot-category-options';
            optionsList.style.display = 'none';
            
            categories[categoryName].forEach(item => {
                const btn = document.createElement('button');
                btn.className = 'guidebot-option-btn';
                btn.setAttribute('data-option', item.option);
                btn.innerHTML = `<i class="${item.icon}"></i><span>${item.text}</span>`;
                btn.addEventListener('click', (e) => {
                    e.stopPropagation();
                    this.handleOptionClick(item.option, btn);
                });
                optionsList.appendChild(btn);
            });
            
            categoryBtn.addEventListener('click', () => {
                const isOpen = optionsList.style.display !== 'none';
                const icon = categoryBtn.querySelector('.guidebot-category-icon');
                
                // Close all other categories
                document.querySelectorAll('.guidebot-category-options').forEach(opt => {
                    if (opt !== optionsList) {
                        opt.style.display = 'none';
                        opt.parentElement.querySelector('.guidebot-category-btn').classList.remove('active');
                    }
                });
                
                // Toggle current category
                if (isOpen) {
                    optionsList.style.display = 'none';
                    categoryBtn.classList.remove('active');
                    icon.style.transform = 'rotate(0deg)';
                } else {
                    optionsList.style.display = 'flex';
                    categoryBtn.classList.add('active');
                    icon.style.transform = 'rotate(180deg)';
                }
                
                this.scrollToBottom();
            });
            
            categoryWrapper.appendChild(categoryBtn);
            categoryWrapper.appendChild(optionsList);
            optionsDiv.appendChild(categoryWrapper);
        });
        
        // Create a separate bot message for the options
        const messageDiv = document.createElement('div');
        messageDiv.className = 'guidebot-message bot-message';
        
        const avatar = document.createElement('div');
        avatar.className = 'message-avatar';
        avatar.innerHTML = '<i class="fas fa-robot"></i>';
        
        const content = document.createElement('div');
        content.className = 'message-content';
        content.appendChild(optionsDiv);
        
        messageDiv.appendChild(avatar);
        messageDiv.appendChild(content);
        
        this.messagesContainer.appendChild(messageDiv);
        
        // Scroll to bottom to show the options
        setTimeout(() => {
            this.scrollToBottom();
        }, 100);
    }
    
    addRatingForm(container) {
        const ratingDiv = document.createElement('div');
        ratingDiv.className = 'guidebot-rating-form';
        ratingDiv.innerHTML = `
            <div style="margin-top: 16px;">
                <label style="display: block; margin-bottom: 8px; font-size: 13px; font-weight: 500; color: var(--text-primary);">
                    Rating: <span style="color: #ef4444;">*</span>
                </label>
                <div class="star-rating" style="display: flex; gap: 8px; margin-bottom: 12px;">
                    ${[1, 2, 3, 4, 5].map(i => `
                        <button class="star-btn" data-rating="${i}" style="
                            background: none;
                            border: none;
                            font-size: 32px;
                            color: #ddd;
                            cursor: pointer;
                            transition: all 0.2s;
                            padding: 0;
                        ">⭐</button>
                    `).join('')}
                </div>
                <input type="hidden" id="guidebot-rating-value" value="0">
                
                <label style="display: block; margin-bottom: 8px; font-size: 13px; font-weight: 500; color: var(--text-primary);">
                    Feedback (Optional):
                </label>
                <textarea id="guidebot-feedback-text" rows="3" placeholder="Share your thoughts about the platform..." style="
                    width: 100%;
                    padding: 10px 12px;
                    border: 1px solid rgba(102, 126, 234, 0.3);
                    border-radius: 8px;
                    background: rgba(255, 255, 255, 0.1);
                    color: var(--text-primary);
                    font-size: 14px;
                    font-family: inherit;
                    outline: none;
                    resize: vertical;
                    margin-bottom: 12px;
                "></textarea>
                
                <button class="guidebot-submit-rating" style="
                    padding: 12px 24px;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    border: none;
                    border-radius: 8px;
                    color: white;
                    font-size: 14px;
                    font-weight: 500;
                    cursor: pointer;
                    width: 100%;
                    transition: all 0.3s;
                ">
                    <i class="fas fa-star"></i> Submit Rating
                </button>
            </div>
        `;
        
        container.appendChild(ratingDiv);
        
        // Star rating interaction
        const starBtns = ratingDiv.querySelectorAll('.star-btn');
        const ratingInput = ratingDiv.querySelector('#guidebot-rating-value');
        
        starBtns.forEach((btn, index) => {
            btn.addEventListener('click', () => {
                const rating = index + 1;
                ratingInput.value = rating;
                
                starBtns.forEach((b, i) => {
                    if (i < rating) {
                        b.style.color = '#ffd700';
                        b.style.transform = 'scale(1.2)';
                    } else {
                        b.style.color = '#ddd';
                        b.style.transform = 'scale(1)';
                    }
                });
            });
            
            btn.addEventListener('mouseenter', () => {
                const hoverRating = index + 1;
                starBtns.forEach((b, i) => {
                    if (i < hoverRating) {
                        b.style.color = '#ffd700';
                    }
                });
            });
        });
        
        ratingDiv.addEventListener('mouseleave', () => {
            const currentRating = parseInt(ratingInput.value);
            starBtns.forEach((b, i) => {
                if (i < currentRating) {
                    b.style.color = '#ffd700';
                } else {
                    b.style.color = '#ddd';
                }
            });
        });
        
        const submitBtn = ratingDiv.querySelector('.guidebot-submit-rating');
        submitBtn.addEventListener('click', () => {
            this.submitRating();
        });
        
        this.scrollToBottom();
    }
    
    addSuggestionForm(container) {
        const formDiv = document.createElement('div');
        formDiv.className = 'guidebot-suggestion-form';
        formDiv.innerHTML = `
            <div style="margin-top: 16px;">
                <label style="display: block; margin-bottom: 8px; font-size: 13px; font-weight: 500; color: var(--text-primary);">
                    Your Suggestion: <span style="color: #ef4444;">*</span>
                </label>
                <textarea id="guidebot-suggestion-text" rows="4" placeholder="Please share your suggestions to improve the platform..." style="
                    width: 100%;
                    padding: 10px 12px;
                    border: 1px solid rgba(102, 126, 234, 0.3);
                    border-radius: 8px;
                    background: rgba(255, 255, 255, 0.1);
                    color: var(--text-primary);
                    font-size: 14px;
                    font-family: inherit;
                    outline: none;
                    resize: vertical;
                    margin-bottom: 12px;
                "></textarea>
                
                <button class="guidebot-submit-suggestion" style="
                    padding: 12px 24px;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    border: none;
                    border-radius: 8px;
                    color: white;
                    font-size: 14px;
                    font-weight: 500;
                    cursor: pointer;
                    width: 100%;
                    transition: all 0.3s;
                ">
                    <i class="fas fa-paper-plane"></i> Submit Suggestion
                </button>
            </div>
        `;
        
        container.appendChild(formDiv);
        
        const submitBtn = formDiv.querySelector('.guidebot-submit-suggestion');
        submitBtn.addEventListener('click', () => {
            this.submitSuggestion();
        });
        
        this.scrollToBottom();
    }
    
    async submitRating() {
        const rating = document.getElementById('guidebot-rating-value').value;
        const feedbackText = document.getElementById('guidebot-feedback-text').value.trim();
        
        if (!rating || rating === '0') {
            alert('Please select a rating');
            return;
        }
        
        try {
            const response = await fetch('/api/guide/feedback', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    rating: parseInt(rating),
                    feedback_text: feedbackText
                })
            });
            
            const data = await response.json();
            
            if (data.success) {
                this.addMessage('✅ Thank you for your feedback! Your rating has been recorded.', 'bot', null, false, false, true);
                // Clear form
                document.getElementById('guidebot-rating-value').value = '0';
                document.getElementById('guidebot-feedback-text').value = '';
                // Reset stars
                document.querySelectorAll('.star-btn').forEach(btn => {
                    btn.style.color = '#ddd';
                    btn.style.transform = 'scale(1)';
                });
            } else {
                this.addMessage(`❌ Error: ${data.message || 'Failed to submit. Please try again.'}`, 'bot', null, false, false, true);
            }
        } catch (error) {
            console.error('Error submitting rating:', error);
            this.addMessage('❌ An error occurred. Please try again later.', 'bot', null, false, false, true);
        }
    }
    
    async submitSuggestion() {
        const suggestionText = document.getElementById('guidebot-suggestion-text').value.trim();
        
        if (!suggestionText) {
            alert('Please enter your suggestion');
            return;
        }
        
        try {
            const response = await fetch('/api/guide/suggestion', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    suggestion_text: suggestionText
                })
            });
            
            const data = await response.json();
            
            if (data.success) {
                this.addMessage('✅ Thank you for your suggestion! We appreciate your input and will review it.', 'bot', null, false, false, true);
                document.getElementById('guidebot-suggestion-text').value = '';
            } else {
                this.addMessage(`❌ Error: ${data.message || 'Failed to submit. Please try again.'}`, 'bot', null, false, false, true);
            }
        } catch (error) {
            console.error('Error submitting suggestion:', error);
            this.addMessage('❌ An error occurred. Please try again later.', 'bot', null, false, false, true);
        }
    }
    
    formatMessage(text) {
        text = text.replace(/\n/g, '<br>');
        text = text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
        text = text.replace(/\*(.*?)\*/g, '<em>$1</em>');
        text = text.replace(/^\- (.*)$/gm, '<li>$1</li>');
        text = text.replace(/(<li>.*<\/li>)/s, '<ul class="help-list">$1</ul>');
        return text;
    }
    
    showTyping() {
        const typingDiv = document.createElement('div');
        typingDiv.className = 'guidebot-message bot-message typing-indicator';
        typingDiv.id = 'typing-indicator';
        typingDiv.innerHTML = `
            <div class="message-avatar"><i class="fas fa-robot"></i></div>
            <div class="message-content">
                <span></span>
                <span></span>
                <span></span>
            </div>
        `;
        this.messagesContainer.appendChild(typingDiv);
        this.scrollToBottom();
    }
    
    hideTyping() {
        const typing = document.getElementById('typing-indicator');
        if (typing) typing.remove();
    }
    
    scrollToBottom() {
        setTimeout(() => {
            this.messagesContainer.scrollTop = this.messagesContainer.scrollHeight;
        }, 100);
    }
    
    processMessage(message) {
        return {
            text: `I'm here to help! Please select an option from the menu above, or ask me a question. 😊`,
            actions: []
        };
    }
}

// Initialize Guidebot when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    new GuideGuidebot();
});

