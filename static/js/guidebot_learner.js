// ==================== GUIDEBOT FOR LEARNER DASHBOARD ====================

class LearnerGuidebot {
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
            this.addMessage(response.text, 'bot', response.actions, response.showForm);
        }, 800 + Math.random() * 500);
    }
    
    handleOptionClick(option, buttonElement) {
        const optionText = buttonElement.querySelector('span').textContent;
        this.addMessage(`I want to know: ${optionText}`, 'user');
        
        this.showTyping();
        
        setTimeout(() => {
            this.hideTyping();
            const response = this.getOptionResponse(option);
            this.addMessage(response.text, 'bot', response.actions, response.showForm, false, true);
        }, 600);
    }
    
    getOptionResponse(option) {
        const responses = {
            'upload-assignment': {
                text: `**How to Upload Assignment:**\n\n1. Go to the **"Assignments"** tab in the sidebar\n2. Select an assignment from the list\n3. Read each question carefully\n4. For each question:\n   • Click the upload area\n   • Select images of your handwritten answers\n   • You can upload multiple images per question\n5. Review your uploaded images\n6. Click **"Submit Assignment"**\n\n📝 **Tips:**\n• Make sure your handwriting is clear\n• Ensure all questions are answered\n• Images should be well-lit and readable\n• The AI will automatically grade your answers\n\n✨ Your assignment will be evaluated automatically!`,
                actions: []
            },
            'chat-guide': {
                text: `**Chat with Guide:**\n\nI can help you connect with your course guides! Click the button below to open the messages page. 💬`,
                actions: [
                    { text: 'Open Messages', query: 'open-messages' }
                ]
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
        
        if (quickActions && quickActions.length > 0) {
            const actionsDiv = document.createElement('div');
            actionsDiv.className = 'quick-actions';
            quickActions.forEach(action => {
                const btn = document.createElement('button');
                btn.className = 'quick-action-btn';
                btn.textContent = action.text;
                btn.addEventListener('click', () => {
                    if (action.query === 'open-messages') {
                        window.location.href = '/messages';
                    } else {
                        this.input.value = action.query;
                        this.sendMessage();
                    }
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
        
        // Learner options organized by category
        const categories = {
            'Know': [
                { option: 'upload-assignment', icon: 'fas fa-upload', text: 'How to Upload Assignment' }
            ],
            'Contact & Feedback': [
                { option: 'chat-guide', icon: 'fas fa-comments', text: 'Chat with Guide' },
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
    
    async submitSuggestion() {
        const suggestionText = document.getElementById('guidebot-suggestion-text').value.trim();
        
        if (!suggestionText) {
            alert('Please enter your suggestion');
            return;
        }
        
        try {
            const response = await fetch('/api/learner/suggestion', {
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
        const lowerMessage = message.toLowerCase();
        
        if (lowerMessage.includes('open-messages') || lowerMessage.includes('open messages') || lowerMessage.includes('chat guide')) {
            window.location.href = '/messages';
            return {
                text: `Taking you to the Messages page! 💬`,
                actions: []
            };
        }
        
        return {
            text: `I'm here to help! Please select an option from the menu above, or ask me a question. 😊`,
            actions: []
        };
    }
}

// Initialize Guidebot when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    new LearnerGuidebot();
});

