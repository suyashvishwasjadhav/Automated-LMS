// ==================== GUIDEBOT CHATBOT FUNCTIONALITY ====================

class Guidebot {
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
    }
    
    handleOptionClick(option, buttonElement) {
        // Add user message showing what they clicked
        const optionText = buttonElement.querySelector('span').textContent;
        this.addMessage(`I want to know: ${optionText}`, 'user');
        
        // Show typing indicator
        this.showTyping();
        
        // Process response after delay
        setTimeout(() => {
            this.hideTyping();
            const response = this.getOptionResponse(option);
            this.addMessage(response.text, 'bot', response.actions, response.showForm, false, true);
        }, 600);
    }
    
    getOptionResponse(option) {
        const responses = {
            'create-course': {
                text: `**How to Create a Course:**\n\n1. Click on **"Create Course"** in the sidebar\n2. Enter the course name (minimum 3 characters)\n3. Provide the guide's email address\n4. Optionally add learner emails (comma-separated, max 20)\n5. Click **"Create Course"**\n\nThe guide will receive login credentials via email automatically! 🎓`,
                actions: [
                    { text: 'Open Create Course', query: 'open-create-course' }
                ]
            },
            'delete-learners': {
                text: `**How to Delete Learners:**\n\n1. Navigate to your course dashboard\n2. Click on **"View Details"** for the course\n3. Find the learner you want to remove in the learners list\n4. Click the delete/remove button next to their name\n5. Confirm the deletion when prompted\n\n⚠️ **Note:** This action cannot be undone. The learner will lose access to the course immediately.`,
                actions: []
            },
            'learner-limit': {
                text: `**Learner Limits:**\n\n• Maximum **20 learners** can be added per course\n• You can add learners when creating a course or later\n• To add more learners, click the **"+"** button on any course card\n• Learners can be added in batches of up to 20 at a time\n\n💡 **Tip:** If you need more than 20 learners, consider creating multiple courses or upgrading your plan.`,
                actions: []
            },
            'guide-limit': {
                text: `**Guide Assignment:**\n\n• **One guide** can be assigned per course\n• Each course has a single guide who manages it\n• The guide is assigned when you create the course\n• You can message your guides through the Messages section\n• Guides help manage course content, assignments, and learners\n\n👨‍🏫 Guides are essential for course management and learner support.`,
                actions: [
                    { text: 'Open Messages', query: 'open-messages' }
                ]
            },
            'learners-by-date': {
                text: `**Learners Added by Date:**\n\nI can help you find how many learners were added on a specific date! 📅\n\nPlease select a time period or choose a custom date:`,
                actions: [
                    { text: 'Today', query: 'learners-today' },
                    { text: 'Yesterday', query: 'learners-yesterday' },
                    { text: 'This Week', query: 'learners-week' },
                    { text: 'This Month', query: 'learners-month' },
                    { text: 'Select Custom Date', query: 'learners-custom-date' }
                ]
            },
            'contact-dev': {
                text: `**Contact Developer/Support:**\n\nPlease fill out the form below to contact our development team. We'll get back to you as soon as possible! 💬`,
                actions: [],
                showForm: true
            }
        };
        
        return responses[option] || {
            text: `I'm not sure about that option. Please try selecting another option from the menu.`,
            actions: []
        };
    }
    
    toggle(e) {
        if (e) e.stopPropagation();
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
        // Hide badge when opened
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
        
        // Add user message
        this.addMessage(message, 'user');
        this.input.value = '';
        
        // Show typing indicator
        this.showTyping();
        
        // Process response after delay
        setTimeout(() => {
            this.hideTyping();
            const response = this.processMessage(message);
            this.addMessage(response.text, 'bot', response.actions, response.showForm, response.showDatePicker);
        }, 800 + Math.random() * 500);
    }
    
    addMessage(text, type, quickActions = null, showForm = false, showDatePicker = false, showCategorizedOptions = false) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `guidebot-message ${type}-message`;
        
        const avatar = document.createElement('div');
        avatar.className = 'message-avatar';
        avatar.innerHTML = type === 'bot' ? '<i class="fas fa-robot"></i>' : '<i class="fas fa-user"></i>';
        
        const content = document.createElement('div');
        content.className = 'message-content';
        
        // Parse text for formatting
        if (typeof text === 'string') {
            content.innerHTML = this.formatMessage(text);
        } else {
            content.appendChild(text);
        }
        
        messageDiv.appendChild(avatar);
        messageDiv.appendChild(content);
        
        this.messagesContainer.appendChild(messageDiv);
        this.scrollToBottom();
        
        // Add contact form if needed
        if (showForm) {
            this.addContactForm(content);
        }
        
        // Add date picker if needed
        if (showDatePicker) {
            this.addDatePicker(content);
        }
        
        // Add quick actions if provided
        if (quickActions && quickActions.length > 0) {
            const actionsDiv = document.createElement('div');
            actionsDiv.className = 'quick-actions';
            quickActions.forEach(action => {
                const btn = document.createElement('button');
                btn.className = 'quick-action-btn';
                btn.textContent = action.text;
                btn.addEventListener('click', () => {
                    // Handle special actions
                    if (action.query === 'open-create-course') {
                        if (typeof switchSection === 'function') {
                            switchSection('create-course');
                            this.close();
                        }
                    } else if (action.query === 'open-messages') {
                        window.location.href = '/messages';
                    } else if (action.query === 'learners-custom-date') {
                        this.addDatePicker(content);
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
        
        // Executive options organized by category (top options first)
        const categories = {
            'Contact': [
                { option: 'contact-dev', icon: 'fas fa-envelope', text: 'Contact Developer' }
            ],
            'Know': [
                { option: 'create-course', icon: 'fas fa-plus-circle', text: 'How to Create Course' },
                { option: 'delete-learners', icon: 'fas fa-user-minus', text: 'How to Delete Learners' },
                { option: 'learner-limit', icon: 'fas fa-users', text: 'How Many Learners Can Be Added' },
                { option: 'guide-limit', icon: 'fas fa-user-tie', text: 'How Many Guides Per Course' },
                { option: 'learners-by-date', icon: 'fas fa-calendar-alt', text: 'Learners Added by Date' }
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
    
    addDatePicker(container) {
        const datePickerDiv = document.createElement('div');
        datePickerDiv.className = 'guidebot-date-picker';
        datePickerDiv.innerHTML = `
            <label style="display: block; margin: 12px 0 8px; font-size: 13px; font-weight: 500; color: var(--text-primary);">
                Select Date:
            </label>
            <input type="date" id="guidebot-custom-date" class="guidebot-date-input" style="
                width: 100%;
                padding: 10px 12px;
                border: 1px solid rgba(102, 126, 234, 0.3);
                border-radius: 8px;
                background: rgba(255, 255, 255, 0.1);
                color: var(--text-primary);
                font-size: 14px;
                outline: none;
                transition: all 0.3s;
            ">
            <button class="guidebot-submit-date" style="
                margin-top: 10px;
                padding: 10px 20px;
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
                <i class="fas fa-search"></i> Check Learners
            </button>
        `;
        
        container.appendChild(datePickerDiv);
        
        const submitBtn = datePickerDiv.querySelector('.guidebot-submit-date');
        submitBtn.addEventListener('click', () => {
            const dateInput = datePickerDiv.querySelector('#guidebot-custom-date');
            const selectedDate = dateInput.value;
            if (selectedDate) {
                this.addMessage(`Check learners for ${selectedDate}`, 'user');
                this.fetchLearnersByDate(selectedDate);
            } else {
                alert('Please select a date');
            }
        });
        
        this.scrollToBottom();
    }
    
    addContactForm(container) {
        const formDiv = document.createElement('div');
        formDiv.className = 'guidebot-contact-form';
        formDiv.innerHTML = `
            <div style="margin-top: 16px;">
                <label style="display: block; margin-bottom: 8px; font-size: 13px; font-weight: 500; color: var(--text-primary);">
                    Issue Type: <span style="color: #ef4444;">*</span>
                </label>
                <select id="guidebot-issue-type" style="
                    width: 100%;
                    padding: 10px 12px;
                    border: 1px solid rgba(102, 126, 234, 0.3);
                    border-radius: 8px;
                    background: rgba(255, 255, 255, 0.1);
                    color: var(--text-primary);
                    font-size: 14px;
                    outline: none;
                    margin-bottom: 12px;
                ">
                    <option value="">Select issue type...</option>
                    <option value="urgent_issue">🚨 Urgent Issue</option>
                    <option value="complaint">⚠️ Complaint</option>
                    <option value="suggestions">💡 Suggestions</option>
                    <option value="requirements">📋 Requirements</option>
                </select>
                
                <label style="display: block; margin-bottom: 8px; font-size: 13px; font-weight: 500; color: var(--text-primary);">
                    Details: <span style="color: #ef4444;">*</span>
                </label>
                <textarea id="guidebot-contact-details" rows="4" placeholder="Please describe your issue, complaint, suggestion, or requirement in detail..." style="
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
                
                <button class="guidebot-submit-contact" style="
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
                    <i class="fas fa-paper-plane"></i> Submit
                </button>
            </div>
        `;
        
        container.appendChild(formDiv);
        
        const submitBtn = formDiv.querySelector('.guidebot-submit-contact');
        submitBtn.addEventListener('click', () => {
            this.submitContactForm();
        });
        
        this.scrollToBottom();
    }
    
    async submitContactForm() {
        const issueType = document.getElementById('guidebot-issue-type').value;
        const details = document.getElementById('guidebot-contact-details').value.trim();
        
        if (!issueType) {
            alert('Please select an issue type');
            return;
        }
        
        if (!details) {
            alert('Please enter details');
            return;
        }
        
        try {
            const response = await fetch('/api/executive/contact', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    issue_type: issueType,
                    details: details
                })
            });
            
            const data = await response.json();
            
            if (data.success) {
                this.addMessage('✅ Your message has been submitted successfully! Our team will get back to you soon.', 'bot', null, false, false, true);
                // Clear form
                document.getElementById('guidebot-issue-type').value = '';
                document.getElementById('guidebot-contact-details').value = '';
            } else {
                this.addMessage(`❌ Error: ${data.message || 'Failed to submit. Please try again.'}`, 'bot', null, false, false, true);
            }
        } catch (error) {
            console.error('Error submitting contact form:', error);
            this.addMessage('❌ An error occurred. Please try again later.', 'bot', null, false, false, true);
        }
    }
    
    formatMessage(text) {
        // Convert line breaks
        text = text.replace(/\n/g, '<br>');
        // Convert **bold** to <strong>
        text = text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
        // Convert *italic* to <em>
        text = text.replace(/\*(.*?)\*/g, '<em>$1</em>');
        // Convert lists
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
        
        // Handle quick action queries
        if (lowerMessage.includes('open-create-course') || lowerMessage.includes('open create course')) {
            if (typeof switchSection === 'function') {
                switchSection('create-course');
                this.close();
            }
            return {
                text: `Opening the Create Course section for you! 🚀`,
                actions: []
            };
        }
        
        if (lowerMessage.includes('open-messages') || lowerMessage.includes('open messages')) {
            window.location.href = '/messages';
            return {
                text: `Taking you to the Messages page! 💬`,
                actions: []
            };
        }
        
        // Handle date queries
        if (lowerMessage.includes('learners-today') || lowerMessage.includes('learners added today')) {
            this.fetchLearnersByDate('today');
            return {
                text: `Let me check the learners added today... 📊`,
                actions: []
            };
        }
        
        if (lowerMessage.includes('learners-yesterday') || lowerMessage.includes('learners added yesterday')) {
            this.fetchLearnersByDate('yesterday');
            return {
                text: `Let me check the learners added yesterday... 📊`,
                actions: []
            };
        }
        
        if (lowerMessage.includes('learners-week') || lowerMessage.includes('learners added this week')) {
            this.fetchLearnersByDate('this week');
            return {
                text: `Let me check the learners added this week... 📊`,
                actions: []
            };
        }
        
        if (lowerMessage.includes('learners-month') || lowerMessage.includes('learners added this month')) {
            this.fetchLearnersByDate('this month');
            return {
                text: `Let me check the learners added this month... 📊`,
                actions: []
            };
        }
        
        if (lowerMessage.includes('learners-custom-date')) {
            // Date picker will be added in the response handler
            return {
                text: `Please select a date below:`,
                actions: [],
                showDatePicker: true
            };
        }
        
        // Default fallback - show options again
        return {
            text: `I'm here to help! Please select an option from the menu above, or ask me a question. 😊`,
            actions: []
        };
    }
    
    async fetchLearnersByDate(dateString) {
        try {
            // Fetch from API
            const response = await fetch(`/api/executive/learners-by-date?date=${encodeURIComponent(dateString)}`);
            const data = await response.json();
            
            if (data.success) {
                const dateLabel = dateString === 'today' ? 'today' : 
                                 dateString === 'yesterday' ? 'yesterday' :
                                 dateString.includes('week') ? 'this week' :
                                 dateString.includes('month') ? 'this month' : 
                                 data.date;
                
                let message = `📊 **Learners Added ${dateLabel}:**\n\n• Total: **${data.count}** learner${data.count !== 1 ? 's' : ''}`;
                
                if (data.courses && data.courses.length > 0) {
                    message += `\n• Across **${data.courses.length}** course${data.courses.length !== 1 ? 's' : ''}`;
                    if (data.courses.length <= 3) {
                        message += `\n• Courses: ${data.courses.map(c => c.name).join(', ')}`;
                    }
                }
                
                message += `\n\n${data.count > 0 ? 'Great progress! 🎉' : 'No learners were added during this period.'}`;
                
                this.addMessage(message, 'bot', null, false, false, true);
            } else {
                this.addMessage(`I couldn't fetch that information: ${data.message || 'Unknown error'}. Please try again or contact support.`, 'bot', null, false, false, true);
            }
        } catch (error) {
            console.error('Error fetching learners:', error);
            this.addMessage(`I encountered an error while fetching the data. Please try asking again or contact support.`, 'bot', null, false, false, true);
        }
    }
}

// Initialize Guidebot when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    new Guidebot();
});

