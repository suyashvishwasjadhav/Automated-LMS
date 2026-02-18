// ==================== INITIALIZATION ====================
const socket = io();
let currentReceiverId = null;
let currentContactName = '';
const CURRENT_USER_ID = parseInt(document.body.dataset.userId || '0');
let isTyping = false;
let typingTimeout = null;

// ==================== THEME MANAGEMENT ====================
function initTheme() {
    const savedTheme = localStorage.getItem('theme') || 'light';
    document.documentElement.setAttribute('data-theme', savedTheme);
}

function toggleTheme() {
    const currentTheme = document.documentElement.getAttribute('data-theme');
    const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
    
    document.documentElement.setAttribute('data-theme', newTheme);
    localStorage.setItem('theme', newTheme);
    
    // Trigger theme change animation
    createThemeChangeEffect();
}

function createThemeChangeEffect() {
    const overlay = document.createElement('div');
    overlay.style.cssText = `
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: radial-gradient(circle at center, rgba(102, 126, 234, 0.3), transparent);
        pointer-events: none;
        z-index: 9999;
        animation: themeFlash 0.5s ease-out;
    `;
    
    const style = document.createElement('style');
    style.textContent = `
        @keyframes themeFlash {
            0% { opacity: 0; transform: scale(0); }
            50% { opacity: 1; transform: scale(1.5); }
            100% { opacity: 0; transform: scale(2); }
        }
    `;
    
    document.head.appendChild(style);
    document.body.appendChild(overlay);
    
    setTimeout(() => {
        overlay.remove();
        style.remove();
    }, 500);
}

// ==================== ANIMATED BACKGROUND ====================
function createAnimatedBackground() {
    const bgContainer = document.createElement('div');
    bgContainer.className = 'animated-bg';
    
    // Create gradient orbs
    for (let i = 1; i <= 3; i++) {
        const orb = document.createElement('div');
        orb.className = `gradient-orb orb-${i}`;
        bgContainer.appendChild(orb);
    }
    
    // Create floating particles
    for (let i = 0; i < 20; i++) {
        const particle = document.createElement('div');
        particle.className = 'particle';
        particle.style.cssText = `
            width: ${Math.random() * 6 + 2}px;
            height: ${Math.random() * 6 + 2}px;
            left: ${Math.random() * 100}%;
            bottom: -20px;
            animation-delay: ${Math.random() * 15}s;
            animation-duration: ${Math.random() * 10 + 10}s;
        `;
        bgContainer.appendChild(particle);
    }
    
    document.body.insertBefore(bgContainer, document.body.firstChild);
}

// ==================== CONTACT SELECTION ====================
function selectContact(userId, name) {
    currentReceiverId = userId;
    currentContactName = name;
    
    // Update UI
    document.getElementById('current-contact').textContent = `Chat with ${name}`;
    document.getElementById('message-input').disabled = false;
    document.getElementById('send-btn').disabled = false;
    
    // Update active state in sidebar
    document.querySelectorAll('.sidebar-menu a').forEach(link => {
        link.classList.remove('active');
    });
    event.target.closest('a').classList.add('active');
    
    // Load messages with animation
    loadMessages(userId);
    
    // Add contact selection animation
    createContactSelectEffect();
}

function createContactSelectEffect() {
    const chatHeader = document.getElementById('chat-header');
    chatHeader.style.animation = 'none';
    setTimeout(() => {
        chatHeader.style.animation = 'slideDown 0.6s cubic-bezier(0.4, 0, 0.2, 1)';
    }, 10);
}

// ==================== MESSAGE LOADING ====================
async function loadMessages(userId) {
    const container = document.getElementById('message-container');
    
    // Show loading spinner
    container.innerHTML = '<div class="spinner"></div>';
    
    try {
        const response = await fetch(`/api/messages/${userId}`);
        const messages = await response.json();
        
        // Clear container
        container.innerHTML = '';
        
        // Add empty state if no messages
        if (messages.length === 0) {
            showEmptyState(container);
            return;
        }
        
        // Add messages with staggered animation
        messages.forEach((msg, index) => {
            setTimeout(() => {
                addMessageToDOM(msg, container);
            }, index * 50);
        });
        
        // Scroll to bottom after all messages are loaded
        setTimeout(() => {
            container.scrollTop = container.scrollHeight;
        }, messages.length * 50 + 100);
        
    } catch (error) {
        console.error('Error loading messages:', error);
        container.innerHTML = '<div class="empty-state"><p>Error loading messages. Please try again.</p></div>';
    }
}

function showEmptyState(container) {
    container.innerHTML = `
        <div class="empty-state">
            <div class="empty-state-icon">💬</div>
            <h3>No messages yet</h3>
            <p>Start the conversation by sending a message!</p>
        </div>
    `;
}

function addMessageToDOM(msg, container) {
    const msgDiv = document.createElement('div');
    msgDiv.className = 'message ' + (msg.sender_id === CURRENT_USER_ID ? 'message-sent' : 'message-received');
    msgDiv.textContent = msg.content;
    msgDiv.style.opacity = '0';
    
    container.appendChild(msgDiv);
    
    // Trigger animation
    setTimeout(() => {
        msgDiv.style.opacity = '1';
    }, 10);
}

// ==================== MESSAGE SENDING ====================
function sendMessage() {
    const input = document.getElementById('message-input');
    const content = input.value.trim();
    
    if (!content || !currentReceiverId) return;
    
    // Send message via socket
    socket.emit('send_message', {
        receiver_id: currentReceiverId,
        content: content
    });
    
    // Clear input with animation
    input.style.transform = 'scale(0.95)';
    setTimeout(() => {
        input.value = '';
        input.style.transform = 'scale(1)';
        input.focus();
    }, 100);
    
    // Create send effect
    createSendEffect();
    
    // Stop typing indicator
    if (isTyping) {
        socket.emit('stop_typing', { receiver_id: currentReceiverId });
        isTyping = false;
    }
}

function createSendEffect() {
    const btn = document.getElementById('send-btn');
    
    // Create ripple effect
    const ripple = document.createElement('div');
    ripple.style.cssText = `
        position: absolute;
        top: 50%;
        left: 50%;
        width: 20px;
        height: 20px;
        background: rgba(255, 255, 255, 0.6);
        border-radius: 50%;
        transform: translate(-50%, -50%);
        animation: rippleEffect 0.6s ease-out;
        pointer-events: none;
    `;
    
    const style = document.createElement('style');
    style.textContent = `
        @keyframes rippleEffect {
            to {
                width: 100px;
                height: 100px;
                opacity: 0;
            }
        }
    `;
    
    document.head.appendChild(style);
    btn.style.position = 'relative';
    btn.appendChild(ripple);
    
    setTimeout(() => {
        ripple.remove();
        style.remove();
    }, 600);
}

// ==================== TYPING INDICATOR ====================
function handleTyping() {
    if (!currentReceiverId) return;
    
    if (!isTyping) {
        isTyping = true;
        socket.emit('typing', { receiver_id: currentReceiverId });
    }
    
    clearTimeout(typingTimeout);
    typingTimeout = setTimeout(() => {
        isTyping = false;
        socket.emit('stop_typing', { receiver_id: currentReceiverId });
    }, 2000);
}

function showTypingIndicator() {
    const container = document.getElementById('message-container');
    
    // Remove existing typing indicator
    const existing = container.querySelector('.typing-indicator');
    if (existing) return;
    
    const indicator = document.createElement('div');
    indicator.className = 'typing-indicator';
    indicator.innerHTML = `
        <div class="typing-dot"></div>
        <div class="typing-dot"></div>
        <div class="typing-dot"></div>
    `;
    
    container.appendChild(indicator);
    container.scrollTop = container.scrollHeight;
}

function hideTypingIndicator() {
    const indicator = document.querySelector('.typing-indicator');
    if (indicator) {
        indicator.style.animation = 'messageSlide 0.3s reverse';
        setTimeout(() => indicator.remove(), 300);
    }
}

// ==================== SOCKET EVENT HANDLERS ====================
socket.on('new_message', function(data) {
    if (data.sender_id == currentReceiverId || data.receiver_id == currentReceiverId) {
        const container = document.getElementById('message-container');
        
        // Remove empty state if exists
        const emptyState = container.querySelector('.empty-state');
        if (emptyState) {
            emptyState.remove();
        }
        
        addMessageToDOM(data, container);
        
        // Auto scroll to bottom
        setTimeout(() => {
            container.scrollTop = container.scrollHeight;
        }, 100);
        
        // Play notification sound (optional)
        if (data.sender_id !== CURRENT_USER_ID) {
            playNotificationSound();
            createMessageNotification(data.content);
        }
    }
});

socket.on('typing', function(data) {
    if (data.sender_id == currentReceiverId) {
        showTypingIndicator();
    }
});

socket.on('stop_typing', function(data) {
    if (data.sender_id == currentReceiverId) {
        hideTypingIndicator();
    }
});

socket.on('user_online', function(data) {
    updateUserStatus(data.user_id, true);
});

socket.on('user_offline', function(data) {
    updateUserStatus(data.user_id, false);
});

// ==================== USER STATUS ====================
function updateUserStatus(userId, isOnline) {
    const statusElements = document.querySelectorAll(`[data-user-id="${userId}"] .user-status`);
    statusElements.forEach(el => {
        el.classList.remove('status-online', 'status-offline');
        el.classList.add(isOnline ? 'status-online' : 'status-offline');
    });
}

// ==================== NOTIFICATIONS ====================
function createMessageNotification(content) {
    // Create floating notification
    const notification = document.createElement('div');
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        background: var(--glass-bg);
        backdrop-filter: blur(20px);
        border: 1px solid var(--glass-border);
        border-radius: 16px;
        padding: 16px 24px;
        max-width: 300px;
        box-shadow: 0 8px 32px var(--shadow-color);
        z-index: 10000;
        animation: slideInRight 0.5s cubic-bezier(0.4, 0, 0.2, 1);
    `;
    
    notification.innerHTML = `
        <div style="display: flex; align-items: center; gap: 12px;">
            <div style="font-size: 24px;">💬</div>
            <div>
                <div style="font-weight: 600; color: var(--text-primary); margin-bottom: 4px;">
                    New message from ${currentContactName}
                </div>
                <div style="color: var(--text-secondary); font-size: 14px;">
                    ${content.substring(0, 50)}${content.length > 50 ? '...' : ''}
                </div>
            </div>
        </div>
    `;
    
    const style = document.createElement('style');
    style.textContent = `
        @keyframes slideInRight {
            from {
                transform: translateX(400px);
                opacity: 0;
            }
            to {
                transform: translateX(0);
                opacity: 1;
            }
        }
        @keyframes slideOutRight {
            from {
                transform: translateX(0);
                opacity: 1;
            }
            to {
                transform: translateX(400px);
                opacity: 0;
            }
        }
    `;
    
    document.head.appendChild(style);
    document.body.appendChild(notification);
    
    setTimeout(() => {
        notification.style.animation = 'slideOutRight 0.5s cubic-bezier(0.4, 0, 0.2, 1)';
        setTimeout(() => {
            notification.remove();
            style.remove();
        }, 500);
    }, 4000);
}

function playNotificationSound() {
    // Create a subtle notification sound using Web Audio API
    try {
        const audioContext = new (window.AudioContext || window.webkitAudioContext)();
        const oscillator = audioContext.createOscillator();
        const gainNode = audioContext.createGain();
        
        oscillator.connect(gainNode);
        gainNode.connect(audioContext.destination);
        
        oscillator.frequency.value = 800;
        oscillator.type = 'sine';
        
        gainNode.gain.setValueAtTime(0.3, audioContext.currentTime);
        gainNode.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 0.5);
        
        oscillator.start(audioContext.currentTime);
        oscillator.stop(audioContext.currentTime + 0.5);
    } catch (e) {
        console.log('Audio notification not available');
    }
}

// ==================== NAVIGATION ====================
function goBack() {
    // Add exit animation
    const container = document.querySelector('.container');
    container.style.animation = 'fadeOut 0.3s ease-out';
    
    setTimeout(() => {
        if (document.referrer && document.referrer !== '') {
            window.history.back();
        } else {
            // Fallback to role-specific dashboard
            const userRole = document.body.dataset.userRole;
            if (userRole === 'Guide') {
                window.location.href = '/guide/dashboard';
            } else if (userRole === 'Learner') {
                window.location.href = '/learner/dashboard';
            } else if (userRole === 'Executive') {
                window.location.href = '/executive/dashboard';
            } else {
                window.location.href = '/';
            }
        }
    }, 300);
}

// ==================== MOUSE INTERACTION EFFECTS ====================
function addMouseInteractionEffects() {
    const dashboard = document.querySelector('.dashboard-content');
    if (!dashboard) return;
    
    dashboard.addEventListener('mousemove', (e) => {
        const rect = dashboard.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        
        // Create subtle glow effect at mouse position
        const glow = document.createElement('div');
        glow.style.cssText = `
            position: absolute;
            left: ${x}px;
            top: ${y}px;
            width: 200px;
            height: 200px;
            background: radial-gradient(circle, rgba(102, 126, 234, 0.1), transparent);
            pointer-events: none;
            transform: translate(-50%, -50%);
            transition: opacity 0.3s;
            border-radius: 50%;
            z-index: 0;
        `;
        
        // Remove old glow
        const oldGlow = dashboard.querySelector('.mouse-glow');
        if (oldGlow) oldGlow.remove();
        
        glow.className = 'mouse-glow';
        dashboard.appendChild(glow);
        
        setTimeout(() => glow.remove(), 300);
    });
}

// ==================== KEYBOARD SHORTCUTS ====================
function setupKeyboardShortcuts() {
    document.addEventListener('keydown', (e) => {
        // Ctrl/Cmd + K to focus search/input
        if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
            e.preventDefault();
            document.getElementById('message-input').focus();
        }
        
        // Escape to clear input
        if (e.key === 'Escape') {
            document.getElementById('message-input').value = '';
        }
    });
    
    // Enter to send message
    document.getElementById('message-input').addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });
    
    // Typing indicator
    document.getElementById('message-input').addEventListener('input', handleTyping);
}

// ==================== SMOOTH SCROLL ====================
function setupSmoothScroll() {
    const container = document.getElementById('message-container');
    if (!container) return;
    
    // Smooth scroll behavior
    container.style.scrollBehavior = 'smooth';
    
    // Auto-scroll on new messages
    const observer = new MutationObserver(() => {
        if (container.scrollHeight - container.scrollTop - container.clientHeight < 100) {
            container.scrollTop = container.scrollHeight;
        }
    });
    
    observer.observe(container, { childList: true });
}

// ==================== CONTACT SEARCH ====================
function addContactSearch() {
    const sidebar = document.querySelector('.sidebar');
    if (!sidebar) return;
    
    const searchContainer = document.createElement('div');
    searchContainer.style.cssText = `
        padding: 0 15px 20px;
    `;
    
    searchContainer.innerHTML = `
        <input type="text" 
               id="contact-search" 
               placeholder="Search contacts..." 
               style="width: 100%; 
                      padding: 12px 16px; 
                      background: rgba(255, 255, 255, 0.1); 
                      border: 1px solid var(--glass-border); 
                      border-radius: 12px; 
                      color: var(--text-primary); 
                      font-size: 14px; 
                      outline: none;
                      transition: all 0.3s;">
    `;
    
    const h3 = sidebar.querySelector('h3');
    h3.after(searchContainer);
    
    // Search functionality
    const searchInput = document.getElementById('contact-search');
    searchInput.addEventListener('input', (e) => {
        const query = e.target.value.toLowerCase();
        const contacts = document.querySelectorAll('.sidebar-menu li');
        
        contacts.forEach(contact => {
            const name = contact.textContent.toLowerCase();
            if (name.includes(query)) {
                contact.style.display = 'block';
                contact.style.animation = 'fadeIn 0.3s';
            } else {
                contact.style.display = 'none';
            }
        });
    });
    
    searchInput.addEventListener('focus', () => {
        searchInput.style.borderColor = 'var(--accent-primary)';
        searchInput.style.background = 'rgba(255, 255, 255, 0.15)';
    });
    
    searchInput.addEventListener('blur', () => {
        searchInput.style.borderColor = 'var(--glass-border)';
        searchInput.style.background = 'rgba(255, 255, 255, 0.1)';
    });
}

// ==================== MESSAGE REACTIONS ====================
function addMessageReactions() {
    const container = document.getElementById('message-container');
    if (!container) return;
    
    container.addEventListener('contextmenu', (e) => {
        if (e.target.classList.contains('message')) {
            e.preventDefault();
            showReactionMenu(e.target, e.clientX, e.clientY);
        }
    });
}

function showReactionMenu(messageEl, x, y) {
    // Remove existing menu
    const existing = document.querySelector('.reaction-menu');
    if (existing) existing.remove();
    
    const menu = document.createElement('div');
    menu.className = 'reaction-menu';
    menu.style.cssText = `
        position: fixed;
        left: ${x}px;
        top: ${y}px;
        background: var(--glass-bg);
        backdrop-filter: blur(20px);
        border: 1px solid var(--glass-border);
        border-radius: 12px;
        padding: 8px;
        display: flex;
        gap: 8px;
        box-shadow: 0 8px 32px var(--shadow-color);
        z-index: 10000;
        animation: scaleIn 0.2s cubic-bezier(0.4, 0, 0.2, 1);
    `;
    
    const reactions = ['👍', '❤️', '😂', '😮', '😢', '🙏'];
    reactions.forEach(emoji => {
        const btn = document.createElement('button');
        btn.textContent = emoji;
        btn.style.cssText = `
            background: transparent;
            border: none;
            font-size: 24px;
            cursor: pointer;
            padding: 8px;
            border-radius: 8px;
            transition: transform 0.2s, background 0.2s;
        `;
        
        btn.addEventListener('mouseenter', () => {
            btn.style.transform = 'scale(1.3)';
            btn.style.background = 'rgba(255, 255, 255, 0.1)';
        });
        
        btn.addEventListener('mouseleave', () => {
            btn.style.transform = 'scale(1)';
            btn.style.background = 'transparent';
        });
        
        btn.addEventListener('click', () => {
            addReactionToMessage(messageEl, emoji);
            menu.remove();
        });
        
        menu.appendChild(btn);
    });
    
    const style = document.createElement('style');
    style.textContent = `
        @keyframes scaleIn {
            from {
                transform: scale(0);
                opacity: 0;
            }
            to {
                transform: scale(1);
                opacity: 1;
            }
        }
    `;
    
    document.head.appendChild(style);
    document.body.appendChild(menu);
    
    // Close menu on click outside
    setTimeout(() => {
        document.addEventListener('click', function closeMenu() {
            menu.remove();
            style.remove();
            document.removeEventListener('click', closeMenu);
        });
    }, 100);
}

function addReactionToMessage(messageEl, emoji) {
    let reactionContainer = messageEl.querySelector('.message-reactions');
    
    if (!reactionContainer) {
        reactionContainer = document.createElement('div');
        reactionContainer.className = 'message-reactions';
        reactionContainer.style.cssText = `
            display: flex;
            gap: 4px;
            margin-top: 8px;
            flex-wrap: wrap;
        `;
        messageEl.appendChild(reactionContainer);
    }
    
    const reaction = document.createElement('span');
    reaction.textContent = emoji;
    reaction.style.cssText = `
        font-size: 16px;
        padding: 4px 8px;
        background: rgba(255, 255, 255, 0.2);
        border-radius: 12px;
        animation: reactionPop 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    `;
    
    const style = document.createElement('style');
    style.textContent = `
        @keyframes reactionPop {
            0% { transform: scale(0); }
            50% { transform: scale(1.2); }
            100% { transform: scale(1); }
        }
    `;
    
    document.head.appendChild(style);
    reactionContainer.appendChild(reaction);
    
    setTimeout(() => style.remove(), 300);
}

// ==================== INITIALIZATION ====================
document.addEventListener('DOMContentLoaded', () => {
    // Initialize all features
    initTheme();
    createAnimatedBackground();
    setupKeyboardShortcuts();
    setupSmoothScroll();
    addMouseInteractionEffects();
    addContactSearch();
    addMessageReactions();
    
    // Set user data in body
    document.body.dataset.userId = CURRENT_USER_ID;
    
    console.log('✨ Interactive messaging system initialized!');
});

// Export functions for use in HTML
window.selectContact = selectContact;
window.sendMessage = sendMessage;
window.goBack = goBack;
window.toggleTheme = toggleTheme;