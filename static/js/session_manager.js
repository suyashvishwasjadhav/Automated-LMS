/**
 * Session Management for Multi-Tab Synchronization
 * Handles cross-tab logout detection and session timeout warnings
 */

(function() {
    'use strict';
    
    // Configuration
    const SESSION_CHECK_INTERVAL = 5000; // Check every 5 seconds
    const SESSION_TIMEOUT_WARNING = 25 * 60 * 1000; // Warn at 25 minutes (5 min before timeout)
    const STORAGE_KEY = 'app_session_status';
    
    let lastActivityTime = Date.now();
    let warningShown = false;
    
    /**
     * Update last activity time on user interaction
     */
    function updateActivity() {
        lastActivityTime = Date.now();
        warningShown = false;
        
        // Store activity in localStorage for cross-tab sync
        try {
            localStorage.setItem('last_activity', lastActivityTime.toString());
        } catch (e) {
            console.warn('localStorage not available:', e);
        }
    }
    
    /**
     * Check if user has been logged out in another tab
     */
    function checkSessionStatus() {
        try {
            const sessionStatus = localStorage.getItem(STORAGE_KEY);
            
            // If session status is 'logged_out', redirect to login
            if (sessionStatus === 'logged_out') {
                // Clear the flag
                localStorage.removeItem(STORAGE_KEY);
                
                // Show message and redirect
                alert('You have been logged out from another tab.');
                window.location.href = '/login';
                return;
            }
            
            // Check for inactivity timeout warning
            const timeSinceActivity = Date.now() - lastActivityTime;
            if (timeSinceActivity > SESSION_TIMEOUT_WARNING && !warningShown) {
                warningShown = true;
                showTimeoutWarning();
            }
            
        } catch (e) {
            console.warn('Session check error:', e);
        }
    }
    
    /**
     * Show session timeout warning
     */
    function showTimeoutWarning() {
        const remainingTime = Math.ceil((30 * 60 * 1000 - (Date.now() - lastActivityTime)) / 60000);
        
        if (remainingTime > 0) {
            const warning = document.createElement('div');
            warning.id = 'session-timeout-warning';
            warning.style.cssText = `
                position: fixed;
                top: 20px;
                right: 20px;
                background: #ff9800;
                color: white;
                padding: 15px 20px;
                border-radius: 5px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                z-index: 10000;
                max-width: 300px;
                font-family: Arial, sans-serif;
            `;
            warning.innerHTML = `
                <strong>⚠️ Session Timeout Warning</strong>
                <p style="margin: 10px 0 0 0; font-size: 14px;">
                    Your session will expire in approximately ${remainingTime} minute${remainingTime !== 1 ? 's' : ''} due to inactivity.
                    Click anywhere to stay logged in.
                </p>
            `;
            
            document.body.appendChild(warning);
            
            // Remove warning after 10 seconds or on user interaction
            const removeWarning = () => {
                const warningEl = document.getElementById('session-timeout-warning');
                if (warningEl) {
                    warningEl.remove();
                }
            };
            
            setTimeout(removeWarning, 10000);
            document.addEventListener('click', removeWarning, { once: true });
        }
    }
    
    /**
     * Mark session as logged out in localStorage
     * This will be detected by other tabs
     */
    function markSessionLoggedOut() {
        try {
            localStorage.setItem(STORAGE_KEY, 'logged_out');
        } catch (e) {
            console.warn('Could not mark session as logged out:', e);
        }
    }
    
    /**
     * Initialize session manager
     */
    function init() {
        // Track user activity
        const activityEvents = ['mousedown', 'keydown', 'scroll', 'touchstart', 'click'];
        activityEvents.forEach(event => {
            document.addEventListener(event, updateActivity, { passive: true });
        });
        
        // Check session status periodically
        setInterval(checkSessionStatus, SESSION_CHECK_INTERVAL);
        
        // Listen for logout events
        const logoutLinks = document.querySelectorAll('a[href*="/logout"]');
        logoutLinks.forEach(link => {
            link.addEventListener('click', markSessionLoggedOut);
        });
        
        // Listen for storage events (cross-tab communication)
        window.addEventListener('storage', function(e) {
            if (e.key === STORAGE_KEY && e.newValue === 'logged_out') {
                // Another tab logged out
                alert('You have been logged out from another tab.');
                window.location.href = '/login';
            }
        });
        
        // Clear logged_out flag on page load (user is logged in)
        try {
            if (localStorage.getItem(STORAGE_KEY) === 'logged_out') {
                localStorage.removeItem(STORAGE_KEY);
            }
        } catch (e) {
            console.warn('Could not clear session status:', e);
        }
        
        console.log('✅ Session manager initialized');
    }
    
    // Initialize when DOM is ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
    
})();
