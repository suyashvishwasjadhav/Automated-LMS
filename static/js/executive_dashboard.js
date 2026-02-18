// Theme Toggle
const themeToggle = document.getElementById('themeToggle');
const html = document.documentElement;

// Check for saved theme preference or default to light mode
const currentTheme = localStorage.getItem('theme') || 'light';
html.setAttribute('data-theme', currentTheme);
updateThemeIcon(currentTheme);

themeToggle.addEventListener('click', () => {
    const theme = html.getAttribute('data-theme') === 'light' ? 'dark' : 'light';
    html.setAttribute('data-theme', theme);
    localStorage.setItem('theme', theme);
    updateThemeIcon(theme);
    
    // Add animation effect
    themeToggle.style.transform = 'scale(0.8) rotate(180deg)';
    setTimeout(() => {
        themeToggle.style.transform = 'scale(1) rotate(0deg)';
    }, 300);
});

function updateThemeIcon(theme) {
    const icon = themeToggle.querySelector('i');
    icon.className = theme === 'light' ? 'fas fa-moon' : 'fas fa-sun';
}

// Navigation
const navItems = document.querySelectorAll('.nav-item');
const contentSections = document.querySelectorAll('.content-section');

navItems.forEach(item => {
    item.addEventListener('click', (e) => {
        // Only prevent default and switch sections for items with data-section attribute
        // Allow normal navigation for links without data-section (like Messages)
        if (item.dataset.section) {
            e.preventDefault();
            const targetSection = item.dataset.section;
            switchSection(targetSection);
        }
        // If no data-section, let the link navigate normally
    });
});

function switchSection(sectionId) {
    // Update nav active state
    navItems.forEach(item => {
        item.classList.remove('active');
        if (item.dataset.section === sectionId) {
            item.classList.add('active');
        }
    });
    
    // Update content visibility
    contentSections.forEach(section => {
        section.classList.remove('active');
        if (section.id === sectionId) {
            section.classList.add('active');
            
            // Load data for specific sections
            if (sectionId === 'all-courses') {
                loadAllCourses();
            }
        }
    });
    
    // Smooth scroll to top
    document.querySelector('.dashboard-main').scrollTo({
        top: 0,
        behavior: 'smooth'
    });
}

// Calculate total learners
function calculateTotalLearners() {
    const courseCards = document.querySelectorAll('.course-card');
    let total = 0;
    courseCards.forEach(card => {
        const learnersText = card.querySelector('.learners-header h4')?.textContent;
        const match = learnersText?.match(/\((\d+)\)/);
        if (match) {
            total += parseInt(match[1]);
        }
    });
    document.getElementById('totalLearners').textContent = total;
}

calculateTotalLearners();

// Load All Courses
async function loadAllCourses() {
    const container = document.getElementById('allCoursesContainer');
    container.innerHTML = '<div class="loading-spinner"><i class="fas fa-spinner fa-spin"></i> Loading courses...</div>';
    
    try {
        // Simulate API call - in real implementation, fetch from backend
        setTimeout(() => {
            const dashboardCourses = document.querySelectorAll('#dashboard .course-card');
            if (dashboardCourses.length > 0) {
                container.innerHTML = '';
                dashboardCourses.forEach(card => {
                    const clone = card.cloneNode(true);
                    container.appendChild(clone);
                });
            } else {
                container.innerHTML = `
                    <div class="empty-state glass-card">
                        <i class="fas fa-inbox"></i>
                        <h3>No Courses Available</h3>
                        <p>There are no courses in the system yet.</p>
                    </div>
                `;
            }
        }, 500);
    } catch (error) {
        container.innerHTML = `
            <div class="empty-state glass-card">
                <i class="fas fa-exclamation-circle"></i>
                <h3>Error Loading Courses</h3>
                <p>Please try again later.</p>
            </div>
        `;
    }
}

// Multi-step form variables
let currentStep = 1;
const totalSteps = 2;

function nextStep() {
    // Validate current step before moving forward
    if (currentStep === 1) {
        const courseName = document.getElementById('course_name').value.trim();
        const guideEmail = document.getElementById('guide_email').value.trim();
        
        if (!courseName) {
            showAlert('Please enter a course name', 'error');
            return;
        }
        
        if (courseName.length < 3) {
            showAlert('Course name must be at least 3 characters', 'error');
            return;
        }
        
        if (!guideEmail) {
            showAlert('Please enter guide email', 'error');
            return;
        }
        
        if (!validateEmail(guideEmail)) {
            showAlert('Please enter a valid guide email', 'error');
            return;
        }
    }
    
    if (currentStep < totalSteps) {
        document.querySelector(`.form-step[data-step="${currentStep}"]`).classList.remove('active');
        currentStep++;
        document.querySelector(`.form-step[data-step="${currentStep}"]`).classList.add('active');
    }
}

function prevStep() {
    if (currentStep > 1) {
        document.querySelector(`.form-step[data-step="${currentStep}"]`).classList.remove('active');
        currentStep--;
        document.querySelector(`.form-step[data-step="${currentStep}"]`).classList.add('active');
    }
}

// Create Course Form with Full Validation
document.getElementById('createCourseForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    // Get form values
    const courseName = document.getElementById('course_name').value.trim();
    const guideEmail = document.getElementById('guide_email').value.trim();
    const learnerEmailsInput = document.getElementById('learner_emails').value.trim();
    
    // Validate course name
    if (!courseName) {
        showAlert('Please enter a course name', 'error');
        return;
    }
    
    if (courseName.length < 3) {
        showAlert('Course name must be at least 3 characters', 'error');
        return;
    }
    
    // Validate guide email
    if (!guideEmail) {
        showAlert('Please enter guide email', 'error');
        return;
    }
    
    if (!validateEmail(guideEmail)) {
        showAlert('Please enter a valid guide email', 'error');
        return;
    }
    
    // Validate learner emails if provided
    if (learnerEmailsInput) {
        const learnerEmails = learnerEmailsInput.split(',').map(e => e.trim()).filter(e => e);
        
        if (learnerEmails.length > 20) {
            showAlert('Maximum 20 learners can be added per course', 'error');
            return;
        }
        
        // Check each email format
        const invalidEmails = learnerEmails.filter(email => !validateEmail(email));
        if (invalidEmails.length > 0) {
            showAlert(`Invalid email format: ${invalidEmails.slice(0, 3).join(', ')}${invalidEmails.length > 3 ? '...' : ''}`, 'error');
            return;
        }
    }
    
    // Prepare form data
    const formData = new FormData(e.target);
    
    // Add CSRF token
    const csrfToken = document.querySelector('meta[name="csrf-token"]')?.getAttribute('content');
    if (csrfToken) {
        formData.append('csrf_token', csrfToken);
    }
    
    const submitBtn = e.target.querySelector('button[type="submit"]');
    const originalText = submitBtn.innerHTML;
    
    // Disable button and show loading
    submitBtn.disabled = true;
    submitBtn.innerHTML = '<span class="loading"></span> Creating Course...';
    
    try {
        const response = await fetch('/executive/create_course', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (data.success) {
            showAlert(data.message, 'success');
            
            // Reset form
            e.target.reset();
            currentStep = 1;
            document.querySelectorAll('.form-step').forEach(step => step.classList.remove('active'));
            document.querySelector('.form-step[data-step="1"]').classList.add('active');
            
            // Reload page after 2 seconds to show new course
            setTimeout(() => {
                location.reload();
            }, 2000);
        } else {
            showAlert(data.message || 'Failed to create course', 'error');
        }
    } catch (error) {
        console.error('Create course error:', error);
        showAlert('An error occurred while creating the course. Please try again.', 'error');
    } finally {
        // Re-enable button
        submitBtn.disabled = false;
        submitBtn.innerHTML = originalText;
    }
});

// Announcement Form
document.getElementById('announcementForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const title = document.getElementById('ann_title').value.trim();
    const content = document.getElementById('ann_content').value.trim();
    
    if (!title || !content) {
        showAlert('Please fill in all required fields', 'error');
        return;
    }
    
    const formData = new FormData(e.target);
    
    // Add CSRF token
    const csrfToken = document.querySelector('meta[name="csrf-token"]')?.getAttribute('content');
    if (csrfToken) {
        formData.append('csrf_token', csrfToken);
    }
    
    const submitBtn = e.target.querySelector('button[type="submit"]');
    const originalText = submitBtn.innerHTML;
    
    submitBtn.disabled = true;
    submitBtn.innerHTML = '<span class="loading"></span> Posting...';
    
    try {
        const response = await fetch('/executive/announcements', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (data.success) {
            showAlert(data.message, 'success');
            setTimeout(() => location.reload(), 1500);
        } else {
            showAlert(data.message, 'error');
        }
    } catch (error) {
        showAlert('An error occurred while posting the announcement', 'error');
    } finally {
        submitBtn.disabled = false;
        submitBtn.innerHTML = originalText;
    }
});

// Add Learners Modal Functions
function openAddLearnersModal(courseId) {
    document.getElementById('add_course_id').value = courseId;
    document.getElementById('addLearnersModal').classList.add('active');
    document.body.style.overflow = 'hidden';
}

function closeAddLearnersModal() {
    document.getElementById('addLearnersModal').classList.remove('active');
    document.body.style.overflow = 'auto';
    document.getElementById('addLearnersForm').reset();
}

// Add Learners Form with Validation
document.getElementById('addLearnersForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const emailsInput = document.getElementById('new_learner_emails').value.trim();
    
    if (!emailsInput) {
        showAlert('Please enter at least one learner email', 'error');
        return;
    }
    
    const emails = emailsInput.split(',').map(e => e.trim()).filter(e => e);
    
    // Validate emails
    const invalidEmails = emails.filter(email => !validateEmail(email));
    if (invalidEmails.length > 0) {
        showAlert(`Invalid email format: ${invalidEmails.join(', ')}`, 'error');
        return;
    }
    
    if (emails.length > 20) {
        showAlert('Maximum 20 learners can be added at once', 'error');
        return;
    }
    
    const formData = new FormData(e.target);
    
    // Add CSRF token
    const csrfToken = document.querySelector('meta[name="csrf-token"]')?.getAttribute('content');
    if (csrfToken) {
        formData.append('csrf_token', csrfToken);
    }
    
    const submitBtn = e.target.querySelector('button[type="submit"]');
    const originalText = submitBtn.innerHTML;
    
    submitBtn.disabled = true;
    submitBtn.innerHTML = '<span class="loading"></span> Adding...';
    
    try {
        const response = await fetch('/executive/add_learners', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (data.success) {
            showAlert(data.message, 'success');
            closeAddLearnersModal();
            setTimeout(() => location.reload(), 2000);
        } else {
            showAlert(data.message, 'error');
        }
    } catch (error) {
        showAlert('An error occurred while adding learners', 'error');
    } finally {
        submitBtn.disabled = false;
        submitBtn.innerHTML = originalText;
    }
});

// Payment Modal Functions
function openPaymentModal(planType, amount) {
    document.getElementById('plan_type').value = planType;
    document.getElementById('plan_amount').value = amount;
    document.getElementById('selectedPlan').textContent = `${planType.charAt(0).toUpperCase() + planType.slice(1)} Plan - $${amount}/month`;
    
    const subtotal = amount;
    const tax = (amount * 0.1).toFixed(2);
    const total = (amount * 1.1).toFixed(2);
    
    document.getElementById('subtotal').textContent = `$${subtotal}`;
    document.getElementById('tax').textContent = `$${tax}`;
    document.getElementById('total').textContent = `$${total}`;
    
    document.getElementById('paymentModal').classList.add('active');
    document.body.style.overflow = 'hidden';
}

function closePaymentModal() {
    document.getElementById('paymentModal').classList.remove('active');
    document.body.style.overflow = 'auto';
    document.getElementById('paymentForm').reset();
}

// Payment Form with Validation
document.getElementById('paymentForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const cardName = e.target.card_name.value.trim();
    const cardNumber = e.target.card_number.value.replace(/\s/g, '');
    const cardExpiry = e.target.card_expiry.value.trim();
    const cardCvv = e.target.card_cvv.value.trim();
    
    // Validate card name
    if (cardName.length < 3) {
        showAlert('Please enter a valid cardholder name', 'error');
        return;
    }
    
    // Validate card number (16 digits)
    if (!/^\d{16}$/.test(cardNumber)) {
        showAlert('Please enter a valid 16-digit card number', 'error');
        return;
    }
    
    // Validate expiry (MM/YY format)
    if (!/^\d{2}\/\d{2}$/.test(cardExpiry)) {
        showAlert('Please enter expiry in MM/YY format', 'error');
        return;
    }
    
    const [month, year] = cardExpiry.split('/').map(num => parseInt(num));
    const currentYear = new Date().getFullYear() % 100;
    const currentMonth = new Date().getMonth() + 1;
    
    if (month < 1 || month > 12) {
        showAlert('Invalid expiry month', 'error');
        return;
    }
    
    if (year < currentYear || (year === currentYear && month < currentMonth)) {
        showAlert('Card has expired', 'error');
        return;
    }
    
    // Validate CVV (3 digits)
    if (!/^\d{3}$/.test(cardCvv)) {
        showAlert('Please enter a valid 3-digit CVV', 'error');
        return;
    }
    
    const submitBtn = e.target.querySelector('button[type="submit"]');
    const originalText = submitBtn.innerHTML;
    
    submitBtn.disabled = true;
    submitBtn.innerHTML = '<span class="loading"></span> Processing...';
    
    // Simulate payment processing
    setTimeout(() => {
        showAlert('Payment successful! Your plan has been upgraded.', 'success');
        closePaymentModal();
        submitBtn.disabled = false;
        submitBtn.innerHTML = originalText;
        
        // Update UI to reflect upgrade
        setTimeout(() => {
            const currentPlanCard = document.querySelector('.pricing-card:first-child');
            if (currentPlanCard) {
                currentPlanCard.querySelector('.plan-badge').textContent = 'Previous';
                const btn = currentPlanCard.querySelector('button');
                btn.textContent = 'Downgrade';
                btn.classList.remove('btn-secondary');
                btn.classList.add('btn-primary');
                btn.disabled = false;
            }
            
            const proPlanCard = document.querySelector('.pricing-card.featured');
            if (proPlanCard) {
                proPlanCard.querySelector('.plan-badge').textContent = 'Current';
                proPlanCard.querySelector('.plan-badge').classList.remove('popular');
                const btn = proPlanCard.querySelector('button');
                btn.textContent = 'Current Plan';
                btn.disabled = true;
            }
        }, 1000);
    }, 2000);
});

// Format card number input
document.querySelector('input[name="card_number"]').addEventListener('input', function(e) {
    let value = e.target.value.replace(/\s/g, '');
    let formattedValue = value.match(/.{1,4}/g)?.join(' ') || value;
    e.target.value = formattedValue;
});

// Format expiry input
document.querySelector('input[name="card_expiry"]').addEventListener('input', function(e) {
    let value = e.target.value.replace(/\D/g, '');
    if (value.length >= 2) {
        value = value.slice(0, 2) + '/' + value.slice(2, 4);
    }
    e.target.value = value;
});

// Format CVV input
document.querySelector('input[name="card_cvv"]').addEventListener('input', function(e) {
    e.target.value = e.target.value.replace(/\D/g, '').slice(0, 3);
});

// OTP Modal Functions
function closeOtpModal() {
    document.getElementById('otpModal').classList.remove('active');
    document.body.style.overflow = 'auto';
    document.getElementById('otpForm').reset();
}

async function deleteUser(userId, userRole) {
    // Only allow deletion of Learner role
    if (userRole !== 'Learner') {
        showAlert('Only Learner accounts can be deleted', 'error');
        return;
    }
    
    if (!confirm('Are you sure you want to delete this learner? This action cannot be undone.')) {
        return;
    }
    
    try {
        // Get CSRF token
        const csrfToken = document.querySelector('meta[name="csrf-token"]')?.getAttribute('content');
        
        const headers = {};
        if (csrfToken) {
            headers['X-CSRFToken'] = csrfToken;
        }
        
        const response = await fetch(`/executive/delete_user/${userId}`, {
            method: 'POST',
            headers: headers
        });
        
        const data = await response.json();
        
        if (data.success) {
            showAlert(data.message, 'success');
            document.getElementById('delete_user_id').value = userId;
            document.getElementById('otpModal').classList.add('active');
            document.body.style.overflow = 'hidden';
        } else {
            showAlert(data.message, 'error');
        }
    } catch (error) {
        console.error('Delete error:', error);
        showAlert('An error occurred', 'error');
    }
}

// OTP Form with Validation
document.getElementById('otpForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const otpCode = document.getElementById('otp_code').value.trim();
    
    if (!/^\d{6}$/.test(otpCode)) {
        showAlert('Please enter a valid 6-digit OTP', 'error');
        return;
    }
    
    const formData = new FormData(e.target);
    
    // Add CSRF token
    const csrfToken = document.querySelector('meta[name="csrf-token"]')?.getAttribute('content');
    if (csrfToken) {
        formData.append('csrf_token', csrfToken);
    }
    
    const submitBtn = e.target.querySelector('button[type="submit"]');
    const originalText = submitBtn.innerHTML;
    
    submitBtn.disabled = true;
    submitBtn.innerHTML = '<span class="loading"></span> Verifying...';
    
    try {
        const response = await fetch('/executive/verify_delete_user', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (data.success) {
            showAlert(data.message, 'success');
            closeOtpModal();
            setTimeout(() => location.reload(), 2000);
        } else {
            showAlert(data.message, 'error');
        }
    } catch (error) {
        showAlert('An error occurred', 'error');
    } finally {
        submitBtn.disabled = false;
        submitBtn.innerHTML = originalText;
    }
});

// View Course Details
function viewCourseDetails(courseId) {
    showAlert('Course details view coming soon!', 'success');
}

// Email Validation Helper
function validateEmail(email) {
    const re = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return re.test(email);
}

// Alert System
function showAlert(message, type) {
    const alertContainer = document.getElementById('alert-container');
    const alertClass = type === 'success' ? 'alert-success' : 'alert-error';
    const icon = type === 'success' ? 'fas fa-check-circle' : 'fas fa-exclamation-circle';
    
    const alert = document.createElement('div');
    alert.className = `alert ${alertClass}`;
    alert.innerHTML = `
        <i class="${icon}"></i>
        <span>${message}</span>
    `;
    
    alertContainer.appendChild(alert);
    
    setTimeout(() => {
        alert.style.animation = 'slideOutRight 0.4s ease-out';
        setTimeout(() => alert.remove(), 400);
    }, 5000);
}

// File upload preview
document.getElementById('ann_file').addEventListener('change', function(e) {
    const file = e.target.files[0];
    if (file) {
        const label = document.querySelector('.file-upload-label span');
        label.textContent = `Selected: ${file.name}`;
    }
});

// Close modals on overlay click
document.querySelectorAll('.modal-overlay').forEach(overlay => {
    overlay.addEventListener('click', function() {
        this.parentElement.classList.remove('active');
        document.body.style.overflow = 'auto';
    });
});

// Escape key to close modals
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
        document.querySelectorAll('.modal.active').forEach(modal => {
            modal.classList.remove('active');
            document.body.style.overflow = 'auto';
        });
    }
});

// Smooth scroll for anchor links
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function(e) {
        const href = this.getAttribute('href');
        if (href !== '#') {
            e.preventDefault();
            const target = document.querySelector(href);
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        }
    });
});

// Add loading animation to course cards
const courseCards = document.querySelectorAll('.course-card');
courseCards.forEach((card, index) => {
    card.style.animationDelay = `${index * 0.1}s`;
});

// Initialize tooltips (simple implementation)
document.querySelectorAll('[title]').forEach(element => {
    element.addEventListener('mouseenter', function() {
        const tooltip = document.createElement('div');
        tooltip.className = 'tooltip';
        tooltip.textContent = this.getAttribute('title');
        tooltip.style.cssText = `
            position: absolute;
            background: var(--glass-bg);
            padding: 8px 12px;
            border-radius: 8px;
            font-size: 13px;
            pointer-events: none;
            z-index: 1000;
            box-shadow: var(--shadow-md);
        `;
        document.body.appendChild(tooltip);
        
        const rect = this.getBoundingClientRect();
        tooltip.style.top = `${rect.top - tooltip.offsetHeight - 8}px`;
        tooltip.style.left = `${rect.left + (rect.width - tooltip.offsetWidth) / 2}px`;
        
        this._tooltip = tooltip;
    });
    
    element.addEventListener('mouseleave', function() {
        if (this._tooltip) {
            this._tooltip.remove();
            delete this._tooltip;
        }
    });
});

// Page load animation
window.addEventListener('load', () => {
    document.body.style.opacity = '0';
    setTimeout(() => {
        document.body.style.transition = 'opacity 0.5s ease-in';
        document.body.style.opacity = '1';
    }, 100);
});

console.log('Executive Dashboard initialized successfully!');



// View Course Details - Enhanced with Modal
function viewCourseDetails(courseId) {
    fetch(`/api/course/${courseId}/details`)
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                openCourseDetailsModal(data.course);
            } else {
                showAlert(data.message || 'Failed to load course details', 'error');
            }
        })
        .catch(error => {
            showAlert('Error loading course details', 'error');
        });
}

function openCourseDetailsModal(course) {
    const modal = document.getElementById('courseDetailsModal');
    if (!modal) {
        createCourseDetailsModal();
        return openCourseDetailsModal(course);
    }
    
    document.getElementById('detail_course_name').textContent = course.course_name;
    document.getElementById('detail_organization').textContent = course.organization;
    document.getElementById('detail_created_at').textContent = course.created_at;
    
    if (course.guide) {
        document.getElementById('detail_guide_name').textContent = course.guide.full_name || course.guide.username;
        document.getElementById('detail_guide_email').textContent = course.guide.email;
        
        const guidePhoto = document.getElementById('detail_guide_photo');
        if (course.guide.profile_photo) {
            guidePhoto.innerHTML = `<img src="${course.guide.profile_photo}" alt="${course.guide.full_name}">`;
        } else {
            guidePhoto.innerHTML = '<div class="avatar-placeholder"><i class="fas fa-user"></i></div>';
        }
    }
    
    const learnersContainer = document.getElementById('detail_learners_list');
    learnersContainer.innerHTML = '';
    
    if (course.learners && course.learners.length > 0) {
        course.learners.forEach(learner => {
            const learnerCard = document.createElement('div');
            learnerCard.className = 'detail-learner-card';
            learnerCard.innerHTML = `
                <div class="learner-card-photo">
                    ${learner.profile_photo 
                        ? `<img src="${learner.profile_photo}" alt="${learner.full_name}">`
                        : '<div class="avatar-placeholder"><i class="fas fa-user"></i></div>'
                    }
                    <span class="status-dot ${learner.is_online ? 'online' : 'offline'}"></span>
                </div>
                <div class="learner-card-info">
                    <h4>${learner.full_name || learner.username}</h4>
                    <p>${learner.email}</p>
                </div>
                <button class="btn-icon btn-danger-icon" onclick="deleteLearner(${learner.id}, '${learner.full_name || learner.username}')" title="Delete Learner">
                    <i class="fas fa-trash"></i>
                </button>
            `;
            learnersContainer.appendChild(learnerCard);
        });
    } else {
        learnersContainer.innerHTML = '<p class="no-learners">No learners enrolled yet</p>';
    }
    
    modal.classList.add('active');
    document.body.style.overflow = 'hidden';
}

function closeCourseDetailsModal() {
    const modal = document.getElementById('courseDetailsModal');
    if (modal) {
        modal.classList.remove('active');
        document.body.style.overflow = 'auto';
    }
}

function createCourseDetailsModal() {
    const modalHTML = `
        <div id="courseDetailsModal" class="modal">
            <div class="modal-overlay" onclick="closeCourseDetailsModal()"></div>
            <div class="modal-content glass-card course-details-modal">
                <button class="modal-close" onclick="closeCourseDetailsModal()">
                    <i class="fas fa-times"></i>
                </button>
                <div class="modal-header">
                    <h2><i class="fas fa-book-reader"></i> Course Details</h2>
                </div>
                <div class="modal-body">
                    <div class="detail-section">
                        <h3><i class="fas fa-info-circle"></i> Course Information</h3>
                        <div class="detail-grid">
                            <div class="detail-item">
                                <label>Course Name</label>
                                <p id="detail_course_name"></p>
                            </div>
                            <div class="detail-item">
                                <label>Organization</label>
                                <p id="detail_organization"></p>
                            </div>
                            <div class="detail-item">
                                <label>Created Date</label>
                                <p id="detail_created_at"></p>
                            </div>
                        </div>
                    </div>
                    
                    <div class="detail-section">
                        <h3><i class="fas fa-user-tie"></i> Guide Information</h3>
                        <div class="guide-detail-card">
                            <div id="detail_guide_photo" class="guide-photo"></div>
                            <div class="guide-info">
                                <h4 id="detail_guide_name"></h4>
                                <p id="detail_guide_email"></p>
                            </div>
                        </div>
                    </div>
                    
                    <div class="detail-section">
                        <h3><i class="fas fa-users"></i> Enrolled Learners</h3>
                        <div id="detail_learners_list" class="learners-detail-list"></div>
                    </div>
                </div>
            </div>
        </div>
    `;
    
    document.body.insertAdjacentHTML('beforeend', modalHTML);
}

function deleteLearner(learnerId, learnerName) {
    if (!confirm(`Are you sure you want to delete ${learnerName}? This action cannot be undone.`)) {
        return;
    }
    deleteUser(learnerId, 'Learner');
}

async function deleteUser(userId, userRole) {
    if (userRole !== 'Learner') {
        showAlert('Only Learner accounts can be deleted', 'error');
        return;
    }
    
    try {
        // Get CSRF token
        const csrfToken = document.querySelector('meta[name="csrf-token"]')?.getAttribute('content');
        
        const headers = {
            'Content-Type': 'application/json',
        };
        if (csrfToken) {
            headers['X-CSRFToken'] = csrfToken;
        }
        
        const response = await fetch(`/executive/delete_user/${userId}`, {
            method: 'POST',
            headers: headers
        });
        
        const data = await response.json();
        
        if (data.success) {
            showAlert('OTP sent to your email. Please verify to complete deletion.', 'success');
            document.getElementById('delete_user_id').value = userId;
            closeCourseDetailsModal();
            document.getElementById('otpModal').classList.add('active');
            document.body.style.overflow = 'hidden';
        } else {
            showAlert(data.message, 'error');
        }
    } catch (error) {
        console.error('Delete error:', error);
        showAlert('An error occurred while processing the deletion request', 'error');
    }
}

window.viewCourseDetails = viewCourseDetails;
window.closeCourseDetailsModal = closeCourseDetailsModal;
window.deleteLearner = deleteLearner;
window.deleteUser = deleteUser;