// Theme Toggle - White (default) to Colorful (light-mode)
const themeToggle = document.getElementById('themeToggle');
const body = document.body;

themeToggle.addEventListener('click', () => {
    body.classList.toggle('light-mode');
    const icon = themeToggle.querySelector('i');
    
    // Add transition effect
    body.style.transition = 'background 0.8s ease, color 0.8s ease';
    
    if (body.classList.contains('light-mode')) {
        // Colorful mode activated
        icon.classList.remove('fa-moon');
        icon.classList.add('fa-sun');
        localStorage.setItem('theme', 'light');
        
        // Add sparkle effect
        createSparkles();
    } else {
        // White mode (default)
        icon.classList.remove('fa-sun');
        icon.classList.add('fa-moon');
        localStorage.setItem('theme', 'white');
    }
});

// Create sparkle effect when theme changes to colorful
function createSparkles() {
    const sparkleCount = 20;
    const colors = ['#0ea5e9', '#14b8a6', '#3b82f6', '#60a5fa'];
    for (let i = 0; i < sparkleCount; i++) {
        const sparkle = document.createElement('div');
        sparkle.style.cssText = `
            position: fixed;
            width: ${Math.random() * 6 + 4}px;
            height: ${Math.random() * 6 + 4}px;
            background: ${colors[Math.floor(Math.random() * colors.length)]};
            border-radius: 50%;
            left: ${Math.random() * 100}%;
            top: ${Math.random() * 100}%;
            pointer-events: none;
            z-index: 9999;
            animation: sparkleFloat ${Math.random() * 2 + 1}s ease-out forwards;
            box-shadow: 0 0 10px currentColor;
        `;
        document.body.appendChild(sparkle);
        
        setTimeout(() => sparkle.remove(), 2000);
    }
}

// Add sparkle animation
const sparkleStyle = document.createElement('style');
sparkleStyle.textContent = `
    @keyframes sparkleFloat {
        0% {
            opacity: 1;
            transform: translate(0, 0) scale(1);
        }
        100% {
            opacity: 0;
            transform: translate(${Math.random() * 200 - 100}px, ${Math.random() * 200 - 100}px) scale(0);
        }
    }
`;
document.head.appendChild(sparkleStyle);

// Load saved theme
window.addEventListener('DOMContentLoaded', () => {
    const savedTheme = localStorage.getItem('theme');
    if (savedTheme === 'light') {
        body.classList.add('light-mode');
        const icon = themeToggle.querySelector('i');
        icon.classList.remove('fa-moon');
        icon.classList.add('fa-sun');
    } else {
        // Default to white
        body.classList.remove('light-mode');
        const icon = themeToggle.querySelector('i');
        icon.classList.remove('fa-sun');
        icon.classList.add('fa-moon');
    }
    
    // Ensure navbar is visible on initial load
    const navbar = document.getElementById('navbar');
    if (navbar) {
        navbar.classList.add('navbar-visible');
        navbar.classList.remove('navbar-hidden');
    }
});

// Mobile Menu Toggle
const hamburger = document.getElementById('hamburger');
const navLinks = document.querySelector('.nav-links');

if (hamburger && navLinks) {
    hamburger.addEventListener('click', () => {
        hamburger.classList.toggle('active');
        navLinks.classList.toggle('active');
        document.body.style.overflow = navLinks.classList.contains('active') ? 'hidden' : '';
    });
    
    // Close menu when clicking nav links
    document.querySelectorAll('.nav-link, .nav-links .btn').forEach(link => {
        link.addEventListener('click', () => {
            hamburger.classList.remove('active');
            navLinks.classList.remove('active');
            document.body.style.overflow = '';
        });
    });
    
    // Close menu when clicking outside
    document.addEventListener('click', (e) => {
        if (navLinks.classList.contains('active') && 
            !navLinks.contains(e.target) && 
            !hamburger.contains(e.target)) {
            hamburger.classList.remove('active');
            navLinks.classList.remove('active');
            document.body.style.overflow = '';
        }
    });
}

// Scroll Progress Indicator
window.addEventListener('scroll', () => {
    const scrollProgress = document.querySelector('.scroll-progress');
    if (scrollProgress) {
        const scrollHeight = document.documentElement.scrollHeight - window.innerHeight;
        const scrolled = (window.pageYOffset / scrollHeight) * 100;
        scrollProgress.style.width = scrolled + '%';
    }
}, { passive: true });

// Magnetic Button Effect (Desktop Only)
function initMagneticButtons() {
    if (window.innerWidth > 968) {
        document.querySelectorAll('.btn').forEach(btn => {
            btn.addEventListener('mousemove', (e) => {
                const rect = btn.getBoundingClientRect();
                const x = e.clientX - rect.left - rect.width / 2;
                const y = e.clientY - rect.top - rect.height / 2;
                
                btn.style.transform = `translate(${x * 0.2}px, ${y * 0.2}px) scale(1.05)`;
            });
            
            btn.addEventListener('mouseleave', () => {
                btn.style.transform = 'translate(0, 0) scale(1)';
            });
        });
    }
}

// Initialize magnetic buttons after DOM loads
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initMagneticButtons);
} else {
    initMagneticButtons();
}

// Navigation Scroll Effect - Hide on scroll down, show on scroll up
let lastScrollTop = 0;
let scrollTimeout = null;
const scrollThreshold = 10; // Minimum scroll distance to trigger hide/show

window.addEventListener('scroll', () => {
    const navbar = document.getElementById('navbar');
    if (!navbar) return;
    
    const scrollTop = window.pageYOffset || document.documentElement.scrollTop;
    
    // Add scrolled class for styling when scrolled past 50px
    if (scrollTop > 50) {
        navbar.classList.add('scrolled');
    } else {
        navbar.classList.remove('scrolled');
    }
    
    // Hide/show navbar based on scroll direction
    if (Math.abs(lastScrollTop - scrollTop) < scrollThreshold) {
        // Scroll distance too small, don't do anything
        return;
    }
    
    if (scrollTop > lastScrollTop && scrollTop > 100) {
        // Scrolling down - hide navbar
        navbar.classList.remove('navbar-visible');
        navbar.classList.add('navbar-hidden');
    } else if (scrollTop < lastScrollTop) {
        // Scrolling up - show navbar
        navbar.classList.remove('navbar-hidden');
        navbar.classList.add('navbar-visible');
    }
    
    // Always show navbar at the top
    if (scrollTop <= 50) {
        navbar.classList.remove('navbar-hidden');
        navbar.classList.add('navbar-visible');
    }
    
    lastScrollTop = scrollTop;
    
    // Clear any existing timeout
    if (scrollTimeout) {
        clearTimeout(scrollTimeout);
    }
    
    // If user stops scrolling, show navbar after a short delay
    scrollTimeout = setTimeout(() => {
        if (scrollTop > 100) {
            navbar.classList.remove('navbar-hidden');
            navbar.classList.add('navbar-visible');
        }
    }, 1500);
}, { passive: true });

// Hamburger Menu - Removed (no longer needed)

// Smooth Scroll with Offset
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            const offset = 80;
            const targetPosition = target.offsetTop - offset;
            window.scrollTo({
                top: targetPosition,
                behavior: 'smooth'
            });
        }
    });
});

// Optimized Parallax Effect - GPU Accelerated
let lastScrollY = 0;
let ticking = false;
const parallaxCache = {
    shapes: [],
    heroContent: null,
    heroImage: null,
    sections: []
};

// Cache elements once
function initParallax() {
    parallaxCache.shapes = Array.from(document.querySelectorAll('.floating-shape'));
    parallaxCache.heroContent = document.querySelector('.hero-content');
    parallaxCache.heroImage = document.querySelector('.hero-image');
    
    // Add GPU acceleration classes
    parallaxCache.shapes.forEach(shape => {
        shape.style.willChange = 'transform';
        shape.style.transform = 'translate3d(0, 0, 0)';
    });
    
    if (parallaxCache.heroContent) {
        parallaxCache.heroContent.style.willChange = 'transform, opacity';
    }
    if (parallaxCache.heroImage) {
        parallaxCache.heroImage.style.willChange = 'transform';
    }
}

function updateParallax() {
    const scrollY = window.pageYOffset;
    const delta = scrollY - lastScrollY;
    
    // Only update if scroll changed significantly (throttle)
    if (Math.abs(delta) < 1 && scrollY === lastScrollY) {
        ticking = false;
        return;
    }
    
    // Floating shapes - minimal parallax
    parallaxCache.shapes.forEach((shape, index) => {
        const speed = 0.3 + (index * 0.1);
        const y = scrollY * speed;
        shape.style.transform = `translate3d(0, ${y}px, 0)`;
    });
    
    // Hero parallax - only in viewport
    if (scrollY < window.innerHeight) {
        const heroProgress = scrollY / window.innerHeight;
        
        if (parallaxCache.heroContent) {
            const y = scrollY * 0.2;
            // Keep opacity at 1 until significant scroll (don't fade on initial load)
            const opacity = Math.max(0.7, 1 - heroProgress * 0.3);
            parallaxCache.heroContent.style.transform = `translate3d(0, ${y}px, 0)`;
            parallaxCache.heroContent.style.opacity = opacity;
        }
        
        if (parallaxCache.heroImage) {
            const y = scrollY * 0.15;
            const scale = Math.max(0.95, 1 - heroProgress * 0.05);
            parallaxCache.heroImage.style.transform = `translate3d(0, ${y}px, 0) scale(${scale})`;
        }
    } else {
        // Reset opacity when scrolled past hero
        if (parallaxCache.heroContent) {
            parallaxCache.heroContent.style.opacity = '0.5';
        }
    }
    
    lastScrollY = scrollY;
    ticking = false;
}

function requestParallaxTick() {
    if (!ticking) {
        requestAnimationFrame(updateParallax);
        ticking = true;
    }
}

// Initialize on load
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        initParallax();
        // Ensure hero is visible on load
        const heroContent = document.querySelector('.hero-content');
        const heroImage = document.querySelector('.hero-image');
        if (heroContent) {
            heroContent.style.opacity = '1';
        }
        if (heroImage) {
            heroImage.style.opacity = '1';
        }
    });
} else {
    initParallax();
    // Ensure hero is visible on load
    const heroContent = document.querySelector('.hero-content');
    const heroImage = document.querySelector('.hero-image');
    if (heroContent) {
        heroContent.style.opacity = '1';
    }
    if (heroImage) {
        heroImage.style.opacity = '1';
    }
}

window.addEventListener('scroll', requestParallaxTick, { passive: true });

// Create Particles
function createParticles() {
    const particlesContainer = document.getElementById('particles');
    const particleCount = 50;
    
    for (let i = 0; i < particleCount; i++) {
        const particle = document.createElement('div');
        particle.className = 'particle';
        particle.style.cssText = `
            position: absolute;
            width: ${Math.random() * 4 + 1}px;
            height: ${Math.random() * 4 + 1}px;
            background: rgba(255, 255, 255, 0.5);
            border-radius: 50%;
            left: ${Math.random() * 100}%;
            top: ${Math.random() * 100}%;
            animation: particleFloat ${Math.random() * 10 + 5}s infinite ease-in-out;
            animation-delay: ${Math.random() * 5}s;
        `;
        particlesContainer.appendChild(particle);
    }
}

// Add particle animation to CSS
const style = document.createElement('style');
style.textContent = `
    @keyframes particleFloat {
        0%, 100% { transform: translate(0, 0); opacity: 0; }
        10% { opacity: 1; }
        90% { opacity: 1; }
        100% { transform: translate(${Math.random() * 100 - 50}px, ${Math.random() * 100 - 50}px); opacity: 0; }
    }
`;
document.head.appendChild(style);

createParticles();

// AOS (Animate On Scroll) Implementation - Ultra-Smooth with Intersection Observer
const aosObserverOptions = {
    threshold: [0, 0.1, 0.25, 0.5, 0.75, 1],
    rootMargin: '0px 0px -50px 0px'
};

const aosObserver = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            requestAnimationFrame(() => {
                entry.target.classList.add('aos-animate');
                // Remove will-change after animation completes to free GPU resources
                setTimeout(() => {
                    entry.target.style.willChange = 'auto';
                }, 800);
            });
        } else {
            // Reset animation when scrolling up past element
            if (entry.boundingClientRect.top < 0 && entry.target.classList.contains('aos-animate')) {
                requestAnimationFrame(() => {
                    entry.target.classList.remove('aos-animate');
                    entry.target.style.willChange = 'transform, opacity';
                });
            }
        }
    });
}, aosObserverOptions);

// Observe all elements with data-aos - GPU accelerated
document.querySelectorAll('[data-aos]').forEach(element => {
    // GPU acceleration hints for smooth performance
    element.style.willChange = 'transform, opacity';
    element.style.backfaceVisibility = 'hidden';
    aosObserver.observe(element);
});

// Counter Animation
function animateCounter(element) {
    const target = parseInt(element.getAttribute('data-target'));
    const duration = 2000;
    const increment = target / (duration / 16);
    let current = 0;
    
    const updateCounter = () => {
        current += increment;
        if (current < target) {
            element.textContent = Math.floor(current).toLocaleString();
            requestAnimationFrame(updateCounter);
        } else {
            element.textContent = target.toLocaleString() + (element.textContent.includes('%') ? '' : '');
        }
    };
    
    updateCounter();
}

// Observe counters
const counterObserver = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting && !entry.target.classList.contains('counted')) {
            entry.target.classList.add('counted');
            animateCounter(entry.target);
        }
    });
}, { threshold: 0.5 });

document.querySelectorAll('.counter').forEach(counter => {
    counterObserver.observe(counter);
});

// OCR Demo Upload
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const demoResults = document.getElementById('demoResults');
const previewImage = document.getElementById('previewImage');
const extractedText = document.getElementById('extractedText');

// Drag and Drop
uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.style.borderColor = 'var(--primary)';
    uploadArea.style.background = 'rgba(99, 102, 241, 0.1)';
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.style.borderColor = 'rgba(255, 255, 255, 0.3)';
    uploadArea.style.background = 'transparent';
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.style.borderColor = 'rgba(255, 255, 255, 0.3)';
    uploadArea.style.background = 'transparent';
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFileUpload(files[0]);
    }
});

// File Input Change
fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleFileUpload(e.target.files[0]);
    }
});

// Handle File Upload
function handleFileUpload(file) {
    if (!file.type.startsWith('image/')) {
        alert('Please upload an image file');
        return;
    }
    
    // Show loading state
    uploadArea.style.display = 'none';
    demoResults.style.display = 'grid';
    
    // Preview image
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImage.src = e.target.result;
        
        // Simulate OCR processing
        processOCR(file);
    };
    reader.readAsDataURL(file);
}

// Process OCR (Send to backend)
async function processOCR(file) {
    extractedText.textContent = 'Processing image...';
    
    try {
        const formData = new FormData();
        formData.append('image', file);
        
        const response = await fetch('/api/ocr', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error('OCR processing failed');
        }
        
        const data = await response.json();
        
        // Animate accuracy meter
        setTimeout(() => {
            const accuracyFill = document.getElementById('accuracyFill');
            const accuracyText = document.getElementById('accuracyText');
            
            accuracyFill.style.width = data.accuracy + '%';
            accuracyText.textContent = data.accuracy + '%';
            
            // Display extracted text with typewriter effect
            typewriterEffect(data.text, extractedText);
        }, 500);
        
    } catch (error) {
        console.error('Error:', error);
        extractedText.textContent = 'Error processing image. Please try again.';
        
        // For demo purposes, show sample text if API fails
        setTimeout(() => {
            const sampleText = `Sample Extracted Text:\n\nThis is a demonstration of our OCR technology.\nThe system can accurately extract handwritten text\nfrom images with 98.5% accuracy.\n\nFeatures:\n- Multi-language support\n- Fast processing (< 2 seconds)\n- High accuracy recognition\n- Support for various handwriting styles`;
            
            typewriterEffect(sampleText, extractedText);
        }, 1000);
    }
}

// Typewriter Effect
function typewriterEffect(text, element, speed = 20) {
    element.textContent = '';
    let i = 0;
    
    const type = () => {
        if (i < text.length) {
            element.textContent += text.charAt(i);
            i++;
            setTimeout(type, speed);
        }
    };
    
    type();
}

// Reset Demo
function resetDemo() {
    uploadArea.style.display = 'block';
    demoResults.style.display = 'none';
    fileInput.value = '';
    previewImage.src = '';
    extractedText.textContent = '';
    
    const accuracyFill = document.getElementById('accuracyFill');
    accuracyFill.style.width = '0%';
}

// Contact Form - Fixed to work properly
function initContactForm() {
    const contactForm = document.querySelector('.contact-form');
    if (!contactForm) {
        // Retry if form not loaded yet
        setTimeout(initContactForm, 100);
        return;
    }
    
    contactForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const formData = new FormData(e.target);
    const submitBtn = e.target.querySelector('button[type="submit"]');
        if (!submitBtn) return;
        
    const originalText = submitBtn.innerHTML;
    
    submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Sending...';
    submitBtn.disabled = true;
    
    try {
        // Simulate form submission
        await new Promise(resolve => setTimeout(resolve, 2000));
        
        submitBtn.innerHTML = '<i class="fas fa-check"></i> Message Sent!';
        submitBtn.style.background = 'linear-gradient(135deg, #10b981, #059669)';
        
        setTimeout(() => {
            submitBtn.innerHTML = originalText;
            submitBtn.style.background = '';
            submitBtn.disabled = false;
            e.target.reset();
        }, 3000);
        
    } catch (error) {
            console.error('Form submission error:', error);
        submitBtn.innerHTML = '<i class="fas fa-times"></i> Error';
        submitBtn.style.background = 'linear-gradient(135deg, #ef4444, #dc2626)';
        
        setTimeout(() => {
            submitBtn.innerHTML = originalText;
            submitBtn.style.background = '';
            submitBtn.disabled = false;
        }, 3000);
    }
});
}

// Initialize contact form when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initContactForm);
} else {
    initContactForm();
}

// ========================================
// ULTRA-SMOOTH SCROLL ANIMATIONS (100% Lag-Free)
// ========================================
let scrollAnimationCache = {
    elements: [],
    ticking: false,
    lastScrollY: 0,
    scrollDirection: 1
};

// Initialize scroll animations with GPU acceleration
function initSmoothScrollAnimations() {
    // Cache all animated elements
    const animatedElements = document.querySelectorAll(
        'section, .section-title, .section-subtitle, .feature-card, .stat-card, .contact-card, .demo-container, .floating-card, .about-text, .about-image, .contact-info, .contact-form'
    );
    
    scrollAnimationCache.elements = Array.from(animatedElements).map(el => ({
        element: el,
        type: getElementType(el),
        rect: null,
        hasAnimated: false,
        isVisible: false,
        opacity: 0,
        transform: 'translate3d(0, 50px, 0) scale(0.95)'
    }));
    
    // Set initial styles with GPU acceleration
    scrollAnimationCache.elements.forEach(item => {
        const el = item.element;
        
        // Skip hero section
        if (el.id === 'home' || el.classList.contains('hero-section')) {
            item.hasAnimated = true;
            item.isVisible = true;
            return;
        }
        
        // GPU acceleration hints
        el.style.willChange = 'transform, opacity';
        el.style.transform = 'translate3d(0, 50px, 0) scale(0.95)';
        el.style.opacity = '0';
        el.style.transition = 'transform 0.8s cubic-bezier(0.16, 1, 0.3, 1), opacity 0.8s cubic-bezier(0.16, 1, 0.3, 1)';
        el.style.backfaceVisibility = 'hidden';
        el.style.perspective = '1000px';
    });
    
    // Use Intersection Observer for better performance
    const observerOptions = {
        threshold: [0, 0.1, 0.25, 0.5, 0.75, 1],
        rootMargin: '50px 0px -100px 0px'
    };
    
    const scrollObserver = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            const item = scrollAnimationCache.elements.find(item => item.element === entry.target);
            if (!item) return;
            
            item.isVisible = entry.isIntersecting;
            item.rect = entry.boundingClientRect;
            
            if (entry.isIntersecting && !item.hasAnimated) {
                animateElementIn(item, entry.intersectionRatio);
            } else if (!entry.isIntersecting && entry.boundingClientRect.top < 0) {
                // Element scrolled past viewport - reset for re-animation
                animateElementOut(item);
            }
        });
    }, observerOptions);
    
    // Observe all elements
    scrollAnimationCache.elements.forEach(item => {
        scrollObserver.observe(item.element);
    });
    
    // Additional smooth scroll-based parallax for titles and subtitles
    window.addEventListener('scroll', updateScrollBasedAnimations, { passive: true });
}

function getElementType(element) {
    if (element.classList.contains('section-title')) return 'title';
    if (element.classList.contains('section-subtitle')) return 'subtitle';
    if (element.classList.contains('feature-card')) return 'card';
    if (element.classList.contains('stat-card')) return 'stat';
    if (element.classList.contains('contact-card')) return 'contact';
    if (element.tagName === 'SECTION') return 'section';
    return 'default';
}

function animateElementIn(item, ratio) {
    const el = item.element;
    const delay = getStaggerDelay(item);
    
    requestAnimationFrame(() => {
        setTimeout(() => {
            el.style.transform = 'translate3d(0, 0, 0) scale(1)';
            el.style.opacity = '1';
            item.hasAnimated = true;
            
            // Remove will-change after animation to free GPU resources
            setTimeout(() => {
                el.style.willChange = 'auto';
            }, 1000);
            
            // Add aos-animate class if element has data-aos
            if (el.hasAttribute('data-aos')) {
                el.classList.add('aos-animate');
            }
        }, delay);
    });
}

function animateElementOut(item) {
    // Only reset if scrolling down past the element
    if (scrollAnimationCache.scrollDirection > 0) {
        const el = item.element;
        el.style.willChange = 'transform, opacity';
        el.style.transform = 'translate3d(0, -30px, 0) scale(0.98)';
        el.style.opacity = '0';
        item.hasAnimated = false;
    }
}

function getStaggerDelay(item) {
    // Stagger animations based on element type and position
    if (item.type === 'title') return 0;
    if (item.type === 'subtitle') return 100;
    if (item.type === 'card' || item.type === 'stat') {
        const index = Array.from(item.element.parentElement.children).indexOf(item.element);
        return index * 80;
    }
    return 150;
}

// Smooth scroll-based parallax for titles and subtitles
function updateScrollBasedAnimations() {
    if (scrollAnimationCache.ticking) return;
    
    scrollAnimationCache.ticking = true;
    requestAnimationFrame(() => {
        const scrollY = window.pageYOffset;
        const windowHeight = window.innerHeight;
        
        // Update scroll direction
        scrollAnimationCache.scrollDirection = scrollY > scrollAnimationCache.lastScrollY ? 1 : -1;
        scrollAnimationCache.lastScrollY = scrollY;
        
        // Smooth parallax for titles and subtitles
        scrollAnimationCache.elements.forEach(item => {
            if (item.type !== 'title' && item.type !== 'subtitle') return;
            if (!item.rect) return;
            
            const rect = item.rect;
            if (rect.top < windowHeight * 1.2 && rect.bottom > -windowHeight * 0.2) {
                const centerY = rect.top + rect.height / 2;
                const viewportCenter = windowHeight / 2;
                const distance = centerY - viewportCenter;
                const maxDistance = windowHeight;
                
                // Smooth parallax effect
                const parallax = (distance / maxDistance) * 30;
                const opacity = Math.max(0.3, 1 - Math.abs(distance) / (windowHeight * 0.5));
                
                item.element.style.transform = `translate3d(0, ${parallax * 0.3}px, 0)`;
                item.element.style.opacity = opacity;
            }
        });
        
        scrollAnimationCache.ticking = false;
    });
}

// Initialize on load
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        setTimeout(initSmoothScrollAnimations, 100);
    });
} else {
    setTimeout(initSmoothScrollAnimations, 100);
}

// Smooth Scroll-Driven Carousel - Two Rows Opposite Directions
let carouselCache = {
    grid1: null,
    grid2: null,
    wrapper: null,
    section: null,
    container: null,
    lastScroll: 0,
    ticking: false,
    currentX1: 0,
    targetX1: 0,
    currentX2: 0,
    targetX2: 0
};

function initCarousel() {
    carouselCache.grid1 = document.getElementById('featuresGrid1');
    carouselCache.grid2 = document.getElementById('featuresGrid2');
    carouselCache.wrapper = document.querySelector('.features-wrapper');
    carouselCache.section = document.getElementById('features');
    
    if (!carouselCache.grid1 || !carouselCache.grid2 || !carouselCache.section) return;
    
    carouselCache.container = carouselCache.section.querySelector('.container');
    
    // GPU acceleration for both grids
    carouselCache.grid1.style.willChange = 'transform';
    carouselCache.grid1.style.backfaceVisibility = 'hidden';
    
    carouselCache.grid2.style.willChange = 'transform';
    carouselCache.grid2.style.backfaceVisibility = 'hidden';
    
    // Set initial positions (outwards - at edges)
    const containerWidth = carouselCache.container ? carouselCache.container.offsetWidth : window.innerWidth;
    const grid2Width = carouselCache.grid2.scrollWidth;
    
    carouselCache.currentX1 = 0; // Row 1 starts at left
    carouselCache.currentX2 = containerWidth - grid2Width; // Row 2 starts at right
    
    carouselCache.grid1.style.transform = `translate3d(${carouselCache.currentX1}px, 0, 0)`;
    carouselCache.grid2.style.transform = `translate3d(${carouselCache.currentX2}px, 0, 0)`;
    
    // Calculate initial state
    updateCarousel();
}

function updateCarousel() {
    if (!carouselCache.grid1 || !carouselCache.grid2 || !carouselCache.section) {
        carouselCache.ticking = false;
        return;
    }
    
    const scrollY = window.pageYOffset;
    const sectionRect = carouselCache.section.getBoundingClientRect();
    const sectionTop = sectionRect.top + scrollY;
    const sectionHeight = sectionRect.height;
    const windowHeight = window.innerHeight;
    const containerWidth = carouselCache.container ? carouselCache.container.offsetWidth : window.innerWidth;
    
    // Check if mobile
    const isMobile = window.innerWidth <= 968;
    
    // Calculate progress (0 to 1) when section is in viewport
    let progress = 0;
    const viewportTop = scrollY;
    const viewportBottom = scrollY + windowHeight;
    const sectionBottom = sectionTop + sectionHeight;
    
    if (viewportBottom > sectionTop && viewportTop < sectionBottom) {
        // Section is visible - calculate progress based on scroll position
        const scrollInSection = Math.max(0, scrollY - sectionTop);
        // On mobile, use more of the section height for slower scrolling (80% vs 60%)
        const sectionScrollable = sectionHeight * (isMobile ? 0.8 : 0.6);
        progress = Math.min(1, Math.max(0, scrollInSection / sectionScrollable));
    } else if (viewportTop >= sectionBottom) {
        // Past the section
        progress = 1;
    } else {
        // Before the section
        progress = 0;
    }
    
    // Calculate max movement distance
    const grid1Width = carouselCache.grid1.scrollWidth;
    const grid2Width = carouselCache.grid2.scrollWidth;
    // On mobile, reduce movement distance to 15% (instead of 30%) for slower, more controlled scrolling
    const maxMovement = containerWidth * (isMobile ? 0.15 : 0.3);
    
    // Calculate positions for inwards/outwards movement
    // Progress 0 = cards at edges (outwards), Progress 1 = cards at center (inwards)
    // Row 1 starts from left, moves right (inwards)
    // Row 2 starts from right, moves left (inwards)
    
    const startOffset1 = 0; // Row 1 starts at left edge
    const startOffset2 = containerWidth - grid2Width; // Row 2 starts at right edge
    
    // Calculate target positions
    // When progress = 0: cards at edges (outwards)
    // When progress = 1: cards moved towards center (inwards)
    carouselCache.targetX1 = startOffset1 + (maxMovement * progress);
    carouselCache.targetX2 = startOffset2 - (maxMovement * progress);
    
    // Smooth interpolation (easing) - slower on mobile (0.08 vs 0.15) for smoother scrolling
    const ease = isMobile ? 0.08 : 0.15;
    
    // Update row 1 (moves from left towards center)
    const diff1 = carouselCache.targetX1 - carouselCache.currentX1;
    if (Math.abs(diff1) > 0.5) {
        carouselCache.currentX1 += diff1 * ease;
        carouselCache.grid1.style.transform = `translate3d(${carouselCache.currentX1}px, 0, 0)`;
    }
    
    // Update row 2 (moves from right towards center)
    const diff2 = carouselCache.targetX2 - carouselCache.currentX2;
    if (Math.abs(diff2) > 0.5) {
        carouselCache.currentX2 += diff2 * ease;
        carouselCache.grid2.style.transform = `translate3d(${carouselCache.currentX2}px, 0, 0)`;
    }
    
    carouselCache.ticking = false;
}

function requestCarouselTick() {
    if (!carouselCache.ticking) {
        requestAnimationFrame(updateCarousel);
        carouselCache.ticking = true;
    }
}

// Initialize carousel
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        setTimeout(initCarousel, 100);
    });
} else {
    setTimeout(initCarousel, 100);
}

// Update on scroll
window.addEventListener('scroll', requestCarouselTick, { passive: true });

// Update on resize
let resizeTimeout;
window.addEventListener('resize', () => {
    clearTimeout(resizeTimeout);
    resizeTimeout = setTimeout(() => {
        carouselCache.currentX1 = 0;
        carouselCache.currentX2 = 0;
        updateCarousel();
    }, 150);
}, { passive: true });

// Feature cards hover effect
document.querySelectorAll('.feature-card').forEach(card => {
    card.addEventListener('mouseenter', function() {
        this.style.transform = 'translate3d(0, -10px, 0) scale(1.02)';
    });
    
    card.addEventListener('mouseleave', function() {
        this.style.transform = 'translate3d(0, 0, 0) scale(1)';
    });
});

// Add ripple effect to buttons
document.querySelectorAll('.btn').forEach(button => {
    button.addEventListener('click', function(e) {
        const ripple = document.createElement('span');
        ripple.style.cssText = `
            position: absolute;
            border-radius: 50%;
            background: rgba(255, 255, 255, 0.5);
            width: 100px;
            height: 100px;
            margin-left: -50px;
            margin-top: -50px;
            animation: ripple 0.6s;
            pointer-events: none;
        `;
        
        const rect = button.getBoundingClientRect();
        ripple.style.left = e.clientX - rect.left + 'px';
        ripple.style.top = e.clientY - rect.top + 'px';
        
        button.style.position = 'relative';
        button.style.overflow = 'hidden';
        button.appendChild(ripple);
        
        setTimeout(() => ripple.remove(), 600);
    });
});

// Add ripple animation
const rippleStyle = document.createElement('style');
rippleStyle.textContent = `
    @keyframes ripple {
        from {
            transform: scale(0);
            opacity: 1;
        }
        to {
            transform: scale(2);
            opacity: 0;
        }
    }
`;
document.head.appendChild(rippleStyle);

// ========================================
// ABOUT & CONTACT SECTION ANIMATIONS
// ========================================
function initSectionAnimations() {
    // About section elements
    const aboutText = document.querySelector('.about-text');
    const aboutImage = document.querySelector('.about-image');
    const aboutFeatures = document.querySelector('.about-features');
    const statsContainer = document.querySelector('.stats-container');
    
    // Contact section elements
    const contactInfo = document.querySelector('.contact-info');
    const contactForm = document.querySelector('.contact-form');
    
    // Create intersection observer
    const observerOptions = {
        threshold: 0.1, // Lower threshold to trigger earlier
        rootMargin: '0px 0px 0px 0px' // Changed to trigger immediately when visible
    };
    
    const sectionObserver = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('animate-in');
            }
        });
    }, observerOptions);
    
    // Observe all elements
    if (aboutText) sectionObserver.observe(aboutText);
    if (aboutImage) sectionObserver.observe(aboutImage);
    if (aboutFeatures) sectionObserver.observe(aboutFeatures);
    if (statsContainer) sectionObserver.observe(statsContainer);
    if (contactInfo) sectionObserver.observe(contactInfo);
    if (contactForm) {
        sectionObserver.observe(contactForm);
        // Also make form visible immediately if it's already in viewport
        const rect = contactForm.getBoundingClientRect();
        if (rect.top < window.innerHeight && rect.bottom > 0) {
            contactForm.classList.add('animate-in');
        }
    }
    
    // Fallback: Make contact form visible after a delay if observer doesn't trigger
    setTimeout(() => {
        if (contactForm && !contactForm.classList.contains('animate-in')) {
            contactForm.classList.add('animate-in');
        }
        if (contactInfo && !contactInfo.classList.contains('animate-in')) {
            contactInfo.classList.add('animate-in');
        }
    }, 1000);
}

// Initialize on load
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        setTimeout(initSectionAnimations, 100);
    });
} else {
    setTimeout(initSectionAnimations, 100);
}

// ========================================
// INTERACTIVE BACKGROUND WITH CURSOR GLOW
// ========================================
// Create cursor glow element
const cursorGlow = document.createElement('div');
cursorGlow.className = 'cursor-glow';
document.body.appendChild(cursorGlow);

let mouseX = 0;
let mouseY = 0;
let glowX = 0;
let glowY = 0;

// Track mouse movement
document.addEventListener('mousemove', (e) => {
    mouseX = e.clientX;
    mouseY = e.clientY;
});

// Smooth cursor glow animation
function animateCursorGlow() {
    // Smooth interpolation
    glowX += (mouseX - glowX) * 0.1;
    glowY += (mouseY - glowY) * 0.1;
    
    cursorGlow.style.transform = `translate(${glowX - 150}px, ${glowY - 150}px)`;
    cursorGlow.style.opacity = '1';
    
    requestAnimationFrame(animateCursorGlow);
}

// Start cursor glow animation on desktop only
if (window.innerWidth > 968) {
    animateCursorGlow();
}

// ========================================
// ENHANCED INTERACTIVE PARTICLES
// ========================================
function createInteractiveParticles() {
    const particlesContainer = document.getElementById('particles');
    if (!particlesContainer) return;
    
    // Clear existing particles
    particlesContainer.innerHTML = '';
    
    const particleCount = window.innerWidth > 968 ? 80 : 40;
    const colors = ['rgba(14, 165, 233, 0.6)', 'rgba(20, 184, 166, 0.6)', 'rgba(59, 130, 246, 0.6)', 'rgba(96, 165, 250, 0.6)'];
    
    for (let i = 0; i < particleCount; i++) {
        const particle = document.createElement('div');
        particle.className = 'particle';
        
        const size = Math.random() * 6 + 2;
        const color = colors[Math.floor(Math.random() * colors.length)];
        const duration = Math.random() * 20 + 15;
        const delay = Math.random() * 5;
        
        particle.style.cssText = `
            width: ${size}px;
            height: ${size}px;
            background: ${color};
            left: ${Math.random() * 100}%;
            top: ${Math.random() * 100}%;
            animation: particleFloatInteractive ${duration}s infinite ease-in-out;
            animation-delay: ${delay}s;
            box-shadow: 0 0 ${size * 2}px ${color};
        `;
        
        particlesContainer.appendChild(particle);
    }
}

// Enhanced particle animation
const particleAnimationStyle = document.createElement('style');
particleAnimationStyle.textContent = `
    @keyframes particleFloatInteractive {
        0%, 100% {
            transform: translate(0, 0) scale(1);
            opacity: 0;
        }
        10% {
            opacity: 1;
        }
        90% {
            opacity: 1;
        }
        100% {
            transform: translate(${Math.random() * 200 - 100}px, ${Math.random() * 200 - 100}px) scale(${Math.random() * 1.5 + 0.5});
            opacity: 0;
        }
    }
`;
document.head.appendChild(particleAnimationStyle);

// Replace old particles with new interactive ones
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', createInteractiveParticles);
} else {
    createInteractiveParticles();
}

// ========================================
// INFINITE CAROUSEL FOR FEATURE CARDS
// ========================================
function setupInfiniteCarousel() {
    const grid1 = document.getElementById('featuresGrid1');
    const grid2 = document.getElementById('featuresGrid2');
    
    if (!grid1 || !grid2) return;
    
    // Clone cards for infinite effect
    const cards1 = Array.from(grid1.children);
    const cards2 = Array.from(grid2.children);
    
    // Duplicate cards for seamless loop
    cards1.forEach(card => {
        const clone = card.cloneNode(true);
        grid1.appendChild(clone);
    });
    
    cards2.forEach(card => {
        const clone = card.cloneNode(true);
        grid2.appendChild(clone);
    });
}

// Initialize infinite carousel
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', setupInfiniteCarousel);
} else {
    setupInfiniteCarousel();
}

// ========================================
// SCROLL-TRIGGERED ANIMATIONS
// ========================================
// Add bounce animation to elements on scroll
const bounceObserver = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            entry.target.style.animation = 'bounceIn 0.8s cubic-bezier(0.68, -0.55, 0.265, 1.55)';
        }
    });
}, {
    threshold: 0.3
});

// Observe demo stats, about features, contact cards
document.querySelectorAll('.stat-card, .about-feature, .contact-card').forEach(el => {
    bounceObserver.observe(el);
});

// Add bounce animation
const bounceStyle = document.createElement('style');
bounceStyle.textContent = `
    @keyframes bounceIn {
        0% {
            transform: scale(0.3) translateY(50px);
            opacity: 0;
        }
        50% {
            transform: scale(1.05) translateY(-10px);
        }
        70% {
            transform: scale(0.9) translateY(5px);
        }
        100% {
            transform: scale(1) translateY(0);
            opacity: 1;
        }
    }
`;
document.head.appendChild(bounceStyle);

// ========================================
// PARALLAX EFFECT FOR SECTIONS
// ========================================
window.addEventListener('scroll', () => {
    const scrolled = window.pageYOffset;
    
    // Parallax for section backgrounds
    document.querySelectorAll('section').forEach((section, index) => {
        const speed = 0.5 + (index * 0.1);
        const yPos = -(scrolled * speed / 10);
        
        if (section.style) {
            section.style.backgroundPosition = `50% ${yPos}px`;
        }
    });
}, { passive: true });

// ========================================
// INTERACTIVE CARD TILT ON MOUSE MOVE
// ========================================
function addCardTiltEffect() {
    if (window.innerWidth <= 968) return; // Disable on mobile
    
    document.querySelectorAll('.feature-card, .stat-card, .contact-card, .floating-card').forEach(card => {
        card.addEventListener('mousemove', (e) => {
            const rect = card.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            
            const centerX = rect.width / 2;
            const centerY = rect.height / 2;
            
            const rotateX = (y - centerY) / 10;
            const rotateY = (centerX - x) / 10;
            
            card.style.transform = `perspective(1000px) rotateX(${rotateX}deg) rotateY(${rotateY}deg) translateY(-10px) scale(1.05)`;
        });
        
        card.addEventListener('mouseleave', () => {
            card.style.transform = 'perspective(1000px) rotateX(0) rotateY(0) translateY(0) scale(1)';
        });
    });
}

// Initialize tilt effect
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', addCardTiltEffect);
} else {
    addCardTiltEffect();
}

console.log('✨ Interactive background and animations loaded!');