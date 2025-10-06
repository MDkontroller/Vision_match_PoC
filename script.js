// DroneLocator AI Website JavaScript

// Global variables
let map;
let demoRunning = false;
let searchMarkers = [];
let confidenceChart;
let performanceChart;

// DOM Content Loaded
document.addEventListener('DOMContentLoaded', function() {
    initializeNavigation();
    initializeTabs();
    initializeMap();
    initializeCharts();
    initializeDemoControls();
    initializeAnimations();
});

// Navigation functionality
function initializeNavigation() {
    const hamburger = document.querySelector('.hamburger');
    const navMenu = document.querySelector('.nav-menu');
    const navLinks = document.querySelectorAll('.nav-link');

    // Mobile menu toggle
    if (hamburger) {
        hamburger.addEventListener('click', () => {
            hamburger.classList.toggle('active');
            navMenu.classList.toggle('active');
        });
    }

    // Smooth scrolling for navigation links
    navLinks.forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            const targetId = link.getAttribute('href');
            const targetSection = document.querySelector(targetId);
            
            if (targetSection) {
                const offsetTop = targetSection.offsetTop - 70; // Account for fixed nav
                window.scrollTo({
                    top: offsetTop,
                    behavior: 'smooth'
                });
            }

            // Close mobile menu if open
            if (navMenu.classList.contains('active')) {
                hamburger.classList.remove('active');
                navMenu.classList.remove('active');
            }
        });
    });

    // Active nav link highlighting
    window.addEventListener('scroll', () => {
        let current = '';
        const sections = document.querySelectorAll('section');
        
        sections.forEach(section => {
            const sectionTop = section.offsetTop - 100;
            const sectionHeight = section.clientHeight;
            
            if (pageYOffset >= sectionTop && pageYOffset < sectionTop + sectionHeight) {
                current = section.getAttribute('id');
            }
        });

        navLinks.forEach(link => {
            link.classList.remove('active');
            if (link.getAttribute('href') === '#' + current) {
                link.classList.add('active');
            }
        });
    });
}

// Tab functionality for technology section
function initializeTabs() {
    const tabButtons = document.querySelectorAll('.tab-button');
    const tabPanels = document.querySelectorAll('.tab-panel');

    tabButtons.forEach(button => {
        button.addEventListener('click', () => {
            const targetTab = button.getAttribute('data-tab');
            
            // Remove active class from all buttons and panels
            tabButtons.forEach(btn => btn.classList.remove('active'));
            tabPanels.forEach(panel => panel.classList.remove('active'));
            
            // Add active class to clicked button and corresponding panel
            button.classList.add('active');
            const targetPanel = document.getElementById(targetTab);
            if (targetPanel) {
                targetPanel.classList.add('active');
            }
        });
    });
}

// Map initialization
function initializeMap() {
    const mapContainer = document.getElementById('map-container');
    if (!mapContainer) return;

    // Initialize Leaflet map
    map = L.map('map-container').setView([40.7829, -73.9654], 13);

    // Add tile layer
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '© OpenStreetMap contributors'
    }).addTo(map);

    // Add sample markers for demo locations
    const demoLocations = [
        { lat: 40.7829, lng: -73.9654, name: "Central Park, NYC", confidence: 94.2 },
        { lat: 50.2957, lng: 36.6619, name: "Ukraine Region", confidence: 87.5 },
        { lat: 51.5074, lng: -0.1278, name: "London, UK", confidence: 91.8 }
    ];

    demoLocations.forEach(location => {
        const marker = L.marker([location.lat, location.lng]).addTo(map);
        marker.bindPopup(`
            <b>${location.name}</b><br>
            Confidence: ${location.confidence}%<br>
            <small>Demo location</small>
        `);
    });
}

// Charts initialization
function initializeCharts() {
    initializeAccuracyChart();
    initializePerformanceChart();
}

function initializeAccuracyChart() {
    const ctx = document.getElementById('accuracyChart');
    if (!ctx) return;

    confidenceChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: ['Correct Detections', 'False Positives', 'False Negatives'],
            datasets: [{
                data: [96.3, 2.1, 1.6],
                backgroundColor: [
                    '#10b981',
                    '#f59e0b',
                    '#ef4444'
                ],
                borderWidth: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: {
                        padding: 20,
                        font: {
                            size: 12
                        }
                    }
                }
            }
        }
    });
}

function initializePerformanceChart() {
    const ctx = document.getElementById('performanceChart');
    if (!ctx) return;

    performanceChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: ['1km²', '2km²', '5km²', '10km²', '15km²', '20km²'],
            datasets: [{
                label: 'Processing Time (seconds)',
                data: [0.3, 0.5, 1.2, 1.8, 2.4, 3.1],
                borderColor: '#2563eb',
                backgroundColor: 'rgba(37, 99, 235, 0.1)',
                fill: true,
                tension: 0.4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Time (seconds)'
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: 'Search Area'
                    }
                }
            }
        }
    });
}

// Demo controls
function initializeDemoControls() {
    const startDemoBtn = document.getElementById('start-demo');
    const locationSelect = document.getElementById('demo-location');
    const methodSelect = document.getElementById('search-method');

    if (startDemoBtn) {
        startDemoBtn.addEventListener('click', () => {
            if (!demoRunning) {
                startDemo();
            }
        });
    }

    // Location change handler
    if (locationSelect) {
        locationSelect.addEventListener('change', (e) => {
            updateMapLocation(e.target.value);
        });
    }
}

function startDemo() {
    if (demoRunning) return;
    
    demoRunning = true;
    const startBtn = document.getElementById('start-demo');
    const statusText = document.getElementById('status-text');
    const progressFill = document.getElementById('progress-fill');
    
    // Update UI
    startBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Searching...';
    startBtn.disabled = true;
    statusText.textContent = 'Initializing search...';
    
    // Simulate demo progress
    simulateSearch();
}

function simulateSearch() {
    const steps = [
        { progress: 10, status: 'Loading satellite imagery...', time: 500 },
        { progress: 25, status: 'Extracting SIFT features...', time: 800 },
        { progress: 45, status: 'Matching features...', time: 1000 },
        { progress: 65, status: 'Running RL agent...', time: 1200 },
        { progress: 80, status: 'Calculating confidence...', time: 600 },
        { progress: 95, status: 'Finalizing results...', time: 400 },
        { progress: 100, status: 'Search completed!', time: 200 }
    ];
    
    let currentStep = 0;
    
    function executeStep() {
        if (currentStep >= steps.length) {
            completeDemoSearch();
            return;
        }
        
        const step = steps[currentStep];
        updateDemoProgress(step.progress, step.status);
        
        setTimeout(() => {
            currentStep++;
            executeStep();
        }, step.time);
    }
    
    executeStep();
}

function updateDemoProgress(progress, status) {
    const statusText = document.getElementById('status-text');
    const progressFill = document.getElementById('progress-fill');
    
    if (statusText) statusText.textContent = status;
    if (progressFill) progressFill.style.width = progress + '%';
    
    // Update metrics during search
    if (progress > 20) {
        updateDemoMetrics(progress);
    }
}

function updateDemoMetrics(progress) {
    const confidenceValue = document.getElementById('confidence-value');
    const timeValue = document.getElementById('time-value');
    const featuresValue = document.getElementById('features-value');
    const areaValue = document.getElementById('area-value');
    
    // Simulate realistic values based on progress
    const confidence = Math.min(95, Math.floor(progress * 0.95));
    const time = (progress / 100 * 1.8).toFixed(1);
    const features = Math.floor(progress * 8.47);
    const area = "10km²";
    
    if (confidenceValue) confidenceValue.textContent = confidence + '%';
    if (timeValue) timeValue.textContent = time + 's';
    if (featuresValue) featuresValue.textContent = features;
    if (areaValue) areaValue.textContent = area;
}

function completeDemoSearch() {
    const startBtn = document.getElementById('start-demo');
    const statusText = document.getElementById('status-text');
    
    // Final metrics
    updateDemoMetrics(100);
    
    // Add success marker to map
    if (map) {
        const successMarker = L.marker([40.7829, -73.9654]).addTo(map);
        successMarker.bindPopup(`
            <b>🎯 Drone Located!</b><br>
            Confidence: 94.2%<br>
            Processing time: 1.8s<br>
            Features matched: 847
        `).openPopup();
        
        // Pan to marker
        map.setView([40.7829, -73.9654], 15);
    }
    
    // Reset demo state
    setTimeout(() => {
        demoRunning = false;
        if (startBtn) {
            startBtn.innerHTML = '<i class="fas fa-play"></i> Start Demo';
            startBtn.disabled = false;
        }
        if (statusText) {
            statusText.textContent = 'Demo completed successfully!';
        }
    }, 2000);
}

function updateMapLocation(locationKey) {
    if (!map) return;
    
    const locations = {
        'central-park': { lat: 40.7829, lng: -73.9654, zoom: 13 },
        'ukraine': { lat: 50.2957, lng: 36.6619, zoom: 12 },
        'london': { lat: 51.5074, lng: -0.1278, zoom: 13 }
    };
    
    const location = locations[locationKey];
    if (location) {
        map.setView([location.lat, location.lng], location.zoom);
    }
}

// Animation and scroll effects
function initializeAnimations() {
    // Intersection Observer for fade-in animations
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };
    
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('fade-in-up');
            }
        });
    }, observerOptions);
    
    // Observe elements for animation
    const animateElements = document.querySelectorAll('.feature-card, .result-card, .tech-details');
    animateElements.forEach(el => observer.observe(el));
    
    // Parallax effect for hero section
    window.addEventListener('scroll', () => {
        const scrolled = window.pageYOffset;
        const hero = document.querySelector('.hero');
        
        if (hero) {
            const rate = scrolled * -0.5;
            hero.style.transform = `translateY(${rate}px)`;
        }
    });
    
    // Counter animation for stats
    animateCounters();
}

function animateCounters() {
    const counters = document.querySelectorAll('.stat-number, .metric-value, .perf-value');
    
    counters.forEach(counter => {
        const target = parseFloat(counter.textContent.replace(/[^\d.]/g, ''));
        const duration = 2000;
        const step = target / (duration / 16);
        let current = 0;
        
        const timer = setInterval(() => {
            current += step;
            if (current >= target) {
                current = target;
                clearInterval(timer);
            }
            
            // Format the number based on original format
            const originalText = counter.textContent;
            if (originalText.includes('%')) {
                counter.textContent = Math.floor(current) + '%';
            } else if (originalText.includes('s')) {
                counter.textContent = current.toFixed(1) + 's';
            } else if (originalText.includes('km')) {
                counter.textContent = Math.floor(current) + 'km²';
            } else {
                counter.textContent = Math.floor(current);
            }
        }, 16);
    });
}

// Utility functions
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

function throttle(func, limit) {
    let inThrottle;
    return function() {
        const args = arguments;
        const context = this;
        if (!inThrottle) {
            func.apply(context, args);
            inThrottle = true;
            setTimeout(() => inThrottle = false, limit);
        }
    }
}

// Performance monitoring
function trackPerformance() {
    // Track page load time
    window.addEventListener('load', () => {
        const loadTime = performance.now();
        console.log(`Page loaded in ${loadTime.toFixed(2)}ms`);
    });
    
    // Track user interactions
    document.addEventListener('click', (e) => {
        if (e.target.matches('.btn, .nav-link, .tab-button')) {
            console.log(`User clicked: ${e.target.textContent.trim()}`);
        }
    });
}

// Error handling
window.addEventListener('error', (e) => {
    console.error('JavaScript error:', e.error);
    // Could send to analytics service in production
});

// Initialize performance tracking
trackPerformance();

// Smooth scrolling polyfill for older browsers
function smoothScrollPolyfill() {
    if (!('scrollBehavior' in document.documentElement.style)) {
        const links = document.querySelectorAll('a[href^="#"]');
        links.forEach(link => {
            link.addEventListener('click', function(e) {
                e.preventDefault();
                const target = document.querySelector(this.getAttribute('href'));
                if (target) {
                    const offsetTop = target.offsetTop - 70;
                    window.scrollTo({
                        top: offsetTop,
                        behavior: 'smooth'
                    });
                }
            });
        });
    }
}

smoothScrollPolyfill();

// Export functions for external use
window.DroneLocatorDemo = {
    startDemo,
    updateMapLocation,
    initializeMap,
    updateDemoMetrics
};

