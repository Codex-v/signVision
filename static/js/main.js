// Form switching functionality
const loginForm = document.getElementById("login");
const registerForm = document.getElementById("register");
const btn = document.getElementById("btn");

function register() {
    loginForm.style.opacity = "0";
    setTimeout(() => {
        loginForm.style.display = "none";
        registerForm.style.display = "block";
        setTimeout(() => {
            registerForm.style.opacity = "1";
        }, 50);
    }, 300);
    btn.style.left = "110px";
}

function login() {
    registerForm.style.opacity = "0";
    setTimeout(() => {
        registerForm.style.display = "none";
        loginForm.style.display = "block";
        setTimeout(() => {
            loginForm.style.opacity = "1";
        }, 50);
    }, 300);
    btn.style.left = "0";
}

// Initialize forms
loginForm.style.opacity = "1";
registerForm.style.opacity = "0";
registerForm.style.display = "none";



document.getElementById('login').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const formData = new FormData(e.target);
    
    try {
        const response = await fetch('/login', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (data.status === 'success') {
            window.location.href = data.redirect;
        } else {
            alert(data.message);
            console.log(data.message)
        }
    } catch (error) {
        alert('An error occurred during login.');
    }
});



function animateFormSubmission(form) {
    const btn = form.querySelector('.submit-btn');
    btn.style.transform = 'scale(0.95)';
    setTimeout(() => {
        btn.style.transform = 'scale(1)';
    }, 100);
}

// Add input animations
const inputs = document.querySelectorAll('.input');
inputs.forEach(input => {
    input.addEventListener('focus', () => {
        input.parentElement.classList.add('focused');
        const icon = input.parentElement.querySelector('.ai-icon');
        icon.style.opacity = "1";
    });
    
    input.addEventListener('blur', () => {
        if (!input.value) {
            input.parentElement.classList.remove('focused');
        }
        const icon = input.parentElement.querySelector('.ai-icon');
        icon.style.opacity = "0.7";
    });
});

// Neural network animation
function createNeuralNetwork() {
    const canvas = document.createElement('canvas');
    const network = document.querySelector('.neural-network');
    network.appendChild(canvas);
    
    const ctx = canvas.getContext('2d');
    canvas.width = network.offsetWidth;
    canvas.height = network.offsetHeight;
    
    const nodes = [];
    const connections = [];
    
    // Create nodes
    for (let i = 0; i < 30; i++) {
        nodes.push({
            x: Math.random() * canvas.width,
            y: Math.random() * canvas.height,
            radius: Math.random() * 2 + 1,
            speed: Math.random() * 0.5 + 0.2
        });
    }
    
    // Create connections
    for (let i = 0; i < nodes.length; i++) {
        for (let j = i + 1; j < nodes.length; j++) {
            if (Math.random() < 0.1) {
                connections.push([i, j]);
            }
        }
    }
    
    function animate() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        // Update and draw connections
        ctx.strokeStyle = 'rgba(0, 247, 255, 0.1)';
        connections.forEach(([i, j]) => {
            ctx.beginPath();
            ctx.moveTo(nodes[i].x, nodes[i].y);
            ctx.lineTo(nodes[j].x, nodes[j].y);
            ctx.stroke();
        });
        
        // Update and draw nodes
        nodes.forEach(node => {
            node.x += Math.sin(Date.now() * 0.001) * node.speed;
            node.y += Math.cos(Date.now() * 0.001) * node.speed;
            
            if (node.x < 0) node.x = canvas.width;
            if (node.x > canvas.width) node.x = 0;
            if (node.y < 0) node.y = canvas.height;
            if (node.y > canvas.height) node.y = 0;
            
            ctx.beginPath();
            ctx.arc(node.x, node.y, node.radius, 0, Math.PI * 2);
            ctx.fillStyle = 'rgba(0, 247, 255, 0.5)';
            ctx.fill();
        });
        
        requestAnimationFrame(animate);
    }
    
    animate();
}

// Initialize neural network animation
createNeuralNetwork();

// Handle window resize
window.addEventListener('resize', () => {
    const network = document.querySelector('.neural-network');
    network.innerHTML = '';
    createNeuralNetwork();
});

// AI Assistant interaction
const aiAssistant = document.querySelector('.ai-assistant');
aiAssistant.addEventListener('click', () => {
    aiAssistant.style.transform = 'scale(0.9)';
    setTimeout(() => {
        aiAssistant.style.transform = 'scale(1)';
    }, 100);
    // Add your AI assistant functionality here
});