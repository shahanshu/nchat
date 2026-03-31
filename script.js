const chatBox = document.getElementById('chat-box');
const userInput = document.getElementById('user-input');
const sendBtn = document.getElementById('send-btn');

// Generate a unique ID for this chat session
const threadId = "thread_" + Math.random().toString(36).substring(2, 10);
const API_URL = "http://localhost:8000";

// --- Event Listeners ---
sendBtn.addEventListener('click', handleUserInput);
userInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') handleUserInput();
});

// --- Main Chat Logic ---
async function handleUserInput() {
    const text = userInput.value.trim();
    if (!text) return;

    appendUserMessage(text);
    userInput.value = '';

    // Show loading indicator
    const loadingId = appendLoadingMessage();

    try {
        const response = await fetch(`${API_URL}/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: text, thread_id: threadId })
        });

        const data = await response.json();
        removeLoadingMessage(loadingId);

        if (data.status === "needs_confirmation") {
            appendWebSearchPrompt();
        } else {
            appendBotMessage(data.answer, data.metrics);
        }
    } catch (error) {
        removeLoadingMessage(loadingId);
        appendBotMessage("❌ Error: Could not connect to the Python backend. Is FastAPI running?");
    }
}

// --- Append Messages ---
function appendUserMessage(text) {
    const msgDiv = document.createElement('div');
    msgDiv.className = 'message user-message';
    msgDiv.innerHTML = `
        <div class="avatar user-avatar">U</div>
        <div class="message-content">${text}</div>
    `;
    chatBox.appendChild(msgDiv);
    chatBox.scrollTop = chatBox.scrollHeight;
}

function appendBotMessage(text, metrics = null) {
    const msgDiv = document.createElement('div');
    msgDiv.className = 'message bot-message';
    
    // Parse the markdown text into proper HTML with headings, lists, and line breaks
    const formattedText = marked.parse(text);
    
    // HTML with logo avatar and markdown content
    let contentHtml = `
        <img src="logo.png" alt="Bot" class="avatar bot-avatar" onerror="this.style.display='none'">
        <div class="message-content markdown-body">${formattedText}
    `;
    
    // Append the metrics if they exist
    if (metrics) {
        contentHtml += `<div class="metrics-panel">
            ${getMetricHtml('Ans Relevance', metrics.answerRelevance)}
            ${getMetricHtml('Ret Relevance', metrics.retrievalRelevance)}
            ${getMetricHtml('Groundedness', metrics.groundedness)}
            ${getMetricHtml('Correctness', metrics.correctness)}
        </div>`;
    }
    
    contentHtml += `</div>`;
    msgDiv.innerHTML = contentHtml;
    
    chatBox.appendChild(msgDiv);
    chatBox.scrollTop = chatBox.scrollHeight;
}

// --- Loading UI ---
function appendLoadingMessage() {
    const id = "loading_" + Date.now();
    const msgDiv = document.createElement('div');
    msgDiv.className = 'message bot-message';
    msgDiv.id = id;
    
    // New bouncing dots typing indicator
    msgDiv.innerHTML = `
        <img src="logo.png" alt="Bot" class="avatar bot-avatar" onerror="this.style.display='none'">
        <div class="message-content">
            <div class="typing-indicator">
                <span></span><span></span><span></span>
            </div>
        </div>
    `;
    chatBox.appendChild(msgDiv);
    chatBox.scrollTop = chatBox.scrollHeight;
    return id;
}

function removeLoadingMessage(id) {
    const msgDiv = document.getElementById(id);
    if (msgDiv) msgDiv.remove();
}

// --- Metrics UI ---
function getMetricHtml(name, score) {
    let colorClass = 'metric-poor'; 
    let icon = '🔴';

    if (score >= 80) {
        colorClass = 'metric-good'; 
        icon = '🟢';
    } else if (score >= 60) {
        colorClass = 'metric-avg';  
        icon = '🟡';
    }

    return `<span class="metric-badge ${colorClass}">${icon} ${name}: ${score}%</span>`;
}

// --- Web Search Logic ---
function appendWebSearchPrompt() {
    const msgDiv = document.createElement('div');
    msgDiv.className = 'message bot-message';
    msgDiv.innerHTML = `
        <img src="logo.png" alt="Bot" class="avatar bot-avatar" onerror="this.style.display='none'">
        <div class="message-content">
            ⚠️ <strong>Out of Syllabus Detected</strong><br>
            This topic is outside your current IOE subjects context. Would you like me to search the web for an answer?
            <div class="action-buttons">
                <button class="btn btn-yes" onclick="handleWebSearch(true, this)">✅ Yes, Search Web</button>
                <button class="btn btn-no" onclick="handleWebSearch(false, this)">❌ No, Cancel</button>
            </div>
        </div>
    `;
    chatBox.appendChild(msgDiv);
    chatBox.scrollTop = chatBox.scrollHeight;
}

window.handleWebSearch = async function(isYes, buttonElement) {
    const buttons = buttonElement.parentElement.querySelectorAll('button');
    buttons.forEach(btn => btn.disabled = true);
    
    const loadingId = appendLoadingMessage();

    try {
        const response = await fetch(`${API_URL}/websearch`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ confirm: isYes, thread_id: threadId })
        });

        const data = await response.json();
        removeLoadingMessage(loadingId);
        
        appendBotMessage(data.answer, data.metrics);
        
    } catch (error) {
        removeLoadingMessage(loadingId);
        appendBotMessage("❌ Error completing web search.");
    }
}