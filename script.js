// --- Global Elements & Variables ---
const chatBox = document.getElementById('chat-box');
const userInput = document.getElementById('user-input');
const sendBtn = document.getElementById('send-btn');
const newChatBtn = document.getElementById('new-chat-btn');
const chatHistorySidebar = document.getElementById('chat-history');

// Menu toggle elements
const menuBtn = document.getElementById('menu-btn');
const sidebar = document.getElementById('sidebar');

const API_URL = "http://localhost:8000";

// State
let currentChatId = null;
let currentChatTitle = "New Chat";
let chatMessages = []; 
let chatSessions = []; // Stores metadata for sidebar

// --- UI Toggle Logic ---
menuBtn.addEventListener('click', () => {
    // If mobile width
    if (window.innerWidth <= 768) {
        sidebar.classList.toggle('mobile-open');
    } else {
        // If desktop width
        sidebar.classList.toggle('hidden');
    }
});

// --- IndexedDB Setup ---
const DB_NAME = "ChatBotDB";
const STORE_NAME = "chats";

function initDB() {
    return new Promise((resolve, reject) => {
        const request = indexedDB.open(DB_NAME, 1);
        request.onupgradeneeded = (e) => {
            const db = e.target.result;
            if (!db.objectStoreNames.contains(STORE_NAME)) {
                db.createObjectStore(STORE_NAME, { keyPath: "threadId" });
            }
        };
        request.onsuccess = (e) => resolve(e.target.result);
        request.onerror = (e) => reject(e.target.error);
    });
}

async function saveSessionToDB() {
    const db = await initDB();
    const session = {
        threadId: currentChatId,
        title: currentChatTitle,
        messages: chatMessages,
        updatedAt: Date.now()
    };
    
    return new Promise((resolve, reject) => {
        const tx = db.transaction(STORE_NAME, "readwrite");
        tx.objectStore(STORE_NAME).put(session);
        tx.oncomplete = () => {
            loadSidebarChats(); // Refresh sidebar after saving
            resolve();
        };
        tx.onerror = (e) => reject(e.target.error);
    });
}

async function getAllSessions() {
    const db = await initDB();
    return new Promise((resolve, reject) => {
        const tx = db.transaction(STORE_NAME, "readonly");
        const request = tx.objectStore(STORE_NAME).getAll();
        request.onsuccess = () => {
            // Sort by most recently updated
            resolve(request.result.sort((a, b) => b.updatedAt - a.updatedAt));
        };
        request.onerror = (e) => reject(e.target.error);
    });
}

async function getSession(threadId) {
    const db = await initDB();
    return new Promise((resolve, reject) => {
        const tx = db.transaction(STORE_NAME, "readonly");
        const request = tx.objectStore(STORE_NAME).get(threadId);
        request.onsuccess = () => resolve(request.result);
        request.onerror = (e) => reject(e.target.error);
    });
}

// --- Initialization ---
window.onload = async () => {
    await loadSidebarChats();
    if (chatSessions.length > 0) {
        // Load the most recent chat
        await switchChat(chatSessions[0].threadId);
    } else {
        createNewChat();
    }
};

// --- Sidebar Logic ---
async function loadSidebarChats() {
    chatSessions = await getAllSessions();
    chatHistorySidebar.innerHTML = '';
    
    chatSessions.forEach(session => {
        const div = document.createElement('div');
        div.className = `history-item ${session.threadId === currentChatId ? 'active' : ''}`;
        div.textContent = session.title;
        div.onclick = () => switchChat(session.threadId);
        chatHistorySidebar.appendChild(div);
    });
}

function createNewChat() {
    currentChatId = "thread_" + Math.random().toString(36).substring(2, 10);
    currentChatTitle = "New Chat";
    chatMessages = [];
    
    chatBox.innerHTML = '';
    loadSidebarChats(); // Remove active state visually
    
    // Auto close sidebar on mobile when hitting "New chat"
    if (window.innerWidth <= 768) {
        sidebar.classList.remove('mobile-open');
    }

    appendBotMessage("Hello! I am your IOE Syllabus Assistant. I can help you with subjects like **Artificial Intelligence, Operating Systems, and Data Base Management System**.\n\nYou can ask me about marks allocation, syllabus topics, or course content! How can I help you today?", null, false);
}

async function switchChat(threadId) {
    if (threadId === currentChatId) return;
    
    const session = await getSession(threadId);
    if (!session) return;

    currentChatId = session.threadId;
    currentChatTitle = session.title;
    chatMessages = session.messages;

    chatBox.innerHTML = '';
    
    // Auto close sidebar on mobile when switching to an old chat
    if (window.innerWidth <= 768) {
        sidebar.classList.remove('mobile-open');
    }

    // Render stored messages
    if (chatMessages.length === 0) {
        appendBotMessage("Hello! I am your IOE Syllabus Assistant. I can help you with subjects like **Artificial Intelligence, Operating Systems, and Data Base Management System**.\n\nYou can ask me about marks allocation, syllabus topics, or course content! How can I help you today?", null, false);
    } else {
        chatMessages.forEach(msg => {
            if (msg.role === 'user') {
                appendUserMessage(msg.content, false);
            } else if (msg.role === 'bot') {
                appendBotMessage(msg.content, msg.metrics, false);
            }
        });
    }

    loadSidebarChats(); // Update active class visually
}

// --- Event Listeners ---
newChatBtn.addEventListener('click', createNewChat);

sendBtn.addEventListener('click', handleUserInput);
userInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') handleUserInput();
});

// --- Main Chat Logic ---
async function handleUserInput() {
    const text = userInput.value.trim();
    if (!text) return;

    // Set title based on first user message if it's a new chat
    if (chatMessages.length === 0) {
        currentChatTitle = text.length > 25 ? text.substring(0, 25) + '...' : text;
    }

    appendUserMessage(text, true); // True means save to DB
    userInput.value = '';

    const loadingId = appendLoadingMessage();

    try {
        const response = await fetch(`${API_URL}/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: text, thread_id: currentChatId })
        });

        const data = await response.json();
        removeLoadingMessage(loadingId);

        if (data.status === "needs_confirmation") {
            appendWebSearchPrompt();
        } else {
            appendBotMessage(data.answer, data.metrics, true);
        }
    } catch (error) {
        removeLoadingMessage(loadingId);
        appendBotMessage("❌ Error: Could not connect to the Python backend. Is FastAPI running?", null, true);
    }
}

// --- UI Rendering ---
function appendUserMessage(text, save = false) {
    if (save) {
        chatMessages.push({ role: 'user', content: text });
        saveSessionToDB();
    }

    const msgDiv = document.createElement('div');
    msgDiv.className = 'message user-message';
    msgDiv.innerHTML = `
        <div class="avatar user-avatar">U</div>
        <div class="message-content">${text}</div>
    `;
    chatBox.appendChild(msgDiv);
    chatBox.scrollTop = chatBox.scrollHeight;
}

function appendBotMessage(text, metrics = null, save = false) {
    if (save) {
        chatMessages.push({ role: 'bot', content: text, metrics: metrics });
        saveSessionToDB();
    }

    const msgDiv = document.createElement('div');
    msgDiv.className = 'message bot-message';
    
    const formattedText = marked.parse(text);
    
    let contentHtml = `
        <img src="logo.png" alt="Bot" class="avatar bot-avatar" onerror="this.style.display='none'">
        <div class="message-content markdown-body">${formattedText}
    `;
    
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

function appendLoadingMessage() {
    const id = "loading_" + Date.now();
    const msgDiv = document.createElement('div');
    msgDiv.className = 'message bot-message';
    msgDiv.id = id;
    
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

function getMetricHtml(name, score) {
    let colorClass = 'metric-poor'; 
    let icon = '🔴';

    if (score >= 80) {
        colorClass = 'metric-good'; icon = '🟢';
    } else if (score >= 60) {
        colorClass = 'metric-avg';  icon = '🟡';
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
            body: JSON.stringify({ confirm: isYes, thread_id: currentChatId })
        });

        const data = await response.json();
        removeLoadingMessage(loadingId);
        
        appendBotMessage(data.answer, data.metrics, true); // true = save outcome to history
        
    } catch (error) {
        removeLoadingMessage(loadingId);
        appendBotMessage("❌ Error completing web search.", null, true);
    }
}