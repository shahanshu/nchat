// --- Global Elements & Variables ---
const chatBox = document.getElementById('chat-box');
const userInput = document.getElementById('user-input');
const sendBtn = document.getElementById('send-btn');
const newChatBtn = document.getElementById('new-chat-btn');
const chatHistorySidebar = document.getElementById('chat-history');
const menuBtn = document.getElementById('menu-btn');
const sidebar = document.getElementById('sidebar');
const clearAllBtn = document.getElementById('clear-all-btn');

const API_URL = "http://localhost:8000";

// State
let currentChatId = null;
let currentChatTitle = "New Chat";
let chatMessages = [];      // array of {role, content, metrics}
let chatSessions = [];      // metadata for sidebar

// --- UI Toggle Logic ---
menuBtn.addEventListener('click', () => {
    if (window.innerWidth <= 768) {
        sidebar.classList.toggle('mobile-open');
    } else {
        sidebar.classList.toggle('hidden');
    }
});

// Close sidebar when clicking outside on mobile (optional enhancement)
document.addEventListener('click', (e) => {
    if (window.innerWidth <= 768 && sidebar.classList.contains('mobile-open')) {
        if (!sidebar.contains(e.target) && !menuBtn.contains(e.target)) {
            sidebar.classList.remove('mobile-open');
        }
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
            loadSidebarChats();  // refresh sidebar after save
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
            const sessions = request.result.sort((a, b) => b.updatedAt - a.updatedAt);
            resolve(sessions);
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

async function deleteSessionById(threadId) {
    const db = await initDB();
    return new Promise((resolve, reject) => {
        const tx = db.transaction(STORE_NAME, "readwrite");
        tx.objectStore(STORE_NAME).delete(threadId);
        tx.oncomplete = resolve;
        tx.onerror = (e) => reject(e.target.error);
    });
}

async function deleteAllSessions() {
    const db = await initDB();
    return new Promise((resolve, reject) => {
        const tx = db.transaction(STORE_NAME, "readwrite");
        tx.objectStore(STORE_NAME).clear();
        tx.oncomplete = resolve;
        tx.onerror = (e) => reject(e.target.error);
    });
}

// --- Sidebar Loading with Delete Buttons ---
async function loadSidebarChats() {
    chatSessions = await getAllSessions();
    chatHistorySidebar.innerHTML = '';

    chatSessions.forEach(session => {
        const div = document.createElement('div');
        div.className = `history-item ${session.threadId === currentChatId ? 'active' : ''}`;
        
        // Title span with click handler
        const titleSpan = document.createElement('span');
        titleSpan.className = 'history-item-text';
        titleSpan.textContent = session.title;
        titleSpan.onclick = (e) => {
            e.stopPropagation();
            switchChat(session.threadId);
        };
        
        // Delete button
        const delBtn = document.createElement('button');
        delBtn.className = 'delete-chat-btn';
        delBtn.title = 'Delete Chat';
        delBtn.innerHTML = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="3 6 5 6 21 6"></polyline><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path></svg>`;
        delBtn.onclick = async (event) => {
            event.stopPropagation();
            if (confirm("Are you sure you want to delete this chat?")) {
                await deleteSessionById(session.threadId);
                if (currentChatId === session.threadId) {
                    await createNewChat();
                } else {
                    await loadSidebarChats();
                }
            }
        };
        
        div.appendChild(titleSpan);
        div.appendChild(delBtn);
        chatHistorySidebar.appendChild(div);
    });
}

// Clear All Chats Handler
clearAllBtn.onclick = async () => {
    if (!confirm("⚠️ This will permanently delete ALL your chat history. Continue?")) return;
    await deleteAllSessions();
    await createNewChat();  // resets to fresh state
};

// --- Initialization & Core Functions ---
window.onload = async () => {
    await loadSidebarChats();
    if (chatSessions.length > 0) {
        // load most recent
        await switchChat(chatSessions[0].threadId);
    } else {
        await createNewChat();
    }
};

async function createNewChat() {
    currentChatId = "thread_" + Math.random().toString(36).substring(2, 12);
    currentChatTitle = "New Chat";
    chatMessages = [];
    
    chatBox.innerHTML = '';
    await loadSidebarChats();   // refresh active states
    
    // Auto close sidebar on mobile
    if (window.innerWidth <= 768) {
        sidebar.classList.remove('mobile-open');
    }
    
    // Add welcome message
    appendBotMessage("Hello! I am your IOE Syllabus Assistant. I can help you with subjects like **Artificial Intelligence, Operating Systems, and Data Base Management System**.\n\nYou can ask me about marks allocation, syllabus topics, or course content! How can I help you today?", null, false);
    // Save empty session so it appears in sidebar
    await saveSessionToDB();
}

async function switchChat(threadId) {
    if (threadId === currentChatId) return;
    
    const session = await getSession(threadId);
    if (!session) return;
    
    currentChatId = session.threadId;
    currentChatTitle = session.title;
    chatMessages = session.messages ? [...session.messages] : [];
    
    chatBox.innerHTML = '';
    
    if (window.innerWidth <= 768) {
        sidebar.classList.remove('mobile-open');
    }
    
    if (chatMessages.length === 0) {
        appendBotMessage("Hello! I am your IOE Syllabus Assistant. I can help you with subjects like **Artificial Intelligence, Operating Systems, and Data Base Management System**.\n\nYou can ask me about marks allocation, syllabus topics, or course content! How can I help you today?", null, false);
    } else {
        chatMessages.forEach(msg => {
            if (msg.role === 'user') {
                appendUserMessage(msg.content, false);
            } else if (msg.role === 'bot') {
                appendBotMessage(msg.content, msg.metrics || null, false);
            }
        });
    }
    
    await loadSidebarChats(); // highlight active
}

// --- Event Listeners ---
newChatBtn.addEventListener('click', async () => {
    await createNewChat();
});

sendBtn.addEventListener('click', handleUserInput);
userInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') handleUserInput();
});

// --- Main Chat Logic ---
async function handleUserInput() {
    const text = userInput.value.trim();
    if (!text) return;
    
    // Auto title for new chat based on first user message
    if (chatMessages.length === 0) {
        currentChatTitle = text.length > 28 ? text.substring(0, 28) + '...' : text;
    }
    
    appendUserMessage(text, true);
    userInput.value = '';
    
    const loadingId = appendLoadingMessage();
    
    try {
        const response = await fetch(`${API_URL}/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: text, thread_id: currentChatId })
        });
        
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const data = await response.json();
        removeLoadingMessage(loadingId);
        
        if (data.status === "needs_confirmation") {
            appendWebSearchPrompt();
        } else {
            appendBotMessage(data.answer, data.metrics || null, true);
        }
    } catch (error) {
        removeLoadingMessage(loadingId);
        appendBotMessage("❌ Error: Could not connect to the Python backend. Make sure FastAPI is running at " + API_URL, null, true);
    }
}

// --- UI Rendering Helpers ---
function appendUserMessage(text, save = false) {
    if (save) {
        chatMessages.push({ role: 'user', content: text });
        saveSessionToDB();
    }
    const msgDiv = document.createElement('div');
    msgDiv.className = 'message user-message';
    msgDiv.innerHTML = `
        <div class="avatar user-avatar">U</div>
        <div class="message-content">${escapeHtml(text)}</div>
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
    let formattedHtml = marked.parse(text);
    
    let contentHtml = `
        <img src="logo.png" alt="Bot" class="avatar bot-avatar" onerror="this.style.display='none'">
        <div class="message-content markdown-body">${formattedHtml}
    `;
    
    if (metrics && typeof metrics === 'object') {
        contentHtml += `<div class="metrics-panel">`;
        if (metrics.answerRelevance !== undefined) contentHtml += getMetricHtml('Ans Relevance', metrics.answerRelevance);
        if (metrics.retrievalRelevance !== undefined) contentHtml += getMetricHtml('Ret Relevance', metrics.retrievalRelevance);
        if (metrics.groundedness !== undefined) contentHtml += getMetricHtml('Groundedness', metrics.groundedness);
        if (metrics.correctness !== undefined) contentHtml += getMetricHtml('Correctness', metrics.correctness);
        contentHtml += `</div>`;
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
            <div class="typing-indicator"><span></span><span></span><span></span></div>
        </div>
    `;
    chatBox.appendChild(msgDiv);
    chatBox.scrollTop = chatBox.scrollHeight;
    return id;
}

function removeLoadingMessage(id) {
    const el = document.getElementById(id);
    if (el) el.remove();
}

function getMetricHtml(name, score) {
    let colorClass = 'metric-poor';
    let icon = '🔴';
    if (score >= 80) { colorClass = 'metric-good'; icon = '🟢'; }
    else if (score >= 60) { colorClass = 'metric-avg'; icon = '🟡'; }
    return `<span class="metric-badge ${colorClass}">${icon} ${name}: ${score}%</span>`;
}

// Web Search Prompt
function appendWebSearchPrompt() {
    const msgDiv = document.createElement('div');
    msgDiv.className = 'message bot-message';
    msgDiv.innerHTML = `
        <img src="logo.png" alt="Bot" class="avatar bot-avatar" onerror="this.style.display='none'">
        <div class="message-content">
            ⚠️ <strong>Out of Syllabus Detected</strong><br>
            This topic is outside your current IOE subjects context. Would you like me to search the web for an answer?
            <div class="action-buttons">
                <button class="btn btn-yes" onclick="window.handleWebSearch(true, this)">✅ Yes, Search Web</button>
                <button class="btn btn-no" onclick="window.handleWebSearch(false, this)">❌ No, Cancel</button>
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
        
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const data = await response.json();
        removeLoadingMessage(loadingId);
        appendBotMessage(data.answer, data.metrics || null, true);
    } catch (error) {
        removeLoadingMessage(loadingId);
        appendBotMessage("❌ Error completing web search. Please check backend connection.", null, true);
    }
};

// Helper function to escape HTML to prevent XSS
function escapeHtml(str) {
    if (!str) return '';
    return str.replace(/[&<>]/g, function(m) {
        if (m === '&') return '&amp;';
        if (m === '<') return '&lt;';
        if (m === '>') return '&gt;';
        return m;
    }).replace(/[\uD800-\uDBFF][\uDC00-\uDFFF]/g, function(c) {
        return c;
    });
}