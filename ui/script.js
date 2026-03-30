/**
 * AI OS - Web UI JavaScript
 * Interactive frontend for the Local AI Assistant
 */

// API Configuration
const API_URL = localStorage.getItem('apiUrl') || 'http://localhost:8000';

// State
let sessionId = null;
let currentCommand = null;
let isGenerating = false;

// DOM Elements
const elements = {
    // Navigation
    navButtons: document.querySelectorAll('.nav-btn'),
    views: document.querySelectorAll('.view'),
    pageTitle: document.getElementById('page-title'),
    
    // Status
    statusDot: document.getElementById('status-dot'),
    statusText: document.getElementById('status-text'),
    
    // Chat
    messages: document.getElementById('messages'),
    messageInput: document.getElementById('message-input'),
    sendBtn: document.getElementById('send-btn'),
    executeMode: document.getElementById('execute-mode'),
    ragMode: document.getElementById('rag-mode'),
    clearChat: document.getElementById('clear-chat'),
    
    // Commands
    commandInput: document.getElementById('command-input'),
    generateBtn: document.getElementById('generate-btn'),
    generatedCommand: document.getElementById('generated-command'),
    commandPreview: document.getElementById('command-preview'),
    executeBtn: document.getElementById('execute-btn'),
    autoExecute: document.getElementById('auto-execute'),
    checkSafetyBtn: document.getElementById('check-safety-btn'),
    safetyResult: document.getElementById('safety-result'),
    commandOutput: document.getElementById('command-output'),
    copyCmd: document.getElementById('copy-cmd'),
    editCmd: document.getElementById('edit-cmd'),
    quickBtns: document.querySelectorAll('.quick-btn'),
    
    // RAG
    indexPath: document.getElementById('index-path'),
    indexBtn: document.getElementById('index-btn'),
    recursiveIndex: document.getElementById('recursive-index'),
    indexStats: document.getElementById('index-stats'),
    ragQuery: document.getElementById('rag-query'),
    ragSearchBtn: document.getElementById('rag-search-btn'),
    searchResults: document.getElementById('search-results'),
    
    // History
    historyList: document.getElementById('history-list'),
    clearHistoryBtn: document.getElementById('clear-history-btn'),
    
    // Settings
    apiUrl: document.getElementById('api-url'),
    testConnection: document.getElementById('test-connection'),
    themeSelect: document.getElementById('theme-select'),
    fontSize: document.getElementById('font-size'),
    toggleTheme: document.getElementById('toggle-theme'),
    
    // Modal
    confirmModal: document.getElementById('confirm-modal'),
    confirmCommand: document.getElementById('confirm-command'),
    confirmReason: document.getElementById('confirm-reason'),
    cancelCmd: document.getElementById('cancel-cmd'),
    confirmCmd: document.getElementById('confirm-cmd'),
    
    // Toast
    toastContainer: document.getElementById('toast-container'),
};

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    initializeApp();
});

function initializeApp() {
    loadSettings();
    setupEventListeners();
    checkHealth();
    loadHistory();
    
    // Check health periodically
    setInterval(checkHealth, 30000);
}

// Settings
function loadSettings() {
    const savedUrl = localStorage.getItem('apiUrl');
    if (savedUrl) {
        elements.apiUrl.value = savedUrl;
    }
    
    const savedTheme = localStorage.getItem('theme') || 'dark';
    elements.themeSelect.value = savedTheme;
    document.documentElement.setAttribute('data-theme', savedTheme);
    
    const savedFontSize = localStorage.getItem('fontSize') || 'medium';
    elements.fontSize.value = savedFontSize;
    applyFontSize(savedFontSize);
}

function saveSettings() {
    localStorage.setItem('apiUrl', elements.apiUrl.value);
    localStorage.setItem('theme', elements.themeSelect.value);
    localStorage.setItem('fontSize', elements.fontSize.value);
}

function applyFontSize(size) {
    const sizes = {
        small: '14px',
        medium: '16px',
        large: '18px',
    };
    document.documentElement.style.fontSize = sizes[size] || sizes.medium;
}

// Event Listeners
function setupEventListeners() {
    // Navigation
    elements.navButtons.forEach(btn => {
        btn.addEventListener('click', () => switchView(btn.dataset.view));
    });
    
    // Chat
    elements.sendBtn.addEventListener('click', sendMessage);
    elements.messageInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });
    elements.messageInput.addEventListener('input', autoResize);
    elements.clearChat.addEventListener('click', clearChat);
    
    // Commands
    elements.generateBtn.addEventListener('click', generateCommand);
    elements.executeBtn.addEventListener('click', () => executeCommand(currentCommand));
    elements.checkSafetyBtn.addEventListener('click', checkCommandSafety);
    elements.commandInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') generateCommand();
    });
    elements.copyCmd.addEventListener('click', copyCommand);
    elements.editCmd.addEventListener('click', editCommand);
    elements.quickBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            elements.commandInput.value = btn.dataset.cmd;
            generateCommand();
        });
    });
    
    // RAG
    elements.indexBtn.addEventListener('click', indexFiles);
    elements.ragSearchBtn.addEventListener('click', searchRag);
    elements.ragQuery.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') searchRag();
    });
    
    // History
    elements.clearHistoryBtn.addEventListener('click', clearHistory);
    
    // Settings
    elements.testConnection.addEventListener('click', testConnection);
    elements.themeSelect.addEventListener('change', (e) => {
        document.documentElement.setAttribute('data-theme', e.target.value);
        saveSettings();
    });
    elements.fontSize.addEventListener('change', (e) => {
        applyFontSize(e.target.value);
        saveSettings();
    });
    elements.toggleTheme.addEventListener('click', toggleTheme);
    elements.apiUrl.addEventListener('change', saveSettings);
    
    // Modal
    elements.cancelCmd.addEventListener('click', closeModal);
    elements.confirmCmd.addEventListener('click', confirmExecution);
    elements.confirmModal.addEventListener('click', (e) => {
        if (e.target === elements.confirmModal) closeModal();
    });
}

// Navigation
function switchView(viewName) {
    // Update nav buttons
    elements.navButtons.forEach(btn => {
        btn.classList.toggle('active', btn.dataset.view === viewName);
    });
    
    // Update views
    elements.views.forEach(view => {
        view.classList.toggle('active', view.id === `${viewName}-view`);
    });
    
    // Update title
    const titles = {
        chat: 'Chat',
        commands: 'Command Generator',
        rag: 'RAG Search',
        history: 'Command History',
        settings: 'Settings',
    };
    elements.pageTitle.textContent = titles[viewName] || 'AI OS';
    
    // Load data if needed
    if (viewName === 'history') {
        loadHistory();
    }
}

// Health Check
async function checkHealth() {
    try {
        const response = await fetch(`${API_URL}/health`);
        const data = await response.json();
        
        if (data.status === 'healthy') {
            elements.statusDot.className = 'status-dot online';
            elements.statusText.textContent = 'Online';
        } else {
            elements.statusDot.className = 'status-dot';
            elements.statusText.textContent = 'Degraded';
        }
    } catch (error) {
        elements.statusDot.className = 'status-dot offline';
        elements.statusText.textContent = 'Offline';
    }
}

// Chat Functions
async function sendMessage() {
    const message = elements.messageInput.value.trim();
    if (!message || isGenerating) return;
    
    // Add user message
    addMessage(message, 'user');
    elements.messageInput.value = '';
    autoResize();
    
    // Show loading
    isGenerating = true;
    const loadingId = addLoadingMessage();
    
    try {
        const response = await fetch(`${API_URL}/chat/stream`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                message,
                session_id: sessionId,
                execute_command: elements.executeMode.checked,
                use_rag: elements.ragMode.checked,
            }),
        });
        
        removeMessage(loadingId);
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const reader = response.body.getReader();
        const decoder = new TextDecoder("utf-8");
        
        // Create an empty message bubble for the stream
        const msgDiv = addMessage("", 'system');
        const contentDiv = msgDiv.querySelector('.message-content');
        let fullText = "";
        let buffer = "";
        
        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            
            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            
            // Keep the last incomplete line in the buffer
            buffer = lines.pop() || "";
            
            for (const line of lines) {
                if (line.trim().startsWith('data: ')) {
                    const dataStr = line.replace('data: ', '').trim();
                    if (dataStr === '[DONE]') break;
                    
                    try {
                        const data = JSON.parse(dataStr);
                        if (data.session_id && !sessionId) {
                            sessionId = data.session_id;
                        }
                        if (data.error) {
                            fullText += `\n\n**Error:** ${data.error}`;
                            contentDiv.innerHTML = formatMessage(fullText);
                            scrollToBottom();
                        }
                        if (data.chunk) {
                            fullText += data.chunk;
                            contentDiv.innerHTML = formatMessage(fullText);
                            scrollToBottom();
                        }
                    } catch (e) {
                        // Ignore parse errors for incomplete JSON chunks
                    }
                }
            }
        }
        
    } catch (error) {
        removeMessage(loadingId);
        addMessage(`Error: ${error.message}`, 'system', true);
    } finally {
        isGenerating = false;
    }
}

function addMessage(content, role, isError = false) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}`;
    
    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.innerHTML = role === 'user' ? '<i class="fas fa-user"></i>' : '<i class="fas fa-robot"></i>';
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    
    if (isError) {
        contentDiv.innerHTML = `<span style="color: var(--accent-danger)">${escapeHtml(content)}</span>`;
    } else {
        contentDiv.innerHTML = formatMessage(content);
    }
    
    messageDiv.appendChild(avatar);
    messageDiv.appendChild(contentDiv);
    
    elements.messages.appendChild(messageDiv);
    scrollToBottom();
    
    return messageDiv;
}

function addLoadingMessage() {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message system';
    messageDiv.id = 'loading-message';
    
    messageDiv.innerHTML = `
        <div class="message-avatar"><i class="fas fa-robot"></i></div>
        <div class="message-content">
            <div class="spinner"></div>
        </div>
    `;
    
    elements.messages.appendChild(messageDiv);
    scrollToBottom();
    
    return messageDiv.id;
}

function removeMessage(id) {
    const element = typeof id === 'string' ? document.getElementById(id) : id;
    if (element) {
        element.remove();
    }
}

function scrollToBottom() {
    elements.messages.scrollTop = elements.messages.scrollHeight;
}

function formatMessage(content) {
    // Escape HTML
    content = escapeHtml(content);
    
    // Format code blocks
    content = content.replace(/```(\w+)?\n([\s\S]*?)```/g, '<pre><code>$2</code></pre>');
    content = content.replace(/`([^`]+)`/g, '<code>$1</code>');
    
    // Format bold and italic
    content = content.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    content = content.replace(/\*(.*?)\*/g, '<em>$1</em>');
    
    // Format lists
    content = content.replace(/^\s*[-*]\s+(.*)$/gm, '<li>$1</li>');
    content = content.replace(/(<li>.*<\/li>\n?)+/g, '<ul>$&</ul>');
    
    // Format newlines
    content = content.replace(/\n/g, '<br>');
    
    return content;
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function autoResize() {
    elements.messageInput.style.height = 'auto';
    elements.messageInput.style.height = Math.min(elements.messageInput.scrollHeight, 150) + 'px';
}

function clearChat() {
    elements.messages.innerHTML = `
        <div class="message system">
            <div class="message-avatar"><i class="fas fa-robot"></i></div>
            <div class="message-content">
                <p>Chat cleared. How can I help you?</p>
            </div>
        </div>
    `;
    sessionId = null;
}

// Command Functions
async function generateCommand() {
    const request = elements.commandInput.value.trim();
    if (!request || isGenerating) return;
    
    isGenerating = true;
    elements.generateBtn.disabled = true;
    elements.generateBtn.innerHTML = '<div class="spinner"></div> Generating...';
    
    try {
        const response = await fetch(`${API_URL}/chat/simple`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                message: `Generate command: ${request}`,
                execute_command: false,
            }),
        });
        
        const data = await response.json();
        
        // Extract command from response
        let command = data.message;
        
        // Clean up command (remove markdown, explanations)
        const codeMatch = command.match(/```(?:bash|sh|shell)?\n([\s\S]*?)```/i);
        if (codeMatch) {
            command = codeMatch[1].trim();
        } else {
            command = command.replace(/```[\w]*\n?/gi, '').replace(/```/g, '');
            const lines = command.split('\n').map(l => l.trim()).filter(l => l.length > 0);
            if (lines.length > 0) {
                // Return the last line as it's typically the command
                command = lines[lines.length - 1];
            } else {
                command = command.trim();
            }
        }
        
        currentCommand = command;
        elements.generatedCommand.textContent = command;
        
        const isBlocked = command.startsWith('BLOCKED') || command.startsWith('ERROR');
        elements.executeBtn.disabled = isBlocked;
        elements.safetyResult.innerHTML = '';
        elements.commandOutput.innerHTML = '';
        
        if (!isBlocked && elements.autoExecute && elements.autoExecute.checked) {
            executeCommand(currentCommand);
        }
        
    } catch (error) {
        showToast('Error generating command', 'error');
    } finally {
        isGenerating = false;
        elements.generateBtn.disabled = false;
        elements.generateBtn.innerHTML = '<i class="fas fa-magic"></i> Generate';
    }
}

async function checkCommandSafety() {
    if (!currentCommand) return;
    
    try {
        const response = await fetch(`${API_URL}/command/check`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ command: currentCommand }),
        });
        
        const data = await response.json();
        
        const badgeClass = data.risk_level === 'LOW' ? 'safe' : 
                          data.risk_level === 'MEDIUM' ? 'warning' : 'danger';
        
        elements.safetyResult.innerHTML = `
            <div class="safety-badge ${badgeClass}">
                <i class="fas fa-${data.is_safe ? 'check-circle' : 'exclamation-triangle'}"></i>
                ${data.risk_level} Risk - ${data.reason || (data.is_safe ? 'Safe to execute' : 'Blocked')}
            </div>
        `;
        
    } catch (error) {
        showToast('Error checking safety', 'error');
    }
}

async function executeCommand(command) {
    if (!command) return;
    
    // Check safety first
    try {
        const safetyResponse = await fetch(`${API_URL}/command/check`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ command }),
        });
        
        const safety = await safetyResponse.json();
        
        if (!safety.is_safe) {
            showToast(`Command blocked: ${safety.reason}`, 'error');
            return;
        }
        
        if (safety.requires_confirmation) {
            if (elements.autoExecute && elements.autoExecute.checked) {
                await doExecute(command);
                return;
            }
            showConfirmationModal(command, safety.reason);
            return;
        }
        
        // Execute
        await doExecute(command);
        
    } catch (error) {
        showToast('Error executing command', 'error');
    }
}

async function doExecute(command) {
    elements.executeBtn.disabled = true;
    elements.executeBtn.innerHTML = '<div class="spinner"></div> Executing...';
    
    try {
        const response = await fetch(`${API_URL}/command/execute`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ command }),
        });
        
        const data = await response.json();
        
        const exitClass = data.returncode === 0 ? 'success' : 'error';
        
        elements.commandOutput.innerHTML = `
            <div class="output-header">
                <span>Output</span>
                <span class="exit-code ${exitClass}">Exit ${data.returncode}</span>
            </div>
            <pre>${escapeHtml(data.stdout || '(no output)')}</pre>
            ${data.stderr ? `<pre style="color: var(--accent-danger)">${escapeHtml(data.stderr)}</pre>` : ''}
        `;
        
        // Refresh history
        loadHistory();
        
    } catch (error) {
        elements.commandOutput.innerHTML = `<pre style="color: var(--accent-danger)">Error: ${error.message}</pre>`;
    } finally {
        elements.executeBtn.disabled = false;
        elements.executeBtn.innerHTML = '<i class="fas fa-play"></i> Execute';
    }
}

function showConfirmationModal(command, reason) {
    elements.confirmCommand.textContent = command;
    elements.confirmReason.textContent = reason;
    elements.confirmModal.classList.add('active');
}

function closeModal() {
    elements.confirmModal.classList.remove('active');
}

function confirmExecution() {
    closeModal();
    doExecute(currentCommand);
}

function copyCommand() {
    if (!currentCommand) return;
    navigator.clipboard.writeText(currentCommand);
    showToast('Command copied to clipboard', 'success');
}

function editCommand() {
    if (!currentCommand) return;
    elements.commandInput.value = currentCommand;
    elements.commandInput.focus();
}

// RAG Functions
async function indexFiles() {
    const directory = elements.indexPath.value.trim();
    if (!directory) return;
    
    elements.indexBtn.disabled = true;
    elements.indexBtn.innerHTML = '<div class="spinner"></div> Indexing...';
    
    try {
        const response = await fetch(`${API_URL}/rag/index`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                directory,
                recursive: elements.recursiveIndex.checked,
            }),
        });
        
        const data = await response.json();
        
        if (data.success) {
            elements.indexStats.innerHTML = `
                <div class="stat-item">
                    <span class="stat-label">Files Indexed:</span>
                    <span class="stat-value">${data.files_indexed}</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">Chunks Created:</span>
                    <span class="stat-value">${data.chunks_created}</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">Duration:</span>
                    <span class="stat-value">${data.duration_seconds.toFixed(2)}s</span>
                </div>
            `;
            elements.indexStats.classList.remove('hidden');
            showToast(`Indexed ${data.files_indexed} files successfully`, 'success');
        } else {
            showToast(`Indexing failed: ${data.errors.join(', ')}`, 'error');
        }
        
    } catch (error) {
        showToast('Error indexing files', 'error');
    } finally {
        elements.indexBtn.disabled = false;
        elements.indexBtn.innerHTML = '<i class="fas fa-database"></i> Index';
    }
}

async function searchRag() {
    const query = elements.ragQuery.value.trim();
    if (!query) return;
    
    elements.ragSearchBtn.disabled = true;
    elements.ragSearchBtn.innerHTML = '<div class="spinner"></div> Searching...';
    
    try {
        const response = await fetch(`${API_URL}/rag/search`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query, top_k: 5 }),
        });
        
        const data = await response.json();
        
        if (data.results && data.results.length > 0) {
            elements.searchResults.innerHTML = data.results.map((result, i) => `
                <div class="search-result-item">
                    <div class="search-result-header">
                        <span class="search-result-source">${escapeHtml(result.source)}</span>
                        <span class="search-result-score">Score: ${(result.score * 100).toFixed(1)}%</span>
                    </div>
                    <div class="search-result-content">${escapeHtml(result.content.substring(0, 300))}...</div>
                </div>
            `).join('');
        } else {
            elements.searchResults.innerHTML = '<p class="empty-state">No results found</p>';
        }
        
    } catch (error) {
        showToast('Error searching files', 'error');
    } finally {
        elements.ragSearchBtn.disabled = false;
        elements.ragSearchBtn.innerHTML = '<i class="fas fa-search"></i> Search';
    }
}

// History Functions
async function loadHistory() {
    try {
        const response = await fetch(`${API_URL}/command/history?limit=50`);
        const data = await response.json();
        
        if (data.commands && data.commands.length > 0) {
            elements.historyList.innerHTML = data.commands.map(item => {
                const result = item.result;
                let statusClass = 'success';
                let statusIcon = 'check';
                
                if (result.blocked) {
                    statusClass = 'blocked';
                    statusIcon = 'ban';
                } else if (result.returncode !== 0) {
                    statusClass = 'error';
                    statusIcon = 'times';
                }
                
                return `
                    <div class="history-item">
                        <div class="history-status ${statusClass}">
                            <i class="fas fa-${statusIcon}"></i>
                        </div>
                        <div class="history-command">${escapeHtml(result.command)}</div>
                        <div class="history-meta">${new Date(result.executed_at).toLocaleString()}</div>
                    </div>
                `;
            }).join('');
        } else {
            elements.historyList.innerHTML = '<p class="empty-state">No commands executed yet.</p>';
        }
        
    } catch (error) {
        elements.historyList.innerHTML = '<p class="empty-state">Failed to load history</p>';
    }
}

async function clearHistory() {
    if (!confirm('Are you sure you want to clear all history?')) return;
    
    try {
        await fetch(`${API_URL}/command/history`, { method: 'DELETE' });
        elements.historyList.innerHTML = '<p class="empty-state">History cleared.</p>';
        showToast('History cleared', 'success');
    } catch (error) {
        showToast('Error clearing history', 'error');
    }
}

// Settings Functions
async function testConnection() {
    const url = elements.apiUrl.value.trim();
    
    elements.testConnection.disabled = true;
    elements.testConnection.innerHTML = '<div class="spinner"></div> Testing...';
    
    try {
        const response = await fetch(`${url}/health`);
        
        if (response.ok) {
            showToast('Connection successful!', 'success');
            localStorage.setItem('apiUrl', url);
            checkHealth();
        } else {
            showToast('Connection failed', 'error');
        }
    } catch (error) {
        showToast('Connection failed: ' + error.message, 'error');
    } finally {
        elements.testConnection.disabled = false;
        elements.testConnection.innerHTML = '<i class="fas fa-plug"></i> Test Connection';
    }
}

function toggleTheme() {
    const current = document.documentElement.getAttribute('data-theme') || 'dark';
    const next = current === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', next);
    elements.themeSelect.value = next;
    localStorage.setItem('theme', next);
    
    const icon = elements.toggleTheme.querySelector('i');
    icon.className = next === 'dark' ? 'fas fa-moon' : 'fas fa-sun';
}

// Toast Notifications
function showToast(message, type = 'info') {
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    
    const icons = {
        success: 'check-circle',
        error: 'times-circle',
        warning: 'exclamation-triangle',
        info: 'info-circle',
    };
    
    toast.innerHTML = `
        <i class="fas fa-${icons[type]}"></i>
        <span>${message}</span>
    `;
    
    elements.toastContainer.appendChild(toast);
    
    setTimeout(() => {
        toast.style.opacity = '0';
        toast.style.transform = 'translateX(100%)';
        setTimeout(() => toast.remove(), 300);
    }, 4000);
}
