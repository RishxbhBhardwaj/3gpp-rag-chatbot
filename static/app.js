// ═══════════════════════════════════════════════════════════════
// 3GPP Standards Intelligence — Chat Application
// ═══════════════════════════════════════════════════════════════

const API_URL = "/api/query";
const STATUS_URL = "/api/status";

// ─── DOM Elements ────────────────────────────────────────────
const chatArea = document.getElementById("chatArea");
const landing = document.getElementById("landing");
const messagesDiv = document.getElementById("messages");
const inputBox = document.getElementById("inputBox");
const btnSend = document.getElementById("btnSend");
const btnNewChat = document.getElementById("btnNewChat");
const btnToggleSidebar = document.getElementById("btnToggleSidebar");
const btnMic = document.getElementById("btnMic");
const btnThemeToggle = document.getElementById("btnThemeToggle");
const themeIcon = document.getElementById("themeIcon");
const sidebar = document.getElementById("sidebar");
const sbSpecs = document.getElementById("sbSpecs");
const statDocs = document.getElementById("statDocs");
const statChunks = document.getElementById("statChunks");

// ─── State ───────────────────────────────────────────────────
let isLoading = false;
let isRecording = false;
let recognition = null;
let synthesis = window.speechSynthesis;

// ─── Initialize ──────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
    loadStatus();
    setupListeners();
    autoResize(inputBox);
    loadTheme();
});

// ─── Load System Status ──────────────────────────────────────
async function loadStatus() {
    try {
        const res = await fetch(STATUS_URL);
        const data = await res.json();

        statDocs.textContent = data.total_specs || 7;
        statChunks.textContent = (data.total_chunks || 3004).toLocaleString();

        // Populate specs list
        if (data.specs && sbSpecs) {
            const specTitles = {
                "TS 23.501": "5G System Architecture",
                "TS 23.502": "5G System Procedures",
                "TS 23.503": "Policy & Charging Control",
                "TS 38.300": "NR & NG-RAN Overview",
                "TS 38.331": "Radio Resource Control (RRC)",
                "TS 24.501": "5G NAS Protocol",
                "TS 29.500": "Service Based Architecture",
                "TS 33.501": "5G Security Architecture",
            };
            sbSpecs.innerHTML = "";
            for (const [spec, title] of Object.entries(data.specs)) {
                const displayTitle = specTitles[spec] || title;
                sbSpecs.innerHTML += `<div class="sb-spec-item"><span class="sb-spec-dot"></span><strong>${spec}</strong>&nbsp;— ${displayTitle}</div>`;
            }
        }
    } catch (e) {
        console.log("Status fetch failed:", e);
    }
}

// ─── Event Listeners ─────────────────────────────────────────
function setupListeners() {
    // Send button
    btnSend.addEventListener("click", sendMessage);

    // Enter to send, Shift+Enter for newline
    inputBox.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    // Enable/disable send button
    inputBox.addEventListener("input", () => {
        btnSend.disabled = !inputBox.value.trim();
        autoResize(inputBox);
    });

    // New chat
    btnNewChat.addEventListener("click", clearChat);

    // Sidebar toggle
    btnToggleSidebar.addEventListener("click", () => {
        sidebar.classList.toggle("collapsed");
    });

    // Close sidebar on mobile when clicking outside
    document.addEventListener("click", (e) => {
        if (window.innerWidth <= 768) {
            if (!sidebar.contains(e.target) && e.target !== btnToggleSidebar) {
                sidebar.classList.add("collapsed");
            }
        }
    });

    // Example cards
    document.querySelectorAll(".example-card").forEach((card) => {
        card.addEventListener("click", () => {
            const query = card.dataset.query;
            inputBox.value = query;
            btnSend.disabled = false;
            sendMessage();
        });
    });

    // Mic button — Speech to Text
    btnMic.addEventListener("click", toggleVoiceInput);

    // Theme toggle
    btnThemeToggle.addEventListener("click", toggleTheme);
}

// ─── Send Message ────────────────────────────────────────────
async function sendMessage() {
    const question = inputBox.value.trim();
    if (!question || isLoading) return;

    isLoading = true;
    btnSend.disabled = true;
    inputBox.value = "";
    autoResize(inputBox);

    // Hide landing
    landing.classList.add("hidden");

    // Add user message
    appendMessage("user", question);

    // Show loading
    const loadingEl = showLoading();

    try {
        const res = await fetch(API_URL, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ question }),
        });

        if (!res.ok) throw new Error(`HTTP ${res.status}`);

        const data = await res.json();
        loadingEl.remove();

        // Add assistant message
        appendAssistantMessage(data);
    } catch (err) {
        loadingEl.remove();
        appendMessage("assistant", "⚠️ Failed to get response. Please check the server is running.");
        console.error("Query error:", err);
    }

    isLoading = false;
    scrollToBottom();
}

// ─── Append User Message ─────────────────────────────────────
function appendMessage(role, text) {
    const div = document.createElement("div");
    div.className = `message ${role}`;

    const avatar = role === "user" ? "👤" : "⚡";
    const roleName = role === "user" ? "You" : "3GPP Intelligence";

    div.innerHTML = `
        <div class="message-header">
            <div class="message-avatar">${avatar}</div>
            <div class="message-role">${roleName}</div>
        </div>
        <div class="message-body">${formatText(text)}</div>
    `;

    messagesDiv.appendChild(div);
    scrollToBottom();
}

// ─── Append Assistant Message (with sources) ─────────────────
function appendAssistantMessage(data) {
    const div = document.createElement("div");
    div.className = "message assistant";

    let sourcesHTML = "";
    if (data.sources && data.sources.length > 0 && data.decision !== "greeting") {
        sourcesHTML = buildSourcesTable(data.sources);
    }

    let noEvidenceHTML = "";
    if (data.decision === "refuse") {
        noEvidenceHTML = `
            <div class="no-evidence">
                <div class="no-evidence-title">Insufficient Evidence</div>
                <div class="no-evidence-text">The indexed 3GPP standards do not contain enough supporting information to answer this reliably.</div>
                <div class="no-evidence-tips">
                    <span>💡 Try: asking a more specific question</span>
                    <span>💡 Specifying a 3GPP specification (e.g., TS 23.501)</span>
                    <span>💡 Mentioning a specific procedure or section</span>
                </div>
            </div>
        `;
    }

    div.innerHTML = `
        <div class="message-header">
            <div class="message-avatar">⚡</div>
            <div class="message-role">3GPP Intelligence</div>
        </div>
        <div class="message-body">${formatText(data.response)}</div>
        <div class="message-actions" style="padding-left: 38px;">
            <button class="btn-speak" onclick="speakText(\`${data.response.replace(/`/g, "'").replace(/\\/g, "\\\\")}\`, this)">🔊 Read Aloud</button>
        </div>
        ${sourcesHTML}
        ${noEvidenceHTML}
    `;

    messagesDiv.appendChild(div);
    scrollToBottom();
}

// ─── Build Sources Table ─────────────────────────────────────
function buildSourcesTable(sources) {
    // Deduplicate sources
    const seen = new Set();
    const unique = [];
    for (const s of sources) {
        const key = `${s.spec}-${s.page}-${s.section}`;
        if (!seen.has(key)) {
            seen.add(key);
            unique.push(s);
        }
    }

    let rows = "";
    for (const s of unique) {
        rows += `
            <tr>
                <td><span class="spec-badge">${s.spec}</span></td>
                <td>${s.section || "—"}</td>
                <td>${s.page}</td>
            </tr>
        `;
    }

    return `
        <div class="sources-section">
            <div class="sources-header">📄 Sources & Evidence</div>
            <table class="sources-table">
                <thead>
                    <tr>
                        <th>Specification</th>
                        <th>Section</th>
                        <th>Page</th>
                    </tr>
                </thead>
                <tbody>${rows}</tbody>
            </table>
        </div>
    `;
}

// ─── Show Loading ────────────────────────────────────────────
function showLoading() {
    const div = document.createElement("div");
    div.className = "message assistant";
    div.innerHTML = `
        <div class="message-header">
            <div class="message-avatar">⚡</div>
            <div class="message-role">3GPP Intelligence</div>
        </div>
        <div class="loading-msg">
            <div class="loading-dots">
                <span></span><span></span><span></span>
            </div>
            Searching 3GPP specifications...
        </div>
    `;
    messagesDiv.appendChild(div);
    scrollToBottom();
    return div;
}

// ─── Format Text (Markdown to HTML) ──────────────────────────
function formatText(text) {
    if (!text) return "";

    // Escape HTML
    let html = text
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;");

    // Headers
    html = html.replace(/^### (.+)$/gm, "<h4>$1</h4>");
    html = html.replace(/^## (.+)$/gm, "<h3>$1</h3>");
    html = html.replace(/^# (.+)$/gm, "<h2>$1</h2>");

    // Bold
    html = html.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");

    // Italic
    html = html.replace(/\*(.+?)\*/g, "<em>$1</em>");

    // Inline code
    html = html.replace(/`(.+?)`/g, "<code>$1</code>");

    // Process line by line for lists
    const lines = html.split("\n");
    let result = [];
    let inList = false;
    let listType = "";

    for (let i = 0; i < lines.length; i++) {
        const line = lines[i];

        // Bullet list items (-, •, *)
        const bulletMatch = line.match(/^[\-\•\*]\s+(.+)/);
        // Numbered list items
        const numMatch = line.match(/^\d+\.\s+(.+)/);

        if (bulletMatch) {
            if (!inList || listType !== "ul") {
                if (inList) result.push(`</${listType}>`);
                result.push("<ul>");
                inList = true;
                listType = "ul";
            }
            result.push(`<li>${bulletMatch[1]}</li>`);
        } else if (numMatch) {
            if (!inList || listType !== "ol") {
                if (inList) result.push(`</${listType}>`);
                result.push("<ol>");
                inList = true;
                listType = "ol";
            }
            result.push(`<li>${numMatch[1]}</li>`);
        } else {
            if (inList) {
                result.push(`</${listType}>`);
                inList = false;
                listType = "";
            }
            result.push(line);
        }
    }
    if (inList) result.push(`</${listType}>`);

    html = result.join("\n");

    // Paragraphs (double newlines)
    html = html
        .split("\n\n")
        .map((block) => {
            block = block.trim();
            if (!block) return "";
            if (block.startsWith("<ul>") || block.startsWith("<ol>") || 
                block.startsWith("<h") || block.startsWith("<li>")) return block;
            return `<p>${block.replace(/\n/g, "<br>")}</p>`;
        })
        .join("");

    // Clean up any remaining single newlines in paragraphs
    html = html.replace(/<\/p>\n<p>/g, "</p><p>");

    return html;
}

// ─── Clear Chat ──────────────────────────────────────────────
function clearChat() {
    messagesDiv.innerHTML = "";
    landing.classList.remove("hidden");
}

// ─── Auto-resize Textarea ────────────────────────────────────
function autoResize(el) {
    el.style.height = "auto";
    el.style.height = Math.min(el.scrollHeight, 150) + "px";
}

// ─── Scroll to Bottom ────────────────────────────────────────
function scrollToBottom() {
    requestAnimationFrame(() => {
        chatArea.scrollTop = chatArea.scrollHeight;
    });
}

// ─── Voice Input (Speech-to-Text) ────────────────────────────
function toggleVoiceInput() {
    if (isRecording) {
        stopRecording();
    } else {
        startRecording();
    }
}

function startRecording() {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
        alert("Voice input is not supported in this browser.\n\nPlease use Google Chrome for voice support.");
        return;
    }

    try {
        recognition = new SpeechRecognition();
        recognition.lang = "en-US";
        recognition.continuous = true;
        recognition.interimResults = true;

        recognition.onstart = () => {
            isRecording = true;
            btnMic.classList.add("recording");
            inputBox.placeholder = "🎤 Listening... speak now";
        };

        recognition.onresult = (event) => {
            let finalTranscript = "";
            let interimTranscript = "";
            for (let i = 0; i < event.results.length; i++) {
                if (event.results[i].isFinal) {
                    finalTranscript += event.results[i][0].transcript;
                } else {
                    interimTranscript += event.results[i][0].transcript;
                }
            }
            inputBox.value = finalTranscript || interimTranscript;
            btnSend.disabled = !inputBox.value.trim();
            autoResize(inputBox);
        };

        recognition.onerror = (event) => {
            console.error("Speech error:", event.error);
            if (event.error === "not-allowed" || event.error === "permission-denied") {
                alert("Microphone blocked.\n\n1. Click the lock icon in the address bar\n2. Set Microphone to Allow\n3. Reload the page");
            } else if (event.error === "no-speech") {
                // User didn't speak — just stop silently
            } else if (event.error === "network") {
                alert("Voice input requires internet connection for speech recognition service.");
            }
            // Don't alert for other errors — just stop
            stopRecording();
        };

        recognition.onend = () => {
            stopRecording();
            // Auto-send if there's text after speech ends
            if (inputBox.value.trim()) {
                sendMessage();
            }
        };

        recognition.start();
    } catch (err) {
        console.error("Failed:", err);
        // Fallback: try requesting mic permission explicitly
        navigator.mediaDevices.getUserMedia({ audio: true })
            .then((stream) => {
                stream.getTracks().forEach(track => track.stop());
                // Retry after getting permission
                alert("Microphone permission granted. Please click the mic button again.");
            })
            .catch(() => {
                alert("Microphone access denied.\n\nPlease allow microphone in browser settings.");
            });
    }
}

function stopRecording() {
    isRecording = false;
    btnMic.classList.remove("recording");
    inputBox.placeholder = "Ask about 3GPP standards, procedures, specifications...";
    if (recognition) {
        recognition.stop();
        recognition = null;
    }
}

// ─── Text-to-Speech (Read Response Aloud) ────────────────────
function speakText(text, button) {
    // If already speaking, stop
    if (synthesis.speaking) {
        synthesis.cancel();
        document.querySelectorAll(".btn-speak").forEach(b => b.classList.remove("speaking"));
        return;
    }

    // Clean text for speech
    const cleanText = text
        .replace(/\*\*/g, "")
        .replace(/`/g, "")
        .replace(/📚.*$/gm, "")
        .replace(/[#\-\|]/g, "")
        .replace(/\s+/g, " ")
        .trim();

    const utterance = new SpeechSynthesisUtterance(cleanText);
    utterance.lang = "en-US";
    utterance.rate = 0.95;
    utterance.pitch = 1;

    utterance.onstart = () => {
        if (button) button.classList.add("speaking");
    };

    utterance.onend = () => {
        if (button) button.classList.remove("speaking");
    };

    utterance.onerror = () => {
        if (button) button.classList.remove("speaking");
    };

    synthesis.speak(utterance);
}

// ─── Theme Toggle (Light/Dark) ───────────────────────────────
function loadTheme() {
    const saved = localStorage.getItem("theme");
    if (saved === "light") {
        document.body.classList.add("light");
        themeIcon.textContent = "🌙";
    } else {
        document.body.classList.remove("light");
        themeIcon.textContent = "☀️";
    }
}

function toggleTheme() {
    document.body.classList.toggle("light");
    const isLight = document.body.classList.contains("light");
    localStorage.setItem("theme", isLight ? "light" : "dark");
    themeIcon.textContent = isLight ? "🌙" : "☀️";
}
