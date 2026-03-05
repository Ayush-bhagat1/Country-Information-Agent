// Frontend logic for the Country Information Agent

const chatArea = document.getElementById('chatArea');
const messages = document.getElementById('messages');
const welcome = document.getElementById('welcome');
const queryInput = document.getElementById('queryInput');
const sendBtn = document.getElementById('sendBtn');
const modal = document.getElementById('modal');
const pipelineContent = document.getElementById('pipelineContent');

let loading = false;
let msgId = 0;

// Send on Enter
queryInput.addEventListener('keydown', function (e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendQuery();
    }
});

async function sendQuery() {
    const query = queryInput.value.trim();
    if (!query || loading) return;

    welcome.classList.add('hidden');
    addMessage('user', query);
    queryInput.value = '';

    const loadId = addLoading();
    loading = true;
    sendBtn.disabled = true;

    try {
        const res = await fetch('/api/query', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query }),
        });

        removeMsg(loadId);

        if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            addAgentMsg({
                answer: err.detail || 'Something went wrong. Please try again.',
                status: 'error',
                pipeline_steps: [],
            });
            return;
        }

        const data = await res.json();
        addAgentMsg(data);

    } catch (err) {
        removeMsg(loadId);
        addAgentMsg({
            answer: 'Could not reach the server. Make sure the backend is running.',
            status: 'error',
            pipeline_steps: [],
        });
    } finally {
        loading = false;
        sendBtn.disabled = false;
        queryInput.focus();
    }
}

function askExample(btn) {
    // Get the text content but skip the emoji span
    const spans = btn.childNodes;
    let text = '';
    for (const node of spans) {
        if (node.nodeType === Node.TEXT_NODE) {
            text += node.textContent;
        } else if (node.tagName !== 'SPAN') {
            text += node.textContent;
        }
    }
    // fallback: just grab everything after the emoji
    if (!text.trim()) {
        text = btn.textContent.trim().substring(2).trim();
    }
    queryInput.value = text.trim();
    sendQuery();
}

// Message rendering
function addMessage(type, text) {
    const id = 'msg-' + (++msgId);
    const div = document.createElement('div');
    div.id = id;
    div.className = 'msg msg-' + type;
    div.innerHTML = '<div class="bubble">' + escapeHtml(text) + '</div>';
    messages.appendChild(div);
    scrollDown();
    return id;
}

function addAgentMsg(data) {
    const id = 'msg-' + (++msgId);
    const div = document.createElement('div');
    div.id = id;
    const isError = data.status === 'error' || data.error;
    div.className = 'msg msg-agent' + (isError ? ' msg-error' : '');

    let html = '<div class="bubble">';

    // Show country header with flag if available
    if (data.country && data.data && data.data.flag_png) {
        html += '<div class="country-hdr">';
        html += '<img src="' + data.data.flag_png + '" alt="flag">';
        html += '<span>' + escapeHtml(data.country) + '</span>';
        html += '</div>';
    } else if (data.country) {
        html += '<div class="country-hdr"><span>' + escapeHtml(data.country) + '</span></div>';
    }

    // Format the answer text
    let answerHtml = escapeHtml(data.answer || 'No answer.');
    answerHtml = answerHtml.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    answerHtml = answerHtml.replace(/^• /gm, '&bull; ');
    html += '<div class="answer">' + answerHtml + '</div>';
    html += '</div>';

    // Pipeline button and latency
    if (data.pipeline_steps && data.pipeline_steps.length > 0) {
        const stepsStr = JSON.stringify(data.pipeline_steps).replace(/"/g, '&quot;');
        html += '<div class="meta-row">';
        html += '<button class="pipeline-btn" onclick=\'showPipeline(' + stepsStr + ')\'>';
        html += '⚡ Pipeline (' + data.pipeline_steps.length + ' steps)</button>';
        if (data.latency_ms) {
            html += '<span class="latency">' + Math.round(data.latency_ms) + 'ms</span>';
        }
        html += '</div>';
    }

    div.innerHTML = html;
    messages.appendChild(div);
    scrollDown();
}

function addLoading() {
    const id = 'msg-' + (++msgId);
    const div = document.createElement('div');
    div.id = id;
    div.className = 'msg msg-agent loading';
    div.innerHTML = '<div class="bubble"><div class="dots"><span></span><span></span><span></span></div> Thinking...</div>';
    messages.appendChild(div);
    scrollDown();
    return id;
}

function removeMsg(id) {
    const el = document.getElementById(id);
    if (el) el.remove();
}

// Pipeline modal
function showPipeline(steps) {
    const icons = { success: '✓', failed: '✗', skipped: '⊘', error_response: '!' };
    let html = '';

    steps.forEach(function (s) {
        html += '<div class="step">';
        html += '<div class="step-icon ' + s.status + '">' + (icons[s.status] || '•') + '</div>';
        html += '<div class="step-info">';
        html += '<div class="step-name">' + escapeHtml(s.step) + '</div>';
        html += '<div class="step-detail">' + escapeHtml(s.detail || '') + '</div>';
        html += '</div></div>';
    });

    pipelineContent.innerHTML = html;
    modal.classList.add('open');
}

function closeModal(e) {
    if (e.target === modal) modal.classList.remove('open');
}

// Helpers
function escapeHtml(str) {
    const d = document.createElement('div');
    d.textContent = str;
    return d.innerHTML;
}

function scrollDown() {
    requestAnimationFrame(function () {
        chatArea.scrollTop = chatArea.scrollHeight;
    });
}
