// Global state
const state = {
    services: [],       // Array of discovered/added service objects
    steps: [],          // Completed relay steps
    initialMessage: "", // The starting message
    showIntermediates: false,
    judgeAnalysis: "",  // Judge model analysis text
    similarityScores: [], // Cosine similarity to original per step
};

// ─── Authentication ─────────────────────────────────────────

let authToken = sessionStorage.getItem("authToken") || "";

async function authFetch(url, options = {}) {
    if (!options.headers) options.headers = {};
    if (authToken) {
        options.headers["Authorization"] = `Bearer ${authToken}`;
    }

    let resp = await fetch(url, options);

    if (resp.status === 401) {
        const key = prompt("This server requires an API key (APP_API_KEY):");
        if (!key) throw new Error("Authentication required");
        authToken = key.trim();
        sessionStorage.setItem("authToken", authToken);
        options.headers["Authorization"] = `Bearer ${authToken}`;
        resp = await fetch(url, options);
        if (resp.status === 401) {
            sessionStorage.removeItem("authToken");
            authToken = "";
            throw new Error("Invalid API key");
        }
    }

    return resp;
}

// ─── Input Validation ────────────────────────────────────────

const DANGEROUS_PATTERN = /<[^>]*>|javascript:|vbscript:|data:\w+\/\w+|expression\s*\(|on\w+\s*=/i;

function validateInput(value) {
    if (DANGEROUS_PATTERN.test(value)) {
        return "Input contains disallowed HTML or script content";
    }
    return null;
}

function showValidationError(elementId, message) {
    const el = document.getElementById(elementId);
    if (!el) return;
    el.textContent = message;
    el.classList.remove("hidden");
}

function clearValidationError(elementId) {
    const el = document.getElementById(elementId);
    if (!el) return;
    el.textContent = "";
    el.classList.add("hidden");
}

// ─── Service Loading ─────────────────────────────────────────

async function loadServices() {
    const statusEl = document.getElementById("services-status");
    const btn = document.getElementById("refresh-btn");

    statusEl.className = "status-msg info";
    statusEl.innerHTML = '<span class="spinner"></span> Loading services from .env hosts...';
    statusEl.classList.remove("hidden");
    btn.disabled = true;

    try {
        const resp = await authFetch("/api/services");
        const data = await resp.json();

        if (!resp.ok) {
            statusEl.className = "status-msg error";
            statusEl.textContent = data.error || "Failed to load services";
            return;
        }

        if (data.warning) {
            statusEl.className = "status-msg error";
            statusEl.textContent = data.warning;
            return;
        }

        // Reset and repopulate
        state.services = [];
        for (const svc of data.services) {
            addService(svc);
        }

        const modelCount = state.services.reduce((sum, s) => sum + s.models.length, 0);
        statusEl.className = "status-msg success";
        statusEl.textContent = `Found ${data.services.length} service(s) with ${modelCount} inference model(s)`;
    } catch (err) {
        statusEl.className = "status-msg error";
        statusEl.textContent = `Error: ${err.message}`;
    } finally {
        btn.disabled = false;
    }
}

function addService(svc) {
    const key = `${svc.host}:${svc.port}`;
    if (state.services.some(s => `${s.host}:${s.port}` === key)) return;

    state.services.push(svc);
    renderServices();
    refreshGlobalServiceDropdown();
    refreshJudgeServiceDropdown();
    updateAgentConfigs();
}

function renderServices() {
    const container = document.getElementById("services-list");
    container.innerHTML = "";

    for (const svc of state.services) {
        const card = document.createElement("div");
        card.className = "service-card";
        card.innerHTML = `
            <div class="service-info">
                <span class="service-badge ${escapeHtml(svc.service_type)}">${escapeHtml(svc.service_type)}</span>
                <span>${escapeHtml(svc.host)}:${escapeHtml(String(svc.port))}</span>
            </div>
            <span class="model-count">${escapeHtml(String(svc.models.length))} model(s)</span>
        `;
        container.appendChild(card);
    }
}

// ─── Prompt Style ────────────────────────────────────────────

function onPromptStyleChange() {
    const style = document.getElementById("prompt-style").value;
    const customField = document.getElementById("global-custom-field");
    if (style === "custom") {
        customField.classList.remove("hidden");
    } else {
        customField.classList.add("hidden");
    }
}

// ─── Model Mode ──────────────────────────────────────────────

function getModelMode() {
    return document.getElementById("model-mode").value;
}

function onModelModeChange() {
    const mode = getModelMode();
    const sameConfig = document.getElementById("same-model-config");

    if (mode === "same") {
        sameConfig.classList.remove("hidden");
    } else {
        sameConfig.classList.add("hidden");
    }

    updateAgentConfigs();
}

function refreshGlobalServiceDropdown() {
    const sel = document.getElementById("global-service");
    sel.innerHTML = '<option value="">-- Select --</option>';
    state.services.forEach((s, idx) => {
        const opt = document.createElement("option");
        opt.value = idx;
        opt.textContent = `${s.service_type} @ ${s.host}:${s.port}`;
        sel.appendChild(opt);
    });
}

function onGlobalServiceSelect() {
    const serviceIdx = parseInt(document.getElementById("global-service").value);
    const modelSelect = document.getElementById("global-model");
    modelSelect.innerHTML = "";

    if (isNaN(serviceIdx) || !state.services[serviceIdx]) {
        modelSelect.innerHTML = '<option value="">-- Select service first --</option>';
        return;
    }

    const svc = state.services[serviceIdx];
    for (const model of svc.models) {
        const opt = document.createElement("option");
        opt.value = model;
        opt.textContent = model;
        modelSelect.appendChild(opt);
    }
}

function getAllModels() {
    const models = [];
    for (const svc of state.services) {
        for (const model of svc.models) {
            models.push({ service: svc, model });
        }
    }
    return models;
}

// ─── Judge Configuration ─────────────────────────────────────

function onJudgeToggle() {
    const enabled = document.getElementById("judge-enabled").checked;
    const fields = document.getElementById("judge-fields");
    if (enabled) {
        fields.classList.remove("hidden");
    } else {
        fields.classList.add("hidden");
    }
}

function refreshJudgeServiceDropdown() {
    const sel = document.getElementById("judge-service");
    sel.innerHTML = '<option value="">-- Select --</option>';
    state.services.forEach((s, idx) => {
        const opt = document.createElement("option");
        opt.value = idx;
        opt.textContent = `${s.service_type} @ ${s.host}:${s.port}`;
        sel.appendChild(opt);
    });
}

function onJudgeServiceSelect() {
    const serviceIdx = parseInt(document.getElementById("judge-service").value);
    const modelSelect = document.getElementById("judge-model");
    modelSelect.innerHTML = "";

    if (isNaN(serviceIdx) || !state.services[serviceIdx]) {
        modelSelect.innerHTML = '<option value="">-- Select service first --</option>';
        return;
    }

    const svc = state.services[serviceIdx];
    for (const model of svc.models) {
        const opt = document.createElement("option");
        opt.value = model;
        opt.textContent = model;
        modelSelect.appendChild(opt);
    }
}

function getJudgeConfig() {
    if (!document.getElementById("judge-enabled").checked) return null;

    const serviceIdx = parseInt(document.getElementById("judge-service").value);
    if (isNaN(serviceIdx) || !state.services[serviceIdx]) return null;

    const svc = state.services[serviceIdx];
    const model = document.getElementById("judge-model").value;
    if (!model) return null;

    return {
        service_index: svc.index,
        model: model,
    };
}

// ─── Agent Configuration ─────────────────────────────────────

function updateAgentConfigs() {
    const count = parseInt(document.getElementById("agent-count").value) || 3;
    const container = document.getElementById("agent-configs");
    const mode = getModelMode();
    container.innerHTML = "";

    // In "same" mode, no per-agent rows needed at all
    if (mode === "same") return;

    // In "random" mode, show compact labels
    if (mode === "random") {
        for (let i = 0; i < count; i++) {
            const row = document.createElement("div");
            row.className = "agent-row agent-row-compact";
            row.innerHTML = `<div class="agent-row-header">Agent ${escapeHtml(String(i + 1))} <span class="agent-mode-label">(random model)</span></div>`;
            container.appendChild(row);
        }
        return;
    }

    // "per-agent" mode — show service + model dropdowns per agent
    for (let i = 0; i < count; i++) {
        const row = document.createElement("div");
        row.className = "agent-row";
        row.innerHTML = `
            <div class="agent-row-header">Agent ${escapeHtml(String(i + 1))}</div>
            <div class="agent-row-fields">
                <div class="input-group">
                    <label>Service</label>
                    <select id="service-${i}" onchange="onServiceSelect(${i})">
                        <option value="">-- Select --</option>
                        ${state.services.map((s, idx) =>
                            `<option value="${idx}">${escapeHtml(s.service_type)} @ ${escapeHtml(s.host)}:${escapeHtml(String(s.port))}</option>`
                        ).join("")}
                    </select>
                </div>
                <div class="input-group">
                    <label>Model</label>
                    <select id="model-${i}">
                        <option value="">-- Select service first --</option>
                    </select>
                </div>
            </div>
        `;
        container.appendChild(row);
    }
}

function onServiceSelect(agentIndex) {
    const serviceIdx = parseInt(document.getElementById(`service-${agentIndex}`).value);
    const modelSelect = document.getElementById(`model-${agentIndex}`);
    modelSelect.innerHTML = "";

    if (isNaN(serviceIdx) || !state.services[serviceIdx]) {
        modelSelect.innerHTML = '<option value="">-- Select service first --</option>';
        return;
    }

    const svc = state.services[serviceIdx];
    for (const model of svc.models) {
        const opt = document.createElement("option");
        opt.value = model;
        opt.textContent = model;
        modelSelect.appendChild(opt);
    }
}

// ─── Game Execution ──────────────────────────────────────────

async function startGame() {
    const message = document.getElementById("initial-message").value.trim();
    clearValidationError("message-validation");

    // Client-side validation
    if (!message) {
        showValidationError("message-validation", "Please enter an initial message.");
        return;
    }
    const msgErr = validateInput(message);
    if (msgErr) {
        showValidationError("message-validation", msgErr);
        return;
    }

    // Global prompt style
    const promptStyle = document.getElementById("prompt-style").value;
    const customPrompt = document.getElementById("custom-prompt")?.value || "";
    if (promptStyle === "custom" && customPrompt) {
        const cpErr = validateInput(customPrompt);
        if (cpErr) {
            alert(`Custom prompt: ${cpErr}`);
            return;
        }
    }

    // Temperature
    const tempInput = document.getElementById("temperature").value;
    let temperature = null;
    if (tempInput !== "") {
        temperature = parseFloat(tempInput);
        if (isNaN(temperature) || temperature < 0 || temperature > 1) {
            alert("Temperature must be between 0 and 1.");
            return;
        }
    }

    const count = parseInt(document.getElementById("agent-count").value) || 3;
    const mode = getModelMode();
    const agents = [];

    // Resolve global service+model for "same" mode
    let globalSvc = null;
    let globalModel = null;
    if (mode === "same") {
        const gIdx = parseInt(document.getElementById("global-service").value);
        if (isNaN(gIdx) || !state.services[gIdx]) {
            alert("Please select a service for all agents.");
            return;
        }
        globalSvc = state.services[gIdx];
        globalModel = document.getElementById("global-model").value;
        if (!globalModel) {
            alert("Please select a model for all agents.");
            return;
        }
    }

    // Build all models list for random mode
    const allModels = getAllModels();
    if (mode === "random" && allModels.length === 0) {
        alert("No models available. Load services first.");
        return;
    }

    for (let i = 0; i < count; i++) {
        let svc, model;

        if (mode === "same") {
            svc = globalSvc;
            model = globalModel;
        } else if (mode === "random") {
            const pick = allModels[Math.floor(Math.random() * allModels.length)];
            svc = pick.service;
            model = pick.model;
        } else {
            // per-agent
            const serviceIdx = parseInt(document.getElementById(`service-${i}`).value);
            if (isNaN(serviceIdx) || !state.services[serviceIdx]) {
                alert(`Agent ${i + 1}: please select a service.`);
                return;
            }
            svc = state.services[serviceIdx];
            model = document.getElementById(`model-${i}`).value;
            if (!model) {
                alert(`Agent ${i + 1}: please select a model.`);
                return;
            }
        }

        agents.push({
            service_index: svc.index,
            model: model,
        });
    }

    // Validate judge config if enabled
    const judgeEnabled = document.getElementById("judge-enabled").checked;
    let judgeConfig = null;
    if (judgeEnabled) {
        judgeConfig = getJudgeConfig();
        if (!judgeConfig) {
            alert("Judge is enabled but no service/model selected.");
            return;
        }
    }

    state.initialMessage = message;
    state.steps = [];
    state.showIntermediates = false;
    state.judgeAnalysis = "";
    state.similarityScores = [];

    // Show results panel
    const resultsPanel = document.getElementById("results-panel");
    resultsPanel.classList.remove("hidden");
    document.getElementById("progress-area").innerHTML = "";
    document.getElementById("similarity-area").classList.add("hidden");
    document.getElementById("diff-area").classList.add("hidden");
    document.getElementById("judge-area").classList.add("hidden");
    document.getElementById("results-actions").classList.add("hidden");

    const startBtn = document.getElementById("start-btn");
    startBtn.disabled = true;
    startBtn.textContent = "Running...";

    try {
        const body = {
            message,
            agents,
            prompt_style: promptStyle,
            custom_prompt: customPrompt,
            judge: judgeConfig,
        };
        if (temperature !== null) {
            body.temperature = temperature;
        }

        const resp = await authFetch("/api/start", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body),
        });

        const reader = resp.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split("\n");
            buffer = lines.pop();

            for (const line of lines) {
                if (!line.startsWith("data: ")) continue;
                let payload;
                try {
                    payload = JSON.parse(line.slice(6));
                } catch {
                    continue;
                }

                if (payload.type === "step") {
                    state.steps.push(payload.step);
                    renderStep(payload.step);
                    if (payload.step.cosine_similarity != null) {
                        state.similarityScores.push(payload.step.cosine_similarity);
                        document.getElementById("similarity-area").classList.remove("hidden");
                        renderSimilarityChart();
                    }
                } else if (payload.type === "diff") {
                    renderDiff(payload.diff);
                } else if (payload.type === "judge_start") {
                    renderJudgeLoading();
                } else if (payload.type === "judge") {
                    state.judgeAnalysis = payload.analysis;
                    renderJudgeAnalysis(payload.analysis);
                } else if (payload.type === "judge_error") {
                    renderJudgeError(payload.error);
                } else if (payload.type === "error") {
                    renderError(payload.error);
                } else if (payload.type === "done") {
                    document.getElementById("results-actions").classList.remove("hidden");
                }
            }
        }
    } catch (err) {
        renderError(err.message);
    } finally {
        startBtn.disabled = false;
        startBtn.textContent = "Start Game";
    }
}

function renderStep(step) {
    const area = document.getElementById("progress-area");

    const card = document.createElement("div");
    card.className = "step-card";
    card.dataset.stepIndex = step.agent_index;
    card.innerHTML = `
        <div class="step-card-header">
            <span class="step-label">Agent ${escapeHtml(String(step.agent_index + 1))}</span>
            <span class="step-meta">${escapeHtml(step.model)} / ${escapeHtml(step.prompt_style)}</span>
        </div>
        <div class="step-input">
            <div class="step-input-label">Input</div>
            ${escapeHtml(step.input_message)}
        </div>
        <div class="step-output">
            <div class="step-output-label">Output</div>
            ${escapeHtml(step.output_message)}
        </div>
    `;
    area.appendChild(card);
    card.scrollIntoView({ behavior: "smooth", block: "end" });
}

function renderDiff(diffData) {
    const diffArea = document.getElementById("diff-area");
    const diffDisplay = document.getElementById("diff-display");
    diffArea.classList.remove("hidden");

    let html = "";
    for (const [tag, word] of diffData) {
        if (tag === "equal") {
            html += escapeHtml(word) + " ";
        } else if (tag === "insert") {
            html += `<span class="diff-insert">${escapeHtml(word)}</span> `;
        } else if (tag === "delete") {
            html += `<span class="diff-delete">${escapeHtml(word)}</span> `;
        }
    }
    diffDisplay.innerHTML = html;
}

function renderError(message) {
    const area = document.getElementById("progress-area");
    const div = document.createElement("div");
    div.className = "status-msg error";
    div.textContent = `Error: ${message}`;
    area.appendChild(div);
}

// ─── Judge Rendering ─────────────────────────────────────────

function renderJudgeLoading() {
    const area = document.getElementById("judge-area");
    const display = document.getElementById("judge-display");
    area.classList.remove("hidden");
    display.innerHTML = '<span class="spinner"></span> Judge is analyzing the results...';
}

function renderJudgeAnalysis(analysis) {
    const area = document.getElementById("judge-area");
    const display = document.getElementById("judge-display");
    area.classList.remove("hidden");
    display.textContent = analysis;
}

function renderJudgeError(error) {
    const area = document.getElementById("judge-area");
    const display = document.getElementById("judge-display");
    area.classList.remove("hidden");
    display.innerHTML = "";
    const div = document.createElement("div");
    div.className = "status-msg error";
    div.textContent = `Judge error: ${error}`;
    display.appendChild(div);
}

// ─── Results Actions ─────────────────────────────────────────

function toggleIntermediates() {
    state.showIntermediates = !state.showIntermediates;
    const cards = document.querySelectorAll(".step-card");
    cards.forEach(card => card.style.display = "block");
}

async function exportChain() {
    try {
        const resp = await authFetch("/api/export", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                message: state.initialMessage,
                steps: state.steps,
                judge_analysis: state.judgeAnalysis,
            }),
        });
        const blob = await resp.blob();
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = "telephone_chain.txt";
        a.click();
        URL.revokeObjectURL(url);
    } catch (err) {
        alert(`Export failed: ${err.message}`);
    }
}

// ─── Similarity Chart ────────────────────────────────────────

function renderSimilarityChart() {
    const canvas = document.getElementById("similarity-chart");
    const container = canvas.parentElement;
    const dpr = window.devicePixelRatio || 1;
    const w = container.clientWidth;
    const h = 250;
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    canvas.style.width = w + "px";
    canvas.style.height = h + "px";

    const ctx = canvas.getContext("2d");
    ctx.scale(dpr, dpr);

    const scores = state.similarityScores;
    const n = scores.length;
    if (n === 0) return;

    // Layout
    const pad = { top: 20, right: 50, bottom: 35, left: 50 };
    const plotW = w - pad.left - pad.right;
    const plotH = h - pad.top - pad.bottom;

    // Background
    ctx.fillStyle = "#0d1117";
    ctx.fillRect(0, 0, w, h);

    // Gridlines and Y labels
    ctx.strokeStyle = "#21262d";
    ctx.lineWidth = 1;
    ctx.fillStyle = "#8b949e";
    ctx.font = "11px -apple-system, BlinkMacSystemFont, sans-serif";
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    for (let v = 0; v <= 1.0; v += 0.25) {
        const y = pad.top + plotH * (1 - v);
        ctx.beginPath();
        ctx.moveTo(pad.left, y);
        ctx.lineTo(pad.left + plotW, y);
        ctx.stroke();
        ctx.fillText(v.toFixed(2), pad.left - 8, y);
    }

    // X axis labels
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    const maxTicks = Math.min(n, 15);
    const tickStep = Math.max(1, Math.floor(n / maxTicks));
    for (let i = 0; i < n; i += tickStep) {
        const x = pad.left + (n === 1 ? plotW / 2 : (i / (n - 1)) * plotW);
        ctx.fillText(String(i + 1), x, pad.top + plotH + 8);
    }
    // Always label last point
    if (n > 1 && (n - 1) % tickStep !== 0) {
        const x = pad.left + plotW;
        ctx.fillText(String(n), x, pad.top + plotH + 8);
    }

    // X axis title
    ctx.fillStyle = "#8b949e";
    ctx.fillText("Agent #", pad.left + plotW / 2, pad.top + plotH + 22);

    // Plot line
    ctx.strokeStyle = "#58a6ff";
    ctx.lineWidth = 2;
    ctx.lineJoin = "round";
    ctx.beginPath();
    for (let i = 0; i < n; i++) {
        const x = pad.left + (n === 1 ? plotW / 2 : (i / (n - 1)) * plotW);
        const y = pad.top + plotH * (1 - scores[i]);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // Dot markers
    ctx.fillStyle = "#58a6ff";
    for (let i = 0; i < n; i++) {
        const x = pad.left + (n === 1 ? plotW / 2 : (i / (n - 1)) * plotW);
        const y = pad.top + plotH * (1 - scores[i]);
        ctx.beginPath();
        ctx.arc(x, y, 4, 0, Math.PI * 2);
        ctx.fill();
    }

    // Annotate last point value
    const lastX = pad.left + (n === 1 ? plotW / 2 : plotW);
    const lastY = pad.top + plotH * (1 - scores[n - 1]);
    ctx.fillStyle = "#e1e4e8";
    ctx.font = "bold 12px -apple-system, BlinkMacSystemFont, sans-serif";
    ctx.textAlign = "left";
    ctx.textBaseline = "bottom";
    ctx.fillText(scores[n - 1].toFixed(3), lastX + 8, lastY - 4);
}

// ─── Helpers ─────────────────────────────────────────────────

function escapeHtml(text) {
    if (text == null) return "";
    const div = document.createElement("div");
    div.textContent = String(text);
    return div.innerHTML;
}

// ─── Init ────────────────────────────────────────────────────

document.addEventListener("DOMContentLoaded", () => {
    updateAgentConfigs();
    loadServices();
});
