// Reply Options
//
// After an assistant turn, replay the exact chat-completion request that was
// just sent and append only a tail: the finished assistant reply, then one
// short user instruction. The snapshot is the JSON body of the outgoing
// /api/backends/chat-completions/generate fetch, after prompt listeners have
// finished. Do not call Generate(), generateQuietPrompt(), or
// sendGenerationRequest(). Those rebuild the prompt, can drop a different
// oldest message, omit tools, and commit world-info state. Older messages
// stay cacheable only if every message before that tail is unchanged.

const MODULE = 'reply_options';
const LOG = '[Reply Options]';
const ENDPOINT = '/api/backends/chat-completions/generate';
const SKIP_SNAPSHOT_TYPES = new Set(['quiet', 'impersonate']);
const MIN_COUNT = 4;
const MAX_COUNT = 8;
const MIN_TOKENS = 32;
const MAX_TOKENS = 4096;

const LEGACY_INSTRUCTION = 'Suggest {{count}} distinct replies {{user}} could send next, in character. Output only a numbered list, one reply per line, no quotes or commentary.';
const INSTRUCTION_VERSION = 2;
const DEFAULT_INSTRUCTION = '[OOC. Not a story turn. Do not continue as {{char}}.]\n'
    + 'Write exactly {{count}} next chat messages for {{user}}, one per line, numbered "1." "2." and so on. '
    + 'Each line must be the complete message {{user}} would type, in the same style as this chat: spoken words and actions only. '
    + 'Do not analyze the scene. Do not summarize moods, intentions, or state. Do not add headings, labels, or name prefixes.';

const defaultSettings = {
    enabled: true,
    count: 6,
    maxTokens: 600,
    instruction: DEFAULT_INSTRUCTION,
};

/** @type {{ chatKey: string, type: string, body: Record<string, any> } | null} */
let snapshot = null;
/** @type {AbortController | null} */
let abortController = null;
let requestId = 0;
let uiEpoch = 0;
let dismissed = false;
let suggestTimer = 0;
let booted = false;
let renderedFor = '';
let rerunAfterQuiet = false;
let fetchWrapped = false;

export function clampCount(value) {
    const number = Number(value);
    if (!Number.isFinite(number)) return defaultSettings.count;
    return Math.min(MAX_COUNT, Math.max(MIN_COUNT, Math.round(number)));
}

export function clampTokens(value) {
    const number = Number(value);
    if (!Number.isFinite(number)) return defaultSettings.maxTokens;
    return Math.min(MAX_TOKENS, Math.max(MIN_TOKENS, Math.round(number)));
}

export function stripReasoning(text) {
    return String(text ?? '')
        .replace(/<think\b[^>]*>[\s\S]*?<\/think>/gi, '')
        .replace(/<thinking\b[^>]*>[\s\S]*?<\/thinking>/gi, '')
        .replace(/<reasoning\b[^>]*>[\s\S]*?<\/reasoning>/gi, '')
        .trim();
}

function stripFences(text) {
    return text.replace(/^\s*```[a-z0-9_-]*\s*/i, '').replace(/\s*```\s*$/i, '').trim();
}

function cleanOption(text) {
    return String(text ?? '').replace(/^["“”']+|["“”']+$/g, '').trim();
}

const SCENE_NOTE = /^(?:\*\*)?(?:analyze|analysis|context|summary|state|notes?|mood|status|current state)\b/i;

function hasSpokenOrAction(text) {
    return /["“”*]/.test(text);
}

function looksLikeStateList(text) {
    const plain = String(text ?? '').trim();
    if (!plain || plain.length > 220 || hasSpokenOrAction(plain)) return false;
    if (!/,/.test(plain)) return false;
    return !/\b(i|i'm|i’m|im|we|you|he|she|they|is|was|were|am|are|don't|didn't|can't|won't|let's|lets)\b/i.test(plain);
}

/**
 * Scene notes and state summaries are not chat messages.
 * @param {string} text
 * @returns {boolean}
 */
export function isSceneNote(text) {
    const raw = String(text ?? '').trim();
    const plain = raw.replace(/\*\*/g, '').trim();
    if (!plain) return true;
    if (SCENE_NOTE.test(plain) || SCENE_NOTE.test(raw)) return true;
    if (/:\s*$/.test(plain) && plain.length < 80) return true;
    const labeled = plain.match(/^([^:]{1,48}):\s+(.+)$/);
    if (labeled && looksLikeStateList(labeled[2])) return true;
    const boldLabel = raw.match(/^\*\*([^*]{1,48})\*\*\s*:?\s+(.+)$/);
    if (boldLabel && (looksLikeStateList(boldLabel[2]) || SCENE_NOTE.test(boldLabel[1]))) return true;
    if (looksLikeStateList(plain) && plain.length < 120) return true;
    return false;
}

/**
 * Parse a numbered list of chat messages. Scene notes are dropped.
 * Unnumbered lines are kept only when they already look like messages.
 * @param {string} text Model output
 * @param {number} count Maximum options to keep
 * @returns {string[]}
 */
export function parseOptions(text, count) {
    const wanted = clampCount(count);
    const cleaned = stripFences(stripReasoning(text));
    if (!cleaned) return [];

    const lines = cleaned.split(/\r?\n/).map(line => line.trim()).filter(Boolean);
    const numbered = [];
    for (const line of lines) {
        const match = line.match(/^(?:\d{1,2}[.)\]:-])\s+(.+)$/);
        if (match) numbered.push(cleanOption(match[1]));
    }

    // A Claude prefill of "1." is not part of the completion, so the first line may be bare.
    if (numbered.length && lines[0] && !/^\d{1,2}[.)\]:-]/.test(lines[0])) {
        const first = cleanOption(lines[0]);
        if (first) numbered.unshift(first);
    }

    const source = numbered.length ? numbered : lines.map(cleanOption);
    const unique = [];
    for (const option of source) {
        if (!option || isSceneNote(option)) continue;
        if (unique.includes(option)) continue;
        unique.push(option);
        if (unique.length >= wanted) break;
    }
    return unique;
}

function textFromParts(value) {
    if (typeof value === 'string') return value;
    if (!Array.isArray(value)) return '';
    return value.map(part => {
        if (typeof part === 'string') return part;
        if (!part || typeof part !== 'object') return '';
        if (part.type === 'reasoning' || part.type === 'thinking' || part.type === 'tool_use') return '';
        return part.text || part.content || '';
    }).filter(Boolean).join('\n');
}

/**
 * Extract assistant text from a chat-completion response without assuming one provider.
 * @param {any} data Response JSON
 * @returns {string}
 */
export function extractCompletionText(data) {
    if (typeof data === 'string') return data;
    if (!data || typeof data !== 'object') return '';

    const claude = textFromParts(data.content);
    if (claude.trim()) return claude;

    const choice = Array.isArray(data.choices) ? data.choices[0] : null;
    const message = choice?.message;
    const fromChoice = textFromParts(message?.content)
        || textFromParts(choice?.text)
        || textFromParts(choice?.content);
    if (fromChoice.trim()) return fromChoice;

    // Reasoning is a scratchpad ("Analyze the current state"), not a reply.
    // Callers that want to salvage a numbered list from it can read it separately.
    return textFromParts(data.message?.content) || textFromParts(data.text) || '';
}

export function extractReasoningText(data) {
    if (!data || typeof data !== 'object') return '';
    const choice = Array.isArray(data.choices) ? data.choices[0] : null;
    const message = choice?.message;
    const reasoning = message?.reasoning_content || message?.reasoning || data.reasoning || '';
    return typeof reasoning === 'string' ? reasoning : '';
}

function contentKey(message) {
    const content = message?.content;
    return typeof content === 'string' ? content : JSON.stringify(content ?? null);
}

function sameOutgoingMessage(left, right) {
    if (!left || !right) return false;
    return left.role === right.role
        && contentKey(left) === contentKey(right)
        && (left.name ?? undefined) === (right.name ?? undefined)
        && (left.signature ?? undefined) === (right.signature ?? undefined)
        && (left.reasoning ?? undefined) === (right.reasoning ?? undefined)
        && (left.tool_call_id ?? undefined) === (right.tool_call_id ?? undefined)
        && JSON.stringify(left.tool_calls ?? null) === JSON.stringify(right.tool_calls ?? null);
}

/**
 * Index of the first message that differs from the snapshotted prefix, or -1.
 * Continue requests replace the trailing assistant message; that replacement is not a prefix break.
 * @param {object[]} original
 * @param {object[]} next
 * @param {{ ignoreLastOriginal?: boolean }} [options]
 * @returns {number}
 */
export function firstPrefixMismatch(original, next, { ignoreLastOriginal = false } = {}) {
    const comparable = ignoreLastOriginal ? Math.max(0, original.length - 1) : original.length;
    const limit = Math.min(comparable, next.length);
    for (let i = 0; i < limit; i++) {
        if (!sameOutgoingMessage(original[i], next[i])) return i;
    }
    if (next.length < comparable) return next.length;
    return -1;
}

/**
 * Clone the snapshotted messages and append the reply plus the instruction.
 * The original array is not mutated. Continue prompts that already end on the
 * assistant message get that content replaced instead of a second assistant turn,
 * so Claude cache-at-depth still sees exactly two new role switches on a normal turn.
 * @param {object[]} messages
 * @param {{ type?: string }} meta
 * @param {{ content: string, name?: string, signature?: string, reasoning?: string }} assistant
 * @param {string} instruction
 * @returns {{ messages: object[], replacedTail: boolean }}
 */
export function buildOptionsMessages(messages, meta, assistant, instruction) {
    const cloned = messages.map(message => ({ ...message }));
    const last = cloned[cloned.length - 1];
    // Continue requests already end on the partial assistant message. Replacing
    // that tail (not an older message) keeps the next send's history aligned.
    // Appending a second assistant message would merge or shift cache breakpoints.
    const replaceTail = meta?.type === 'continue' && last?.role === 'assistant';

    if (replaceTail) {
        last.content = assistant.content;
        if (assistant.signature) last.signature = assistant.signature;
        if (assistant.reasoning) last.reasoning = assistant.reasoning;
    } else {
        const tail = { role: 'assistant', content: assistant.content };
        if (assistant.name) tail.name = assistant.name;
        if (assistant.signature) tail.signature = assistant.signature;
        if (assistant.reasoning) tail.reasoning = assistant.reasoning;
        cloned.push(tail);
    }

    cloned.push({ role: 'user', content: instruction });
    return { messages: cloned, replacedTail: replaceTail };
}

function assistantFields(snapshotMessages, chatMessage) {
    const named = [...snapshotMessages].reverse().find(message => message?.role === 'assistant' && message.name);
    /** @type {{ content: string, name?: string, signature?: string, reasoning?: string }} */
    const fields = { content: String(chatMessage?.mes ?? '') };
    if (named) {
        fields.name = chatMessage?.name && chatMessage.name !== named.name ? chatMessage.name : named.name;
    }
    const extra = chatMessage?.extra ?? {};
    if (extra.reasoning_signature) fields.signature = extra.reasoning_signature;
    if (extra.reasoning) fields.reasoning = extra.reasoning;
    return fields;
}

function buildRequest(snap, assistant, instruction, maxTokens) {
    const body = cloneJson(snap.body);
    const built = buildOptionsMessages(body.messages, { type: snap.type }, assistant, instruction);
    body.messages = built.messages;
    // Drop the character prefill. A list prefill is added below for Claude only,
    // after the instruction, so it is not part of the stable prefix.
    delete body.assistant_prefill;
    delete body.logprobs;
    delete body.top_logprobs;
    // Stop strings are a generation parameter. Character-name stops would cut
    // the suggestion list and are not part of the cached prefix.
    delete body.stop;
    // A schema can push another user message on some server paths and shift
    // Claude's depth+2 cache breakpoint.
    delete body.json_schema;
    body.max_tokens = maxTokens;
    body.stream = false;
    if (typeof body.n === 'number') body.n = 1;
    // Generation parameters, not prompt bytes. High reasoning effort spends the
    // reply budget on state notes, and a very hot RP temperature ignores the list format.
    if (typeof body.reasoning_effort === 'string' && /^(medium|high|max|xhigh)$/i.test(body.reasoning_effort)) {
        body.reasoning_effort = 'low';
    }
    if (typeof body.temperature === 'number' && body.temperature > 1) {
        body.temperature = 1;
    }
    if (body.chat_completion_source === 'claude') {
        body.assistant_prefill = '1.';
    }
    return { body, replacedTail: built.replacedTail };
}

function cloneJson(value) {
    if (typeof structuredClone === 'function') {
        try {
            return structuredClone(value);
        } catch {
            // Fall through to JSON for objects that contain an unsupported value.
        }
    }
    return JSON.parse(JSON.stringify(value));
}

function context() {
    return globalThis.SillyTavern.getContext();
}

function currentChatKey() {
    const ctx = context();
    const id = ctx.getCurrentChatId?.() ?? ctx.chatId ?? '';
    return `${ctx.groupId ?? ''}:${ctx.characterId ?? ''}:${id}`;
}

function extensionName() {
    try {
        const parts = new URL(import.meta.url).pathname.split('/').filter(Boolean).map(decodeURIComponent);
        const extIndex = parts.lastIndexOf('extensions');
        if (extIndex >= 0 && extIndex < parts.length - 2) {
            return parts.slice(extIndex + 1, -1).join('/');
        }
    } catch (error) {
        console.warn(LOG, 'could not resolve extension folder', error);
    }
    return 'third-party/reply-options';
}

function getSettings() {
    const ctx = context();
    if (!ctx.extensionSettings[MODULE] || typeof ctx.extensionSettings[MODULE] !== 'object') {
        ctx.extensionSettings[MODULE] = structuredClone(defaultSettings);
    }
    const settings = ctx.extensionSettings[MODULE];
    for (const [key, value] of Object.entries(defaultSettings)) {
        if (!Object.hasOwn(settings, key)) settings[key] = value;
    }
    settings.count = clampCount(settings.count);
    settings.maxTokens = clampTokens(settings.maxTokens);
    if (settings.instructionVersion !== INSTRUCTION_VERSION) {
        const current = typeof settings.instruction === 'string' ? settings.instruction.trim() : '';
        if (!current || current === LEGACY_INSTRUCTION) settings.instruction = DEFAULT_INSTRUCTION;
        settings.instructionVersion = INSTRUCTION_VERSION;
        ctx.saveSettingsDebounced();
    }
    if (typeof settings.instruction !== 'string' || !settings.instruction.trim()) {
        settings.instruction = DEFAULT_INSTRUCTION;
    }
    return settings;
}

function saveSettings() {
    context().saveSettingsDebounced();
}

function fillInstruction(template, count) {
    const source = String(template || DEFAULT_INSTRUCTION);
    let text = source;
    try {
        text = context().substituteParamsExtended(source, { count: String(count) });
    } catch (error) {
        console.warn(LOG, 'macro substitution failed', error);
    }
    return String(text).replaceAll('{{count}}', String(count));
}

function isPlainAssistantMessage(message) {
    if (!message || message.is_user || message.is_system) return false;
    if (Array.isArray(message.extra?.tool_invocations) && message.extra.tool_invocations.length > 0) return false;
    return String(message.mes ?? '').trim().length > 0;
}

function captureSnapshot(generateData) {
    const ctx = context();
    if (ctx.mainApi !== 'openai') return;
    if (!generateData || typeof generateData !== 'object') return;
    const type = generateData.type;
    if (!type || SKIP_SNAPSHOT_TYPES.has(type)) return;
    if (!Array.isArray(generateData.messages) || generateData.messages.length === 0) return;

    snapshot = {
        chatKey: currentChatKey(),
        type,
        body: cloneJson(generateData),
    };
    console.debug(LOG, 'snapshotted request', {
        type,
        messages: generateData.messages.length,
        tools: Array.isArray(generateData.tools) ? generateData.tools.length : 0,
    });
}

function logReplay(snap, built) {
    const mismatch = firstPrefixMismatch(snap.body.messages, built.body.messages, {
        ignoreLastOriginal: built.replacedTail,
    });
    if (mismatch >= 0) {
        console.warn(LOG, 'prefix diverged before the tail; older-message cache may miss', {
            index: mismatch,
            role: built.body.messages[mismatch]?.role,
        });
        return;
    }
    console.debug(LOG, 'suffix replay', {
        type: snap.type,
        prefixMessages: built.replacedTail ? snap.body.messages.length - 1 : snap.body.messages.length,
        totalMessages: built.body.messages.length,
    });
}

function errorText(data, fallback) {
    if (typeof data === 'string' && data.trim()) return data.trim();
    const error = data?.error ?? data?.detail?.error;
    if (typeof error === 'string' && error.trim()) return error.trim();
    const message = error?.message || error?.code || data?.message;
    if (typeof message === 'string' && message.trim()) return message.trim();
    return fallback;
}

function shortError(error) {
    const message = error?.message || String(error || 'Suggestion request failed');
    const oneLine = message.replace(/\s+/g, ' ').trim();
    return oneLine.length > 280 ? `${oneLine.slice(0, 280)}…` : oneLine;
}

function isAbortError(error) {
    return error?.name === 'AbortError' || error?.code === 20;
}

function requestUrl(input) {
    if (typeof input === 'string') return input;
    if (typeof URL !== 'undefined' && input instanceof URL) return input.href;
    if (input && typeof input.url === 'string') return input.url;
    return '';
}

function headerValue(headers, name) {
    if (!headers) return '';
    if (typeof Headers !== 'undefined' && headers instanceof Headers) {
        return headers.get(name) || '';
    }
    if (typeof headers !== 'object') return '';
    const key = Object.keys(headers).find(item => item.toLowerCase() === name.toLowerCase());
    return key ? String(headers[key] ?? '') : '';
}

function installFetchSnapshot() {
    if (fetchWrapped || typeof globalThis.fetch !== 'function') return;
    fetchWrapped = true;
    const nativeFetch = globalThis.fetch.bind(globalThis);
    globalThis.fetch = function(input, init) {
        try {
            maybeSnapshotFetch(input, init);
        } catch (error) {
            console.error(LOG, 'failed to snapshot request', error);
        }
        return nativeFetch(input, init);
    };
}

function maybeSnapshotFetch(input, init) {
    const url = requestUrl(input);
    if (!url.includes(ENDPOINT)) return;
    const headers = init?.headers ?? (typeof Request !== 'undefined' && input instanceof Request ? input.headers : undefined);
    // Our own suggestion call uses this header. Snapshotting it would append another tail next time.
    if (headerValue(headers, 'X-Reply-Options') === '1') return;
    const raw = init?.body;
    if (typeof raw !== 'string' || !raw) return;
    captureSnapshot(JSON.parse(raw));
}

async function postCompletion(body, signal) {
    const response = await fetch(ENDPOINT, {
        method: 'POST',
        headers: {
            ...context().getRequestHeaders(),
            'X-Reply-Options': '1',
        },
        cache: 'no-store',
        body: JSON.stringify(body),
        signal,
    });
    const raw = await response.text();
    let data = raw;
    try {
        data = raw ? JSON.parse(raw) : null;
    } catch {
        // Keep the raw body for the error message.
    }
    if (!response.ok || data?.error) {
        throw new Error(errorText(data, `Suggestion request failed (${response.status})`));
    }
    return data;
}

function panelElement() {
    return document.getElementById('reply-options-panel');
}

function ensurePanel() {
    const existing = panelElement();
    if (existing) return existing;
    const sendForm = document.getElementById('send_form');
    if (!sendForm) return null;

    const panel = document.createElement('div');
    panel.id = 'reply-options-panel';
    panel.className = 'reply-options-panel';
    panel.hidden = true;

    const header = document.createElement('div');
    header.className = 'reply-options-header';

    const status = document.createElement('div');
    status.className = 'reply-options-status';
    status.setAttribute('aria-live', 'polite');

    const actions = document.createElement('div');
    actions.className = 'reply-options-actions';

    const retry = document.createElement('button');
    retry.type = 'button';
    retry.className = 'reply-options-icon reply-options-retry';
    retry.title = 'Regenerate suggestions';
    retry.setAttribute('aria-label', 'Regenerate suggestions');
    retry.innerHTML = '<i class="fa-solid fa-rotate-right" aria-hidden="true"></i>';
    retry.addEventListener('click', () => {
        dismissed = false;
        renderedFor = '';
        if (isGenerationActive()) {
            maybeSchedule(100);
            return;
        }
        void generateSuggestions({ force: true });
    });

    const dismiss = document.createElement('button');
    dismiss.type = 'button';
    dismiss.className = 'reply-options-icon';
    dismiss.title = 'Hide suggestions';
    dismiss.setAttribute('aria-label', 'Hide suggestions');
    dismiss.innerHTML = '<i class="fa-solid fa-xmark" aria-hidden="true"></i>';
    dismiss.addEventListener('click', () => {
        dismissed = true;
        cancelScheduled();
        abortRequest();
        invalidateUi();
    });

    const list = document.createElement('div');
    list.className = 'reply-options-list';

    actions.append(retry, dismiss);
    header.append(status, actions);
    panel.append(header, list);
    sendForm.before(panel);
    return panel;
}

function setRetryEnabled(enabled) {
    const button = panelElement()?.querySelector('.reply-options-retry');
    if (button) button.disabled = !enabled;
}

function hidePanel() {
    const panel = panelElement();
    if (!panel) return;
    panel.hidden = true;
    panel.classList.remove('is-loading');
    panel.querySelector('.reply-options-list')?.replaceChildren();
    const status = panel.querySelector('.reply-options-status');
    if (status) status.textContent = '';
    setRetryEnabled(true);
}

function invalidateUi() {
    uiEpoch += 1;
    hidePanel();
}

function showStatus(message, allowRetry) {
    const panel = ensurePanel();
    if (!panel) return;
    panel.hidden = false;
    panel.classList.remove('is-loading');
    const status = panel.querySelector('.reply-options-status');
    if (status) status.textContent = message;
    panel.querySelector('.reply-options-list')?.replaceChildren();
    setRetryEnabled(allowRetry);
}

function showLoading() {
    const panel = ensurePanel();
    if (!panel) return;
    panel.hidden = false;
    panel.classList.add('is-loading');
    const status = panel.querySelector('.reply-options-status');
    if (status) status.textContent = 'Generating suggestions…';
    panel.querySelector('.reply-options-list')?.replaceChildren();
    setRetryEnabled(false);
}

function renderOptions(options) {
    const panel = ensurePanel();
    if (!panel) return;
    const list = panel.querySelector('.reply-options-list');
    if (!list) return;
    panel.hidden = false;
    panel.classList.remove('is-loading');
    const status = panel.querySelector('.reply-options-status');
    if (status) status.textContent = 'Suggestions';
    list.replaceChildren();
    for (const option of options) {
        const button = document.createElement('button');
        button.type = 'button';
        button.className = 'reply-options-item';
        button.textContent = option;
        button.title = 'Put this in the message box';
        button.addEventListener('click', () => applyOption(option));
        list.append(button);
    }
    setRetryEnabled(true);
}

/**
 * The only textarea write. Clicking a suggestion replaces the draft and does not send.
 * @param {string} text
 */
function applyOption(text) {
    const textarea = document.getElementById('send_textarea');
    if (!textarea) return;
    const $textarea = globalThis.$?.('#send_textarea');
    if ($textarea?.length) {
        $textarea.val(text);
    } else {
        textarea.value = text;
    }
    textarea.dispatchEvent(new Event('input', { bubbles: true }));
    textarea.focus();
}

function cancelScheduled() {
    if (suggestTimer) {
        clearTimeout(suggestTimer);
        suggestTimer = 0;
    }
}

function abortRequest() {
    if (!abortController) return;
    abortController.abort();
    abortController = null;
}

function isGenerationActive() {
    if (document.body?.dataset?.generating === 'true') return true;
    // showStopButton() sets this inline. Do not use computed style: the
    // stylesheet can hide the button without meaning a generation is running.
    return document.getElementById('mes_stop')?.style.display === 'flex';
}

function suggestionKey() {
    const ctx = context();
    const last = ctx.chat?.[ctx.chat.length - 1];
    return [
        snapshot?.chatKey ?? '',
        ctx.chat?.length ?? 0,
        last?.swipe_id ?? 0,
        last?.mes ?? '',
    ].join('\n');
}

function maybeSchedule(delay) {
    if (dismissed || !getSettings().enabled) return;
    if (context().mainApi !== 'openai') return;
    if (!snapshot || snapshot.chatKey !== currentChatKey()) return;
    const last = context().chat?.[context().chat.length - 1];
    if (!isPlainAssistantMessage(last)) return;
    scheduleSuggestions(delay);
}

function scheduleSuggestions(delay) {
    if (!getSettings().enabled || dismissed) return;
    cancelScheduled();
    suggestTimer = setTimeout(() => {
        suggestTimer = 0;
        if (dismissed || !getSettings().enabled) return;
        // Tool-call follow-ups and auto-continue keep the stop button up.
        // Sending now would evict the slot underneath that request.
        if (isGenerationActive()) {
            scheduleSuggestions(100);
            return;
        }
        void generateSuggestions();
    }, delay);
}

async function generateSuggestions({ force = false } = {}) {
    if (dismissed) return;
    const settings = getSettings();
    if (!settings.enabled) {
        invalidateUi();
        return;
    }

    const ctx = context();
    const snap = snapshot;
    if (ctx.mainApi !== 'openai' || !snap || snap.chatKey !== currentChatKey()) return;
    const last = ctx.chat?.[ctx.chat.length - 1];
    if (!isPlainAssistantMessage(last)) return;
    if (!Array.isArray(snap.body?.messages) || snap.body.messages.length === 0) return;
    const key = suggestionKey();
    if (!force && key === renderedFor) return;
    if (isGenerationActive()) {
        maybeSchedule(100);
        return;
    }

    const epoch = uiEpoch;
    const myId = ++requestId;
    abortRequest();
    showLoading();

    const controller = new AbortController();
    abortController = controller;
    try {
        const instruction = fillInstruction(settings.instruction, settings.count);
        const assistant = assistantFields(snap.body.messages, last);
        const built = buildRequest(snap, assistant, instruction, settings.maxTokens);
        logReplay(snap, built);
        const data = await postCompletion(built.body, controller.signal);
        if (myId !== requestId || epoch !== uiEpoch || dismissed) return;
        const content = extractCompletionText(data);
        let options = parseOptions(content, settings.count);
        if (options.length < 2) {
            const fromReasoning = parseOptions(extractReasoningText(data), settings.count);
            if (fromReasoning.length > options.length) options = fromReasoning;
        }
        if (!options.length) {
            showStatus('No chat messages were returned. Try again.', true);
            return;
        }
        renderedFor = key;
        renderOptions(options);
    } catch (error) {
        if (myId !== requestId || epoch !== uiEpoch || dismissed) return;
        if (isAbortError(error)) {
            if (!rerunAfterQuiet) showStatus('Suggestion request was interrupted.', true);
            return;
        }
        console.error(LOG, error);
        showStatus(shortError(error), true);
    } finally {
        if (abortController === controller) abortController = null;
    }
}

function onCharacterRendered(messageId, type) {
    if (type === 'quiet' || type === 'impersonate' || type === 'first_message') return;
    const ctx = context();
    if (Number(messageId) !== ctx.chat.length - 1) return;
    dismissed = false;
    // Streaming unlocks the UI before this event. Non-streaming and tool
    // follow-ups are still generating; wait for GENERATION_ENDED in that case.
    if (!isGenerationActive()) maybeSchedule(50);
}

function onGenerationEnded() {
    // GENERATION_ENDED is emitted from hideStopButton() before
    // dataset.generating is cleared. Do not bail on the active flag here;
    // the timer runs after that cleanup and waits if a follow-up is still running.
    if (rerunAfterQuiet) {
        rerunAfterQuiet = false;
        renderedFor = '';
        maybeSchedule(300);
        return;
    }
    maybeSchedule(50);
}

function onSwiped(messageId) {
    const ctx = context();
    const id = Number(messageId);
    if (Number.isFinite(id) && id !== ctx.chat.length - 1) {
        snapshot = null;
        renderedFor = '';
        invalidateUi();
        return;
    }
    dismissed = false;
    renderedFor = '';
    // A new swipe generation emits this before Generate(). GENERATION_STARTED
    // cancels the timer. Cycling an existing swipe does not, so the debounce
    // waits until rapid swipes settle.
    maybeSchedule(400);
}

function onGenerationStarted(type, _params, dryRun) {
    if (dryRun === true) return;
    cancelScheduled();
    abortRequest();
    if (type === 'quiet') {
        rerunAfterQuiet = true;
        return;
    }
    rerunAfterQuiet = false;
    renderedFor = '';
    dismissed = false;
    invalidateUi();
}

function onUserSent() {
    renderedFor = '';
    rerunAfterQuiet = false;
    cancelScheduled();
    abortRequest();
    dismissed = false;
    invalidateUi();
}

function onChatChanged() {
    snapshot = null;
    renderedFor = '';
    rerunAfterQuiet = false;
    cancelScheduled();
    abortRequest();
    dismissed = false;
    invalidateUi();
}

function onHistoryMutated() {
    snapshot = null;
    renderedFor = '';
    rerunAfterQuiet = false;
    cancelScheduled();
    abortRequest();
    dismissed = false;
    invalidateUi();
}

function bindSettings(root) {
    const settings = getSettings();
    const enabled = root.querySelector('#reply_options_enabled');
    const count = root.querySelector('#reply_options_count');
    const countValue = root.querySelector('#reply_options_count_value');
    const maxTokens = root.querySelector('#reply_options_max_tokens');
    const instruction = root.querySelector('#reply_options_instruction');

    enabled.checked = settings.enabled;
    count.value = String(settings.count);
    countValue.textContent = String(settings.count);
    maxTokens.value = String(settings.maxTokens);
    instruction.value = settings.instruction;

    enabled.addEventListener('change', () => {
        getSettings().enabled = enabled.checked;
        saveSettings();
        if (!enabled.checked) {
            cancelScheduled();
            abortRequest();
            invalidateUi();
            return;
        }
        dismissed = false;
        renderedFor = '';
        if (isGenerationActive()) {
            maybeSchedule(100);
            return;
        }
        void generateSuggestions({ force: true });
    });

    count.addEventListener('input', () => {
        const value = clampCount(count.value);
        countValue.textContent = String(value);
        getSettings().count = value;
        saveSettings();
    });

    maxTokens.addEventListener('change', () => {
        const value = clampTokens(maxTokens.value);
        maxTokens.value = String(value);
        getSettings().maxTokens = value;
        saveSettings();
    });

    instruction.addEventListener('change', () => {
        const value = instruction.value.trim();
        getSettings().instruction = value || DEFAULT_INSTRUCTION;
        if (!value) instruction.value = DEFAULT_INSTRUCTION;
        saveSettings();
    });
}

async function mountSettings() {
    if (document.getElementById('reply_options_settings')) return;
    const host = document.getElementById('extensions_settings2') || document.getElementById('extensions_settings');
    if (!host) return;
    const ctx = context();
    const html = await ctx.renderExtensionTemplateAsync(extensionName(), 'settings');
    host.insertAdjacentHTML('beforeend', html);
    const root = document.getElementById('reply_options_settings');
    if (root) bindSettings(root);
}

function bindEvents() {
    const { eventSource, eventTypes } = context();
    eventSource.on(eventTypes.CHARACTER_MESSAGE_RENDERED, onCharacterRendered);
    eventSource.on(eventTypes.MESSAGE_SWIPED, onSwiped);
    eventSource.on(eventTypes.GENERATION_STARTED, onGenerationStarted);
    eventSource.on(eventTypes.GENERATION_ENDED, onGenerationEnded);
    eventSource.on(eventTypes.MESSAGE_SENT, onUserSent);
    eventSource.on(eventTypes.CHAT_CHANGED, onChatChanged);
    eventSource.on(eventTypes.MESSAGE_EDITED, onHistoryMutated);
    eventSource.on(eventTypes.MESSAGE_DELETED, onHistoryMutated);
    eventSource.on(eventTypes.MESSAGE_SWIPE_DELETED, onHistoryMutated);
    eventSource.on(eventTypes.MAIN_API_CHANGED, onChatChanged);
    eventSource.on(eventTypes.APP_READY, () => {
        ensurePanel();
        void mountSettings().catch(error => console.error(LOG, 'settings mount failed', error));
    });
}

async function init() {
    if (booted) return;
    booted = true;
    installFetchSnapshot();
    getSettings();
    ensurePanel();
    try {
        await mountSettings();
    } catch (error) {
        console.error(LOG, 'settings mount failed', error);
    }
    bindEvents();
    console.info(LOG, 'ready', extensionName());
}

function boot() {
    if (typeof globalThis.SillyTavern === 'undefined') return;
    const start = () => {
        void init().catch(error => console.error(LOG, 'init failed', error));
    };
    if (typeof globalThis.jQuery === 'function') {
        globalThis.jQuery(start);
    } else {
        start();
    }
}

installFetchSnapshot();
boot();
