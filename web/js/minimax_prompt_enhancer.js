/** LLM prompt enhancer panel for MiniMax H3 Director (Ollama / Zhipu). */

import { api } from "../../scripts/api.js";
import { resolveTaskKey, taskUsesReferenceImages, taskUsesReferenceVideo } from "./minimax_gen_timeline.js";
import { stripFl2vPromptBody } from "./minimax_fl2v.js";
import { t } from "./minimax_i18n.js";

export const PE_PANEL_COLLAPSED_H = 34;
export const PE_PANEL_EXPANDED_H = 348;

const DEFAULT_LLM_URL = "http://127.0.0.1:11434/v1";
const DEFAULT_LLM_MODEL = "qwen3.5";
const DEFAULT_ZHIPU_URL = "https://open.bigmodel.cn/api/paas/v4";
const DEFAULT_ZHIPU_MODEL = "glm-4.6v-flash";
const DEFAULT_OPENAI_COMPAT_URL = "http://127.0.0.1:8080/v1";
const DEFAULT_API_FORMAT = "Ollama";
const API_OLLAMA = "Ollama";
const API_ZHIPU = "智谱 GLM";
const API_OPENAI_COMPAT = "OpenAI Compatible";
const OPENAI_COMPAT_STANDARD = "Standard";
const OPENAI_COMPAT_LLAMA_SWAP = "llama-swap";
const DEFAULT_OUTPUT_LANGUAGE = "Chinese";
const OUTPUT_LANGUAGE_ZH = "Chinese";
const CHARACTER_DETAIL_NORMAL = "一般";
const CHARACTER_DETAIL_DETAILED = "详尽";
const LEGACY_OPENAI_FORMAT = "OpenAI / vLLM";

const STATUS_COLORS = {
    info: "#9aa3b5",
    loading: "#fbbf24",
    success: "#4ade80",
    error: "#f87171",
};

function coerceLlmUrl(value, defaultUrl = DEFAULT_LLM_URL) {
    const s = String(value ?? "").trim();
    if (/^https?:\/\//i.test(s)) return s.replace(/\/+$/, "");
    return defaultUrl;
}

function coerceLlmModel(value) {
    const s = String(value ?? "").trim();
    if (!s || s === "true" || s === "false") return DEFAULT_LLM_MODEL;
    return s;
}

function normalizeApiFormat(fmt) {
    if (fmt === LEGACY_OPENAI_FORMAT) return API_OPENAI_COMPAT;
    if (fmt === API_ZHIPU || fmt === API_OLLAMA || fmt === API_OPENAI_COMPAT) return fmt;
    return DEFAULT_API_FORMAT;
}

function inferApiFormat(url, explicit) {
    const fmt = normalizeApiFormat(explicit);
    if (fmt === API_ZHIPU || fmt === API_OLLAMA || fmt === API_OPENAI_COMPAT) return fmt;
    const u = coerceLlmUrl(url);
    if (/bigmodel\.cn/i.test(u)) return API_ZHIPU;
    return DEFAULT_API_FORMAT;
}

function defaultsForApiFormat(fmt) {
    if (fmt === API_ZHIPU) return { url: DEFAULT_ZHIPU_URL, model: DEFAULT_ZHIPU_MODEL };
    if (fmt === API_OPENAI_COMPAT) return { url: DEFAULT_OPENAI_COMPAT_URL, model: DEFAULT_LLM_MODEL };
    return { url: "http://127.0.0.1:11434", model: DEFAULT_LLM_MODEL };
}

function normalizeOpenAiCompatMode(mode) {
    return String(mode || "").trim().toLowerCase() === OPENAI_COMPAT_LLAMA_SWAP
        ? OPENAI_COMPAT_LLAMA_SWAP
        : OPENAI_COMPAT_STANDARD;
}

function ensurePeStyles() {
    if (document.getElementById("minimax-pe-styles")) return;
    const style = document.createElement("style");
    style.id = "minimax-pe-styles";
    style.textContent = `
@keyframes minimax-pe-pulse { 0%,100%{opacity:1} 50%{opacity:.65} }
.minimax-pe-loading { animation: minimax-pe-pulse 1.2s ease-in-out infinite !important; }
.minimax-pe-label { font-size: 11px; color: #b8c0d0; flex-shrink: 0; white-space: nowrap; }
.minimax-pe-input, .minimax-pe-select {
    font-size: 11px; line-height: 1.35; min-height: 28px; box-sizing: border-box;
    background: #12151b; color: #e8ecf4; border: 1px solid #2a3140; border-radius: 4px;
}
.minimax-pe-input { padding: 5px 8px; }
.minimax-pe-select { padding: 4px 8px; cursor: pointer; }
.minimax-pe-btn-sm {
    font-size: 11px; line-height: 1.35; min-height: 28px; box-sizing: border-box;
    background: #252a34; color: #e8ecf4; border: 1px solid #2a3140; border-radius: 4px;
    padding: 4px 10px; cursor: pointer; flex-shrink: 0; white-space: nowrap;
}
.minimax-pe-api-row { display: flex; gap: 6px; align-items: center; flex-wrap: nowrap; }
.minimax-pe-api-row .minimax-pe-select { flex: 0 1 38%; min-width: 132px; max-width: 220px; }
.minimax-pe-api-row .minimax-pe-input { flex: 1 1 120px; min-width: 0; }
.minimax-pe-options-row { display: flex; gap: 14px; align-items: center; flex-wrap: wrap; }
.minimax-pe-check-item { display: flex; gap: 4px; align-items: center; }
.minimax-pe-check-item span { font-size: 11px; color: #b8c0d0; }
`;
    document.head.appendChild(style);
}

function el(style, text, tag = "div") {
    const node = document.createElement(tag);
    if (style) Object.assign(node.style, style);
    if (text != null) node.textContent = text;
    return node;
}

function swallowKeys(input) {
    input.addEventListener("keydown", (e) => e.stopPropagation());
    input.addEventListener("keyup", (e) => e.stopPropagation());
}

async function fetchImageB64(imageFile) {
    const resp = await api.fetchApi("/minimax/director/image_b64", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ imageFile }),
    });
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.error || resp.statusText);
    return data.image;
}

function resolveOutputLanguage(pe) {
    const widgetLang = pe.widget?.("llm_output_language")?.value;
    if (widgetLang) {
        const normalized = /^(中文|chinese|zh|cn)/i.test(String(widgetLang).trim())
            ? OUTPUT_LANGUAGE_ZH
            : "English";
        if (pe.langSelect) pe.langSelect.value = normalized;
        return normalized;
    }
    return pe.langSelect?.value || DEFAULT_OUTPUT_LANGUAGE;
}

function coerceFeatureEnhanceValue(value) {
    if (value === true || value === 1) return true;
    if (value === false || value === 0 || value == null || value === "") return false;
    const v = String(value).trim().toLowerCase();
    if (v === "true" || v === "yes" || v === "on") return true;
    // 旧下拉迁移到 BOOLEAN 时可能残留「一般」「详尽」字符串，仅「详尽」视为开启
    if (v.includes("详尽") || v === "detailed" || v === "verbose" || v === "full") return true;
    return false;
}

function resolveCharacterFeatureEnhance(pe, { preferWidget = false } = {}) {
    const enhanceWidget = pe.widget?.("llm_character_feature_enhance");
    const fromWidgetEnhance = enhanceWidget?.value;
    const fromCheck = pe.detailCheck?.checked;
    if (preferWidget && enhanceWidget) {
        return coerceFeatureEnhanceValue(fromWidgetEnhance);
    }
    if (fromCheck != null) return !!fromCheck;
    if (enhanceWidget) return coerceFeatureEnhanceValue(fromWidgetEnhance);
    return false;
}

function formatEnhanceSuccessStatus(taskKey, result) {
    let msg = t("promptEnhancer.success", { task: taskKey, count: result.text?.length ?? 0 });
    if (result.detailedMode) {
        const han = result.hanCount ?? 0;
        const target = result.detailTargetHan ?? 300;
        msg += t("promptEnhancer.hanCount", { count: han, target });
        if (han < target) {
            msg += t("promptEnhancer.stillShort");
        }
    }
    return msg;
}

function resolveCharacterDetailLevel(pe, opts) {
    return resolveCharacterFeatureEnhance(pe, opts) ? CHARACTER_DETAIL_DETAILED : "一般";
}

export function mountPromptEnhancerPanel(editor, parentEl) {
    ensurePeStyles();
    const pe = { editor, open: false, _currentDefaultTemplate: "", _busy: false };

    pe.setStatus = (text, kind = "info") => {
        pe.statusEl.textContent = text || "";
        pe.statusEl.style.color = STATUS_COLORS[kind] || STATUS_COLORS.info;
        pe.statusEl.style.fontWeight = kind === "error" ? "600" : "400";
        pe.statusEl.style.whiteSpace = "pre-wrap";
        pe.statusEl.style.lineHeight = "1.35";
    };

    pe.setEnhanceLoading = (loading, activeBtn = null, label = t("promptEnhancer.enhancing")) => {
        pe._busy = loading;
        pe.enhanceCurrentBtn.disabled = loading;
        pe.enhanceAllBtn.disabled = loading;
        pe.refreshBtn.disabled = loading;
        pe.unloadBtn.disabled = loading;
        if (loading && activeBtn) {
            activeBtn.textContent = label;
            activeBtn.style.background = "#d97706";
            activeBtn.style.cursor = "wait";
            activeBtn.classList.add("minimax-pe-loading");
        } else {
            pe.enhanceCurrentBtn.textContent = t("promptEnhancer.enhanceCurrent");
            pe.enhanceAllBtn.textContent = t("promptEnhancer.enhanceAll");
            pe.enhanceCurrentBtn.style.background = "#3b82f6";
            pe.enhanceAllBtn.style.background = "#6366f1";
            pe.enhanceCurrentBtn.style.cursor = "pointer";
            pe.enhanceAllBtn.style.cursor = "pointer";
            pe.enhanceCurrentBtn.classList.remove("minimax-pe-loading");
            pe.enhanceAllBtn.classList.remove("minimax-pe-loading");
        }
    };

    const header = el({
        display: "flex", justifyContent: "space-between", alignItems: "center",
        background: "#1a1d24", border: "1px solid #2a3140", borderRadius: "4px",
        padding: "6px 8px", cursor: "pointer", userSelect: "none", marginTop: "6px",
    });
    pe.titleEl = el({ fontWeight: "600", fontSize: "10px", color: "#9aa3b5", textTransform: "uppercase" }, t("promptEnhancer.title"));
    header.appendChild(pe.titleEl);
    pe.arrow = el({ fontSize: "10px", color: "#9aa3b5" }, "\u25B6");
    header.appendChild(pe.arrow);

    pe.body = el({
        background: "#1a1d24", border: "1px solid #2a3140", borderTop: "none",
        borderRadius: "0 0 4px 4px", padding: "8px", display: "none",
        flexDirection: "column", gap: "6px", marginTop: "-5px",
    });
    pe.descriptionEl = el({ fontSize: "9px", color: "#7d8698", lineHeight: "1.4" }, t("promptEnhancer.description"));
    pe.body.appendChild(pe.descriptionEl);

    const fmtRow = el({});
    fmtRow.className = "minimax-pe-api-row";
    fmtRow.appendChild(el({}, "API:", "span")).className = "minimax-pe-label";
    pe.apiSelect = document.createElement("select");
    pe.apiSelect.className = "minimax-pe-select";
    for (const [val, label] of [
        [API_OLLAMA, "Ollama (/api/chat)"],
        [API_ZHIPU, t("promptEnhancer.zhipu")],
        [API_OPENAI_COMPAT, "OpenAI Compatible (/v1/chat/completions)"],
    ]) {
        const o = document.createElement("option");
        o.value = val; o.textContent = label;
        pe.apiSelect.appendChild(o);
    }
    pe.urlInput = document.createElement("input");
    pe.urlInput.type = "text";
    pe.urlInput.placeholder = DEFAULT_LLM_URL;
    pe.urlInput.className = "minimax-pe-input";
    pe.urlInput.oninput = () => {
        pe.apiSelect.value = inferApiFormat(pe.urlInput.value, pe.apiSelect.value);
        pe.updateApiFormatUI();
        pe.syncToWidgets();
    };
    swallowKeys(pe.urlInput);
    pe.apiSelect.onchange = () => {
        const d = defaultsForApiFormat(pe.apiSelect.value);
        pe.urlInput.value = d.url;
        if (!pe.modelInput.value.trim() || pe._lastApiFormat !== pe.apiSelect.value) {
            pe.modelInput.value = d.model;
        }
        pe._lastApiFormat = pe.apiSelect.value;
        pe.updateApiFormatUI();
        pe.syncToWidgets();
        pe.fetchModels();
    };
    fmtRow.appendChild(pe.apiSelect);
    fmtRow.appendChild(pe.urlInput);
    pe.refreshBtn = el({}, t("promptEnhancer.refreshModels"), "button");
    pe.refreshBtn.className = "minimax-pe-btn-sm";
    pe.refreshBtn.onclick = () => pe.fetchModels();
    fmtRow.appendChild(pe.refreshBtn);
    pe.visionBadge = el({ fontSize: "9px", color: "#4ade80", flexShrink: "0", display: "none" });
    fmtRow.appendChild(pe.visionBadge);
    pe.body.appendChild(fmtRow);

    const compatRow = el({ display: "none", gap: "6px", alignItems: "center" });
    pe.compatLabel = el({}, t("promptEnhancer.openAiFeatures"), "span");
    pe.compatLabel.className = "minimax-pe-label";
    compatRow.appendChild(pe.compatLabel);
    pe.compatSelect = document.createElement("select");
    pe.compatSelect.className = "minimax-pe-select";
    Object.assign(pe.compatSelect.style, { flex: "1" });
    for (const [val, label] of [
        [OPENAI_COMPAT_STANDARD, t("promptEnhancer.standard")],
        [OPENAI_COMPAT_LLAMA_SWAP, "llama-swap"],
    ]) {
        const o = document.createElement("option");
        o.value = val; o.textContent = label;
        pe.compatSelect.appendChild(o);
    }
    pe.compatSelect.title = t("promptEnhancer.compatTooltip");
    pe.compatSelect.onchange = () => {
        pe.updateApiFormatUI();
        pe.syncToWidgets();
    };
    compatRow.appendChild(pe.compatSelect);
    pe.compatRow = compatRow;
    pe.body.appendChild(compatRow);

    const keyRow = el({ display: "flex", gap: "6px", alignItems: "center" });
    keyRow.dataset.r = "pe-key-row";
    keyRow.appendChild(el({}, "API Key:", "span")).className = "minimax-pe-label";
    pe.apiKeyInput = document.createElement("input");
    pe.apiKeyInput.type = "password";
    pe.apiKeyInput.placeholder = t("promptEnhancer.zhipuKey");
    pe.apiKeyInput.autocomplete = "off";
    pe.apiKeyInput.className = "minimax-pe-input";
    Object.assign(pe.apiKeyInput.style, { flex: "1" });
    pe.apiKeyInput.oninput = () => pe.syncToWidgets();
    swallowKeys(pe.apiKeyInput);
    keyRow.appendChild(pe.apiKeyInput);
    pe.keyRow = keyRow;
    pe.body.appendChild(keyRow);

    const modelRow = el({ display: "flex", gap: "6px", alignItems: "center" });
    pe.modelLabel = el({}, t("promptEnhancer.model"), "span");
    pe.modelLabel.className = "minimax-pe-label";
    modelRow.appendChild(pe.modelLabel);
    pe.modelInput = document.createElement("input");
    pe.modelInput.type = "text";
    pe.modelInput.placeholder = DEFAULT_LLM_MODEL;
    pe.modelInput.className = "minimax-pe-input";
    Object.assign(pe.modelInput.style, { flex: "1" });
    pe.modelInput.oninput = () => pe.syncToWidgets();
    swallowKeys(pe.modelInput);
    modelRow.appendChild(pe.modelInput);
    pe.modelList = document.createElement("datalist");
    pe.modelList.id = `minimax-pe-models-${editor.node.id}`;
    pe.modelInput.setAttribute("list", pe.modelList.id);
    modelRow.appendChild(pe.modelList);
    pe.body.appendChild(modelRow);

    const langRow = el({ display: "flex", gap: "6px", alignItems: "center" });
    pe.languageLabel = el({}, t("promptEnhancer.outputLanguage"), "span");
    pe.languageLabel.className = "minimax-pe-label";
    langRow.appendChild(pe.languageLabel);
    pe.langSelect = document.createElement("select");
    pe.langSelect.className = "minimax-pe-select";
    Object.assign(pe.langSelect.style, { flex: "1" });
    for (const [val, label] of [
        [OUTPUT_LANGUAGE_ZH, t("promptEnhancer.chinese")],
        ["English", t("promptEnhancer.english")],
    ]) {
        const o = document.createElement("option");
        o.value = val; o.textContent = label;
        pe.langSelect.appendChild(o);
    }
    pe.langSelect.title = t("promptEnhancer.languageTooltip");
    pe.langSelect.onchange = () => {
        pe._lastOutputLanguage = pe.langSelect.value || DEFAULT_OUTPUT_LANGUAGE;
        pe.syncToWidgets();
        pe.fetchTemplate(true);
    };
    langRow.appendChild(pe.langSelect);
    pe.body.appendChild(langRow);

    const optionsRow = el({});
    optionsRow.className = "minimax-pe-options-row";
    const detailItem = el({});
    detailItem.className = "minimax-pe-check-item";
    pe.detailCheck = document.createElement("input");
    pe.detailCheck.type = "checkbox";
    pe.detailCheck.checked = false;
    pe.detailCheck.title = t("promptEnhancer.detailTooltip");
    pe.detailCheck.onchange = () => pe.syncToWidgets();
    detailItem.appendChild(pe.detailCheck);
    pe.detailLabel = el({ cursor: "help" }, t("promptEnhancer.detail"), "span");
    pe.detailLabel.title = pe.detailCheck.title;
    detailItem.appendChild(pe.detailLabel);
    optionsRow.appendChild(detailItem);

    const autoItem = el({ cursor: "help" });
    autoItem.className = "minimax-pe-check-item";
    autoItem.title = t("promptEnhancer.autoTooltip");
    pe.autoCheck = document.createElement("input");
    pe.autoCheck.type = "checkbox";
    pe.autoCheck.checked = false;
    pe.autoCheck.title = autoItem.title;
    pe.autoCheck.onchange = () => pe.syncToWidgets();
    autoItem.appendChild(pe.autoCheck);
    pe.autoLabel = el({ cursor: "help" }, t("promptEnhancer.auto"), "span");
    pe.autoLabel.title = autoItem.title;
    autoItem.appendChild(pe.autoLabel);
    pe.autoItem = autoItem;
    optionsRow.appendChild(autoItem);

    pe.unloadWrap = el({});
    pe.unloadWrap.className = "minimax-pe-check-item";
    pe.unloadCheck = document.createElement("input");
    pe.unloadCheck.type = "checkbox";
    pe.unloadCheck.onchange = () => pe.syncToWidgets();
    pe.unloadWrap.appendChild(pe.unloadCheck);
    pe.unloadCheckLabel = el({}, t("promptEnhancer.unloadAfterOllama"), "span");
    pe.unloadWrap.appendChild(pe.unloadCheckLabel);
    optionsRow.appendChild(pe.unloadWrap);
    pe.body.appendChild(optionsRow);

    const btnRow = el({ display: "flex", gap: "6px", flexDirection: "column" });
    const enhanceRow = el({ display: "flex", gap: "6px" });
    pe.enhanceCurrentBtn = el({
        flex: "1", background: "#3b82f6", color: "#fff", border: "none", borderRadius: "4px",
        padding: "6px", fontWeight: "600", fontSize: "10px", cursor: "pointer",
    }, t("promptEnhancer.enhanceCurrent"), "button");
    pe.enhanceCurrentBtn.onclick = () => pe.enhancePrompt("current");
    enhanceRow.appendChild(pe.enhanceCurrentBtn);
    pe.enhanceAllBtn = el({
        flex: "1", background: "#6366f1", color: "#fff", border: "none", borderRadius: "4px",
        padding: "6px", fontWeight: "600", fontSize: "10px", cursor: "pointer",
    }, t("promptEnhancer.enhanceAll"), "button");
    pe.enhanceAllBtn.onclick = () => pe.enhancePrompt("all");
    enhanceRow.appendChild(pe.enhanceAllBtn);
    btnRow.appendChild(enhanceRow);
    const utilRow = el({ display: "flex", gap: "6px" });
    pe.unloadBtn = el({ background: "#252a34", color: "#e8ecf4", border: "1px solid #2a3140", borderRadius: "4px", padding: "6px 10px", fontSize: "10px", cursor: "pointer" }, t("promptEnhancer.unloadOllama"), "button");
    pe.unloadBtn.onclick = () => pe.unloadOllama();
    utilRow.appendChild(pe.unloadBtn);
    pe.unloadBtnRow = utilRow;
    btnRow.appendChild(utilRow);
    pe.body.appendChild(btnRow);

    pe.statusEl = el({ fontSize: "10px", color: STATUS_COLORS.info, minHeight: "16px", padding: "2px 0" });
    pe.body.appendChild(pe.statusEl);

    pe.templateArea = document.createElement("textarea");
    pe.templateArea.rows = 4;
    Object.assign(pe.templateArea.style, { width: "100%", fontSize: "9px", display: "none", background: "#12151b", color: "#d6dbe6", border: "1px solid #2a3140", borderRadius: "3px" });
    pe.templateArea.oninput = () => pe.syncToWidgets();
    swallowKeys(pe.templateArea);
    pe.body.appendChild(pe.templateArea);

    header.onclick = () => {
        pe.open = !pe.open;
        pe.body.style.display = pe.open ? "flex" : "none";
        pe.arrow.style.transform = pe.open ? "rotate(90deg)" : "";
        editor.updateDomWidgetHeight?.();
        if (pe.open && !pe.modelList.options.length) pe.fetchModels(true);
    };

    parentEl.appendChild(header);
    parentEl.appendChild(pe.body);

    pe.widget = (name) => editor.widget(name);

    pe.supportsUnload = () => {
        const fmt = pe.apiSelect.value;
        return fmt === API_OLLAMA
            || (fmt === API_OPENAI_COMPAT && normalizeOpenAiCompatMode(pe.compatSelect?.value) === OPENAI_COMPAT_LLAMA_SWAP);
    };

    pe.updateApiFormatUI = () => {
        const fmt = pe.apiSelect.value;
        const isOpenAi = fmt === API_OPENAI_COMPAT;
        const showKey = fmt === API_ZHIPU || isOpenAi;
        const supportsUnload = pe.supportsUnload();
        if (pe.compatRow) pe.compatRow.style.display = isOpenAi ? "flex" : "none";
        if (pe.keyRow) pe.keyRow.style.display = showKey ? "flex" : "none";
        if (pe.apiKeyInput) {
            pe.apiKeyInput.placeholder = fmt === API_ZHIPU
                ? t("promptEnhancer.zhipuKey")
                : t("promptEnhancer.openAiKey");
        }
        if (pe.unloadWrap) pe.unloadWrap.style.display = supportsUnload ? "flex" : "none";
        if (pe.unloadBtnRow) pe.unloadBtnRow.style.display = supportsUnload ? "flex" : "none";
        const unloadText = fmt === API_OLLAMA
            ? t("promptEnhancer.unloadAfterOllama")
            : t("promptEnhancer.unloadAfterModel");
        if (pe.unloadCheckLabel) pe.unloadCheckLabel.textContent = unloadText;
        if (pe.unloadBtn) {
            pe.unloadBtn.textContent = fmt === API_OLLAMA
                ? t("promptEnhancer.unloadOllama")
                : t("promptEnhancer.unloadModel");
        }
        pe.urlInput.placeholder = defaultsForApiFormat(fmt).url;
        pe.modelInput.placeholder = defaultsForApiFormat(fmt).model;
    };

    pe.applyLocale = () => {
        pe.titleEl.textContent = t("promptEnhancer.title");
        pe.descriptionEl.textContent = t("promptEnhancer.description");
        pe.refreshBtn.textContent = t("promptEnhancer.refreshModels");
        pe.compatLabel.textContent = t("promptEnhancer.openAiFeatures");
        pe.modelLabel.textContent = t("promptEnhancer.model");
        pe.languageLabel.textContent = t("promptEnhancer.outputLanguage");
        pe.compatSelect.title = t("promptEnhancer.compatTooltip");
        pe.detailCheck.title = t("promptEnhancer.detailTooltip");
        pe.detailLabel.textContent = t("promptEnhancer.detail");
        pe.detailLabel.title = pe.detailCheck.title;
        pe.autoItem.title = t("promptEnhancer.autoTooltip");
        pe.autoCheck.title = pe.autoItem.title;
        pe.autoLabel.textContent = t("promptEnhancer.auto");
        pe.autoLabel.title = pe.autoItem.title;
        const zhipuOption = [...pe.apiSelect.options].find((option) => option.value === API_ZHIPU);
        if (zhipuOption) zhipuOption.textContent = t("promptEnhancer.zhipu");
        const standardOption = [...pe.compatSelect.options].find((option) => option.value === OPENAI_COMPAT_STANDARD);
        if (standardOption) standardOption.textContent = t("promptEnhancer.standard");
        const chineseOption = [...pe.langSelect.options].find((option) => option.value === OUTPUT_LANGUAGE_ZH);
        if (chineseOption) chineseOption.textContent = t("promptEnhancer.chinese");
        const englishOption = [...pe.langSelect.options].find((option) => option.value === "English");
        if (englishOption) englishOption.textContent = t("promptEnhancer.english");
        pe.langSelect.title = t("promptEnhancer.languageTooltip");
        if (!pe._busy) pe.setEnhanceLoading(false);
        pe.updateApiFormatUI();
    };

    pe.syncFromWidgets = () => {
        const w = (n) => pe.widget(n);
        const explicitFmt = w("llm_api_format")?.value || DEFAULT_API_FORMAT;
        const fmt = inferApiFormat(w("llm_url")?.value, explicitFmt);
        const url = coerceLlmUrl(w("llm_url")?.value, defaultsForApiFormat(fmt).url);
        pe.urlInput.value = url;
        pe.apiSelect.value = fmt;
        pe._lastApiFormat = pe.apiSelect.value;
        if (pe.compatSelect) {
            pe.compatSelect.value = normalizeOpenAiCompatMode(w("llm_openai_compat_mode")?.value);
        }
        pe.modelInput.value = coerceLlmModel(w("llm_model")?.value);
        if (w("llm_api_key")) pe.apiKeyInput.value = w("llm_api_key").value || "";
        if (w("llm_auto_enhance")) pe.autoCheck.checked = !!w("llm_auto_enhance").value;
        else pe.autoCheck.checked = false;
        if (w("llm_unload_after")) pe.unloadCheck.checked = !!w("llm_unload_after").value;
        const lang = resolveOutputLanguage(pe);
        const prevLang = pe._lastOutputLanguage;
        pe._lastOutputLanguage = lang;
        if (pe.detailCheck) {
            pe.detailCheck.checked = resolveCharacterFeatureEnhance(pe, { preferWidget: true });
        }
        if (w("llm_custom_template")) pe.templateArea.value = w("llm_custom_template").value || "";
        pe.updateApiFormatUI();
        if (prevLang !== null && lang !== prevLang) pe.fetchTemplate(true);
    };

    pe.syncToWidgets = () => {
        const set = (n, v) => { const w = pe.widget(n); if (w) w.value = v; };
        const url = coerceLlmUrl(pe.urlInput.value, defaultsForApiFormat(pe.apiSelect.value).url);
        set("llm_api_format", pe.apiSelect.value);
        set("llm_openai_compat_mode", normalizeOpenAiCompatMode(pe.compatSelect?.value));
        set("llm_url", url);
        set("llm_api_key", pe.apiKeyInput.value || "");
        set("llm_model", coerceLlmModel(pe.modelInput.value));
        set("llm_output_language", pe.langSelect.value || DEFAULT_OUTPUT_LANGUAGE);
        set("llm_character_feature_enhance", !!pe.detailCheck?.checked);
        set("llm_auto_enhance", !!pe.autoCheck.checked);
        set("llm_unload_after", pe.supportsUnload() && !!pe.unloadCheck.checked);
        const custom = pe.templateArea.value.trim();
        set("llm_custom_template", custom !== pe._currentDefaultTemplate ? custom : "");
        editor._markNodeDirtyLight?.();
    };

    pe.fetchModels = async (silent = false) => {
        if (pe._busy) return;
        try {
            const llmUrl = coerceLlmUrl(pe.urlInput.value, defaultsForApiFormat(pe.apiSelect.value).url);
            pe.urlInput.value = llmUrl;
            pe.apiSelect.value = inferApiFormat(llmUrl, pe.apiSelect.value);
            pe.updateApiFormatUI();
            if (!silent) pe.setStatus(t("promptEnhancer.fetchingModels"), "loading");
            const resp = await api.fetchApi("/minimax/director/enhance_models", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    llm_url: llmUrl, api_format: pe.apiSelect.value,
                    openai_compat_mode: normalizeOpenAiCompatMode(pe.compatSelect?.value),
                    api_key: pe.apiKeyInput.value || "",
                }),
            });
            const data = await resp.json();
            if (!resp.ok) {
                if (!silent) pe.setStatus(data.error || t("promptEnhancer.fetchModelsFailed"), "error");
                return;
            }
            pe.modelList.innerHTML = "";
            for (const name of data.models || []) {
                const o = document.createElement("option");
                o.value = name;
                pe.modelList.appendChild(o);
            }
            if (!pe.modelInput.value.trim()) pe.modelInput.value = defaultsForApiFormat(pe.apiSelect.value).model;
            pe.syncToWidgets();
            if (!silent) pe.setStatus(t("promptEnhancer.modelCount", { count: (data.models || []).length }), "success");
        } catch (e) {
            if (!silent) pe.setStatus(t("promptEnhancer.connectionFailed", { error: e.message }), "error");
        }
    };

    pe.fetchTemplate = async (resetIfDefault = false) => {
        const task = resolveTaskKey(editor.getTaskKey?.() || "rv2v");
        const outputLanguage = resolveOutputLanguage(pe);
        try {
            const resp = await api.fetchApi("/minimax/director/get_template", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ task_type: task, output_language: outputLanguage }),
            });
            const data = await resp.json();
            if (data.template) {
                pe._currentDefaultTemplate = data.template;
                if (resetIfDefault || !pe.templateArea.value || pe.templateArea.value === pe._lastFetchedTemplate) {
                    pe.templateArea.value = data.template;
                }
                pe._lastFetchedTemplate = data.template;
            }
        } catch (e) { /* ignore */ }
    };

    pe.getPromptBlock = (segmentIndex = null) => {
        if (editor.isGlobalMode?.()) {
            editor.timeline.global = editor.timeline.global || {};
            return { block: editor.timeline.global, taskKey: resolveTaskKey(editor.getTaskKey?.() || "rv2v"), isGlobal: true };
        }
        const idx = segmentIndex ?? editor.selectedIndex ?? 0;
        const seg = editor.timeline.segments?.[idx];
        const global = editor.timeline.global || {};
        const taskKey = resolveTaskKey(seg?.taskType || global.taskType || editor.getTaskKey?.() || "rv2v");
        return { block: seg || global, taskKey, isGlobal: false, segmentIndex: idx };
    };

    pe.getPromptTextForBlock = (segmentIndex = null) => {
        if (editor.isGlobalMode?.()) {
            return (editor.globalPrompt?.value || editor.timeline.global?.prompt || editor.globalPromptWidget?.value || "").trim();
        }
        const idx = segmentIndex ?? editor.selectedIndex ?? 0;
        const seg = editor.timeline.segments?.[idx];
        const globalPrompt = (editor.timeline.global?.prompt || editor.globalPrompt?.value || "").trim();
        return (seg?.prompt || globalPrompt || "").trim();
    };

    pe.setPromptTextForBlock = (text, segmentIndex = null) => {
        if (editor.isGlobalMode?.()) {
            if (editor.globalPrompt) editor.globalPrompt.value = text;
            editor.timeline.global = editor.timeline.global || {};
            editor.timeline.global.prompt = text;
            if (editor.globalPromptWidget) editor.globalPromptWidget.value = text;
            return;
        }
        const idx = segmentIndex ?? editor.selectedIndex ?? 0;
        const seg = editor.timeline.segments?.[idx];
        if (seg) seg.prompt = text;
        if (idx === editor.selectedIndex && editor.segPrompt) editor.segPrompt.value = text;
    };

    pe.getActivePromptText = () => pe.getPromptTextForBlock();

    pe.setActivePromptText = (text) => {
        pe.setPromptTextForBlock(text);
        editor.commit?.(false, { syncTimeline: true });
    };

    pe.getLlmConfig = () => {
        if (pe.detailCheck) {
            pe.detailCheck.checked = resolveCharacterFeatureEnhance(pe, { preferWidget: true });
        }
        pe.syncToWidgets();
        const llmUrl = coerceLlmUrl(pe.urlInput.value, defaultsForApiFormat(pe.apiSelect.value).url);
        pe.urlInput.value = llmUrl;
        pe.apiSelect.value = inferApiFormat(llmUrl, pe.apiSelect.value);
        pe.updateApiFormatUI();
        const model = coerceLlmModel(pe.modelInput.value);
        pe.modelInput.value = model;
        const outputLanguage = resolveOutputLanguage(pe);
        const characterFeatureEnhance = resolveCharacterFeatureEnhance(pe, { preferWidget: true });
        const customTemplate = pe.templateArea.value.trim() !== pe._currentDefaultTemplate ? pe.templateArea.value.trim() : "";
        return {
            llmUrl, model, apiFormat: pe.apiSelect.value,
            openaiCompatMode: normalizeOpenAiCompatMode(pe.compatSelect?.value),
            apiKey: pe.apiKeyInput.value || "",
            outputLanguage,
            characterFeatureEnhance,
            customTemplate,
        };
    };

    pe.collectVisionImagesForBlock = async (block, taskKey) => {
        const images = [];
        let sourceCount = 0;
        let refCount = 0;
        const refSlots = [];
        const video = editor.timeline?.video || {};
        const videoFile = video.videoFile || video.fileName;
        const isOllama = pe.apiSelect?.value === API_OLLAMA;
        const sourceFrameCount = isOllama ? 2 : 3;
        if (videoFile && editor.getDirectorMode?.() === "video") {
            const resp = await api.fetchApi("/minimax/director/extract_frames", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    filename: videoFile,
                    subfolder: video.subfolder || "",
                    num_frames: sourceFrameCount,
                }),
            });
            const data = await resp.json();
            if (data.frames?.length) { sourceCount = data.frames.length; images.push(...data.frames); }
        }
        const global = editor.timeline.global || {};
        const refsBlock = block || global;
        if (taskUsesReferenceImages(taskKey) && refsBlock?.refs?.length) {
            const sortedRefs = [...refsBlock.refs]
                .filter((r) => r.imageFile || r.imageB64)
                .sort((a, b) => Number(a.index ?? a.slot ?? 0) - Number(b.index ?? b.slot ?? 0));
            for (const ref of sortedRefs) {
                const slot = Number(ref.index ?? ref.slot ?? 0);
                if (ref.imageFile) {
                    const b64 = await fetchImageB64(ref.imageFile);
                    if (b64) { images.push(b64); refCount += 1; refSlots.push(slot); }
                } else if (ref.imageB64) {
                    images.push(ref.imageB64.startsWith("data:") ? ref.imageB64.split(",", 2)[1] : ref.imageB64);
                    refCount += 1;
                    refSlots.push(slot);
                }
            }
        }
        let refVideoCount = 0;
        if (taskUsesReferenceVideo(taskKey)) {
            const rv = refsBlock?.referenceVideo || global.referenceVideo || {};
            const refVid = rv.videoFile || rv.fileName;
            if (refVid) {
                const resp = await api.fetchApi("/minimax/director/extract_frames", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        filename: refVid,
                        subfolder: rv.subfolder || "",
                        num_frames: isOllama ? 1 : 2,
                    }),
                });
                const data = await resp.json();
                if (data.frames?.length) {
                    refVideoCount = data.frames.length;
                    images.push(...data.frames);
                }
            }
        }
        return { images, sourceCount, refCount, refSlots, refVideoCount };
    };

    pe.callEnhanceApi = async (prompt, taskKey, block, cfg) => {
        let images = []; let refCount = 0; let sourceCount = 0; let refSlots = []; let refVideoCount = 0;
        try {
            ({
                images, refCount, sourceCount, refSlots, refVideoCount,
            } = await pe.collectVisionImagesForBlock(block, taskKey));
        } catch (e) {
            console.warn("[MiniMax H3 PE] vision collect failed:", e);
        }
        const resp = await api.fetchApi("/minimax/director/enhance", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    llm_url: cfg.llmUrl, model: cfg.model, prompt, task_type: taskKey,
                    image_num: Math.max(1, refCount), images, api_format: cfg.apiFormat,
                    openai_compat_mode: cfg.openaiCompatMode,
                    api_key: cfg.apiKey, output_language: cfg.outputLanguage,
                    character_feature_enhance: cfg.characterFeatureEnhance,
                    source_count: sourceCount, ref_slots: refSlots, ref_video_count: refVideoCount,
                    llm_unload_after: pe.supportsUnload() && !!pe.unloadCheck.checked, custom_template: cfg.customTemplate,
                }),
        });
        let data = {};
        try { data = await resp.json(); } catch { data = {}; }
        return {
            ok: resp.ok && !!data.response,
            text: data.response || "",
            error: data.error || (resp.ok ? t("promptEnhancer.emptyResponse") : `HTTP ${resp.status}`),
            hanCount: data.han_count,
            detailedMode: !!data.detailed_mode,
            detailTargetHan: data.detail_target_han,
            vision: { images, sourceCount, refCount },
        };
    };

    pe.enhanceOneTarget = async (segmentIndex, cfg, activeBtn, label) => {
        const { block, taskKey } = pe.getPromptBlock(segmentIndex);
        const prompt = pe.getPromptTextForBlock(segmentIndex);
        if (!prompt) return { ok: false, skipped: true, reason: "empty" };
        pe.setEnhanceLoading(true, activeBtn, label);
        pe.setStatus(`${t("promptEnhancer.enhancing")} ${label}`, "loading");
        const result = await pe.callEnhanceApi(prompt, taskKey, block, cfg);
        if (result.ok) {
            const text = taskKey === "fl2v" ? stripFl2vPromptBody(result.text) : result.text;
            pe.setPromptTextForBlock(text, segmentIndex);
            return { ok: true, chars: text.length, taskKey, result: { ...result, text } };
        }
        return { ok: false, error: result.error, taskKey };
    };

    pe.enhancePrompt = async (scope = "current") => {
        if (pe._busy) return;
        const cfg = pe.getLlmConfig();
        if (!cfg.model) { pe.setStatus(t("promptEnhancer.enterModel"), "error"); return; }
        if ((cfg.apiFormat === API_ZHIPU) && !cfg.apiKey) {
            pe.setStatus(t("promptEnhancer.enterApiKey"), "error");
            return;
        }

        const activeBtn = scope === "all" ? pe.enhanceAllBtn : pe.enhanceCurrentBtn;

        if (scope === "current") {
            const prompt = pe.getActivePromptText();
            if (!prompt) { pe.setStatus(t("promptEnhancer.enterPrompt"), "error"); return; }
            pe.setEnhanceLoading(true, activeBtn, t("promptEnhancer.preparing"));
            try {
                const { block, taskKey } = pe.getPromptBlock();
                pe.setEnhanceLoading(true, activeBtn, t("promptEnhancer.collectingAssets"));
                const result = await pe.callEnhanceApi(prompt, taskKey, block, cfg);
                const v = result.vision || {};
                if (v.images?.length) {
                    const sourceLabel = v.sourceCount
                        ? t("promptEnhancer.videoFrames", { count: v.sourceCount })
                        : "";
                    const refLabel = v.refCount
                        ? t("promptEnhancer.referenceImages", { count: v.refCount })
                        : "";
                    pe.visionBadge.textContent = `${sourceLabel}${sourceLabel && refLabel ? " + " : ""}${refLabel}`;
                    pe.visionBadge.style.display = "inline";
                } else {
                    pe.visionBadge.style.display = "none";
                }
                if (result.ok) {
                    const text = taskKey === "fl2v" ? stripFl2vPromptBody(result.text) : result.text;
                    pe.setActivePromptText(text);
                    pe.setStatus(formatEnhanceSuccessStatus(taskKey, { ...result, text }), "success");
                } else {
                    pe.setStatus(result.error, "error");
                }
            } catch (e) {
                pe.setStatus(t("promptEnhancer.requestFailed", { error: e.message }), "error");
            } finally {
                pe.setEnhanceLoading(false);
            }
            return;
        }

        // scope === "all"
        if (editor.isGlobalMode?.()) {
            const prompt = pe.getActivePromptText();
            if (!prompt) { pe.setStatus(t("promptEnhancer.enterGlobalPrompt"), "error"); return; }
            try {
                const r = await pe.enhanceOneTarget(null, cfg, activeBtn, t("promptEnhancer.globalPrompt"));
                if (r.ok) {
                    editor.commit?.(false, { syncTimeline: true });
                    pe.setStatus(formatEnhanceSuccessStatus(r.taskKey, r.result || {}), "success");
                } else if (!r.skipped) {
                    pe.setStatus(r.error || t("promptEnhancer.enhanceFailed"), "error");
                }
            } catch (e) {
                pe.setStatus(t("promptEnhancer.requestFailed", { error: e.message }), "error");
            } finally {
                pe.setEnhanceLoading(false);
            }
            return;
        }

        const segments = editor.timeline.segments || [];
        const targets = segments.map((_, i) => i).filter((i) => pe.getPromptTextForBlock(i));
        if (!targets.length) {
            pe.setStatus(t("promptEnhancer.noSegmentPrompts"), "error");
            return;
        }

        let okCount = 0;
        let lastError = "";
        try {
            for (let n = 0; n < targets.length; n++) {
                const idx = targets[n];
                const label = t("promptEnhancer.segment", { index: idx + 1, total: segments.length });
                const r = await pe.enhanceOneTarget(idx, cfg, activeBtn, label);
                if (r.ok) {
                    okCount += 1;
                    pe.setStatus(t("promptEnhancer.segmentSuccess", { label, done: okCount, total: targets.length }), "loading");
                } else if (!r.skipped) {
                    lastError = r.error || t("promptEnhancer.unknownError");
                    pe.setStatus(t("promptEnhancer.segmentFailed", { label, error: lastError }), "error");
                    break;
                }
            }
            editor.commit?.(false, { syncTimeline: true });
            editor.updateSelectionUI?.();
            if (okCount === targets.length) {
                pe.setStatus(t("promptEnhancer.allDone", { count: okCount }), "success");
            } else if (okCount > 0 && lastError) {
                pe.setStatus(t("promptEnhancer.partialDone", { done: okCount, total: targets.length, error: lastError }), "error");
            }
        } catch (e) {
            pe.setStatus(t("promptEnhancer.requestFailed", { error: e.message }), "error");
        } finally {
            pe.setEnhanceLoading(false);
        }
    };

    pe.unloadModel = async () => {
        const llmUrl = coerceLlmUrl(pe.urlInput.value, defaultsForApiFormat(pe.apiSelect.value).url);
        const apiFormat = pe.apiSelect.value;
        const openaiCompatMode = normalizeOpenAiCompatMode(pe.compatSelect?.value);
        const model = coerceLlmModel(pe.modelInput.value);
        if (!model) { pe.setStatus(t("promptEnhancer.enterModel"), "error"); return; }
        if (!pe.supportsUnload()) {
            pe.setStatus(t("promptEnhancer.unsupportedUnload"), "error");
            return;
        }
        pe.setStatus(t("promptEnhancer.unloading"), "loading");
        try {
            const resp = await api.fetchApi("/minimax/director/unload_model", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    llm_url: llmUrl,
                    model,
                    api_format: apiFormat,
                    openai_compat_mode: openaiCompatMode,
                    api_key: pe.apiKeyInput.value || "",
                }),
            });
            const data = await resp.json();
            if (resp.ok && data.status === "unloaded") {
                pe.setStatus(t("promptEnhancer.modelUnloaded", { provider: data.provider || "LLM" }), "success");
            } else {
                pe.setStatus(data.error || t("promptEnhancer.unloadFailed", { error: t("promptEnhancer.unknownError") }), "error");
            }
        } catch (e) {
            pe.setStatus(t("promptEnhancer.unloadFailed", { error: e.message }), "error");
        }
    };
    pe.unloadOllama = pe.unloadModel;

    pe.onTaskTypeChanged = () => pe.fetchTemplate();
    pe.handleServerEnhanced = (payload) => {
        if (!payload || String(payload.node) !== String(editor.node.id)) return;
        let text = payload.text || "";
        if (editor.isFl2vMode?.() || resolveTaskKey(editor.getTaskKey?.() || "") === "fl2v") {
            text = stripFl2vPromptBody(text);
        }
        pe.setActivePromptText(text);
        pe.setStatus(t("promptEnhancer.autoApplied", { count: text.length }), "success");
    };

    pe._lastOutputLanguage = null;
    pe.syncFromWidgets();
    editor._promptEnhancer = pe;
    pe.applyLocale();
    pe.fetchTemplate(true);
    return pe;
}

export function getPromptEnhancerPanelHeight(editor) {
    const pe = editor?._promptEnhancer;
    if (!pe?.open) return PE_PANEL_COLLAPSED_H;
    return PE_PANEL_COLLAPSED_H + PE_PANEL_EXPANDED_H;
}

export function registerDirectorPromptEnhancerEvents(findDirectorNode) {
    api.addEventListener("minimax_director_enhanced", ({ detail }) => {
        findDirectorNode(detail?.node)?._minimaxEditor?._promptEnhancer?.handleServerEnhanced?.(detail);
    });
}
