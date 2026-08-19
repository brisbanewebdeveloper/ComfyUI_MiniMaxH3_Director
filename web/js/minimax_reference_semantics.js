/** Structured reference metadata editor for deterministic MiniMax prompt sections. */

const IMAGE_RETENTION = ["fully_preserved", "partially_preserved", "reference", "weak_reference"];
const AUDIO_RETENTION = ["fully_copy", "partially_copy", "reference", "weak_reference"];

function field(label, control) {
    const wrap = document.createElement("label");
    wrap.className = "bd-ref-sem-field";
    const title = document.createElement("span");
    title.textContent = label;
    wrap.append(title, control);
    return wrap;
}

function input(value = "") {
    const el = document.createElement("input");
    el.type = "text";
    el.value = value;
    return el;
}

function textarea(value = "") {
    const el = document.createElement("textarea");
    el.rows = 3;
    el.value = value;
    return el;
}

function select(options, value) {
    const el = document.createElement("select");
    for (const option of options) {
        const item = document.createElement("option");
        item.value = option;
        item.textContent = option;
        item.selected = option === value;
        el.appendChild(item);
    }
    return el;
}

function ensureRecord(target, collection, index) {
    target[collection] = Array.isArray(target[collection]) ? target[collection] : [];
    let record = target[collection].find((item) => Number(item.index ?? item.slot) === index);
    if (!record) {
        record = { index };
        target[collection].push(record);
    }
    return record;
}

export const REFERENCE_SEMANTICS_STYLES = `
.bd-ref-sem-backdrop{position:fixed;inset:0;z-index:10080;background:rgba(0,0,0,.66);display:flex;align-items:center;justify-content:center;padding:20px}
.bd-ref-sem-dialog{width:min(520px,calc(100vw - 32px));max-height:calc(100vh - 40px);overflow:auto;background:#171717;border:1px solid #444;border-radius:10px;padding:16px;box-shadow:0 16px 48px rgba(0,0,0,.55);color:#eee}
.bd-ref-sem-dialog h3{margin:0 0 12px;font-size:14px}.bd-ref-sem-grid{display:grid;grid-template-columns:1fr 1fr;gap:10px}
.bd-ref-sem-field{display:flex;flex-direction:column;gap:4px;font-size:11px;color:#aaa}.bd-ref-sem-field.wide{grid-column:1/-1}
.bd-ref-sem-field input,.bd-ref-sem-field select,.bd-ref-sem-field textarea{box-sizing:border-box;width:100%;background:#101010;color:#eee;border:1px solid #383838;border-radius:5px;padding:7px;font:inherit}
.bd-ref-sem-actions{display:flex;justify-content:flex-end;gap:8px;margin-top:14px}.bd-ref-sem-actions button{border:1px solid #444;border-radius:5px;background:#282828;color:#eee;padding:6px 12px;cursor:pointer}.bd-ref-sem-actions .primary{border-color:#43875c;background:#21452f}
@media(max-width:560px){.bd-ref-sem-grid{grid-template-columns:1fr}.bd-ref-sem-field.wide{grid-column:auto}}
`;

export function openReferenceSemanticsEditor({ target, collection, index, media, label, onSave }) {
    if (!target || !collection) return;
    const record = ensureRecord(target, collection, index);
    const backdrop = document.createElement("div");
    backdrop.className = "bd-ref-sem-backdrop";
    const dialog = document.createElement("div");
    dialog.className = "bd-ref-sem-dialog";
    const heading = document.createElement("h3");
    heading.textContent = `${label} · Reference semantics`;
    const grid = document.createElement("div");
    grid.className = "bd-ref-sem-grid";

    const kind = input(record.subjectKind || record.subject_kind || record.kind || "subject");
    const description = textarea(record.description || record.describes || "");
    const retentionOptions = media === "audio" ? AUDIO_RETENTION : IMAGE_RETENTION;
    const retention = select(retentionOptions, record.retention || "reference");
    const retained = textarea(record.retained || record.retentionDescription || "");
    grid.append(field("Kind", kind), field("Retention", retention));
    const descriptionField = field("Description", description);
    descriptionField.classList.add("wide");
    const retainedField = field("What must be retained", retained);
    retainedField.classList.add("wide");
    grid.append(descriptionField, retainedField);

    let voiceOf = null;
    if (media === "audio") {
        voiceOf = input(record.voiceOf || record.voice_of || "");
        voiceOf.placeholder = "Subject 1";
        const voiceField = field("Voice of", voiceOf);
        voiceField.classList.add("wide");
        grid.appendChild(voiceField);
    }

    const actions = document.createElement("div");
    actions.className = "bd-ref-sem-actions";
    const cancel = document.createElement("button");
    cancel.type = "button";
    cancel.textContent = "Cancel";
    const save = document.createElement("button");
    save.type = "button";
    save.className = "primary";
    save.textContent = "Save";
    actions.append(cancel, save);
    dialog.append(heading, grid, actions);
    backdrop.appendChild(dialog);
    document.body.appendChild(backdrop);

    const close = () => backdrop.remove();
    cancel.onclick = close;
    backdrop.addEventListener("click", (event) => { if (event.target === backdrop) close(); });
    save.onclick = () => {
        record.subjectKind = kind.value.trim();
        record.description = description.value.trim();
        record.retention = retention.value;
        record.retained = retained.value.trim();
        if (voiceOf) record.voiceOf = voiceOf.value.trim();
        onSave?.(record);
        close();
    };
}
