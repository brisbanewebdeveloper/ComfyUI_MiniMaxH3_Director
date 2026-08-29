const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");

const i18nSource = fs.readFileSync(path.join(__dirname, "../web/js/minimax_i18n.js"), "utf8");
const enhancerSource = fs.readFileSync(path.join(__dirname, "../web/js/minimax_prompt_enhancer.js"), "utf8");

function readDictionary(name, nextMarker) {
    const start = i18nSource.indexOf(`const ${name} = {`);
    const end = i18nSource.indexOf(nextMarker, start);
    const objectSource = i18nSource.slice(i18nSource.indexOf("{", start), end).trim().replace(/;$/, "");
    return Function(`return (${objectSource});`)();
}

const zh = readDictionary("ZH", "\n\nconst EN =");
const en = readDictionary("EN", "\n\nconst DICTS =");

test("English is the default for the versioned locale preference", () => {
    assert.match(i18nSource, /LOCALE_STORAGE_KEY = "mmx_director_ui_locale_v2"/);
    assert.match(i18nSource, /function detectDefaultLocale\(\) \{\s*return "en";/);
});

test("prompt enhancer translation keys exist in both dictionaries", () => {
    const usedKeys = [...enhancerSource.matchAll(/t\("(promptEnhancer\.[^"]+)"/g)].map((match) => match[1]);

    assert.ok(usedKeys.length > 20);
    for (const key of new Set(usedKeys)) {
        assert.equal(typeof en[key], "string", `missing English key: ${key}`);
        assert.equal(typeof zh[key], "string", `missing Chinese key: ${key}`);
    }
});
