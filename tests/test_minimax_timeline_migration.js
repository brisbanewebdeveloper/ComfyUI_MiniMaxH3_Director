const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");

const source = fs.readFileSync(path.join(__dirname, "../web/js/minimax_timeline.js"), "utf8");
const start = source.indexOf("function isAdvancedDirectorNode");
const end = source.indexOf("\nfunction applyConfiguredWidgetValues", start);
const migrations = Function(`
    function resolveTaskKey(value) {
        return String(value || "").split(/ — | —— | - | – /, 1)[0].trim();
    }
    ${source.slice(start, end)}
    return { migrateAdvancedWidgetValues, migrateEnglishWidgetValues };
`)();
const { migrateAdvancedWidgetValues, migrateEnglishWidgetValues } = migrations;

function makeNode(widgetCount = 10) {
    const names = ["base_a", "base_b", "enable_firstblock_cache", "firstblock_cache_threshold", "firstblock_cache_verbose", "enable_cfg_norm", "cfg_norm_strength", "cfg_norm_pre_cfg"];
    while (names.length < widgetCount) names.push(`tail_${names.length}`);
    return {
        comfyClass: "MiniMaxH3DirectorAdvanced",
        widgets: names.map((name) => ({ name })),
    };
}

test("adds CFGNorm defaults after FirstBlockCache in legacy workflows", () => {
    const node = makeNode();
    const config = { widgets_values: ["a", "b", true, 0.2, true, "tail", 12] };

    assert.equal(migrateAdvancedWidgetValues(node, config), true);
    assert.deepEqual(config.widgets_values, ["a", "b", true, 0.2, true, false, 0.95, false, "tail", 12]);
});

test("adds both enhancement groups when loading a workflow without them", () => {
    const node = makeNode();
    const config = { widgets_values: ["a", "b", "tail", 12] };

    assert.equal(migrateAdvancedWidgetValues(node, config), true);
    assert.deepEqual(config.widgets_values, ["a", "b", false, 0.08, false, false, 0.95, false, "tail", 12]);
});

test("leaves current valid widget values unchanged", () => {
    const node = makeNode();
    const config = { widgets_values: ["a", "b", true, 0.2, true, true, 0.95, false, "tail", 12] };
    const original = [...config.widgets_values];

    assert.equal(migrateAdvancedWidgetValues(node, config), false);
    assert.deepEqual(config.widgets_values, original);
});

test("rewrites legacy Director task labels in English", () => {
    const node = {
        comfyClass: "MiniMaxH3Director",
        widgets: [{ name: "task_type" }, { name: "global_prompt" }],
    };
    const config = { widgets_values: ["fl2v — 首尾帧生视频(First-Last Frame)", "A prompt"] };

    assert.equal(migrateEnglishWidgetValues(node, config), true);
    assert.deepEqual(config.widgets_values, ["fl2v — First-Last Frame to Video", "A prompt"]);
});

test("rewrites legacy prompt enhancer choices without changing their meaning", () => {
    const node = {
        comfyClass: "MiniMaxH3DirectorEnhancePrompt",
        widgets: [
            { name: "prompt" },
            { name: "task_type" },
            { name: "openai_compat_mode" },
            { name: "output_language" },
        ],
    };
    const config = {
        widgets_values: [
            "A prompt",
            "r2v — 参考主体生视频(Reference to Video)",
            "标准",
            "中文",
        ],
    };

    assert.equal(migrateEnglishWidgetValues(node, config), true);
    assert.deepEqual(config.widgets_values, [
        "A prompt",
        "r2v — Reference to Video",
        "Standard",
        "Chinese",
    ]);
});
