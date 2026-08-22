const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");

const source = fs.readFileSync(path.join(__dirname, "../web/js/minimax_timeline.js"), "utf8");
const start = source.indexOf("function isAdvancedDirectorNode");
const end = source.indexOf("\nfunction applyConfiguredWidgetValues", start);
const migrateAdvancedWidgetValues = Function(`${source.slice(start, end)}\nreturn migrateAdvancedWidgetValues;`)();

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
