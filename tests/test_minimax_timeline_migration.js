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

function makeCurrentAdvancedNode() {
    const names = [
        "task_type", "global_prompt", "bd_grp_sample", "cfg", "seed", "control_after_generate",
        "frame_rate", "width", "height", "ref_max_size", "total_frames", "timeline_data",
        "bd_grp_advanced", "steps", "sampler", "scheduler", "shift_video", "shift_audio",
        "lora_config", "enable_sage_attention", "sage_attention", "allow_sage_compile",
        "enable_h3_mem_eff_sage", "enable_sol_attn", "sol_tau", "sol_start_percent", "sol_end_percent",
        "sol_min_tokens", "sol_int8_qk", "sol_sink_conditioning", "sol_morton", "sol_morton_curve",
        "sol_int8_pv", "sol_verbose", "sol_use_tma", "sol_tau_profile", "sol_dense_blocks",
        "enable_easycache", "easycache_reuse_threshold", "easycache_start_percent", "easycache_end_percent",
        "easycache_verbose", "enable_firstblock_cache", "firstblock_cache_threshold", "firstblock_cache_verbose",
        "enable_cfg_norm", "cfg_norm_strength", "cfg_norm_pre_cfg", "bd_grp_perf",
        "clear_vram_between_segments", "export_source_images", "minimax_director_ui",
    ];
    return {
        comfyClass: "MiniMaxH3DirectorAdvanced",
        widgets: names.map((name) => ({ name })),
    };
}

function makeCurrentAdvancedValues(node) {
    const values = Array(node.widgets.length).fill(false);
    const set = (name, value) => {
        values[node.widgets.findIndex((widget) => widget.name === name)] = value;
    };
    set("bd_grp_advanced", "Advanced sampling");
    set("steps", 4);
    set("sampler", "euler");
    set("scheduler", "simple");
    set("shift_video", 12);
    set("shift_audio", 3);
    set("lora_config", "[]");
    set("firstblock_cache_threshold", 0.08);
    set("cfg_norm_strength", 0.95);
    set("bd_grp_perf", "Performance");
    set("clear_vram_between_segments", true);
    set("export_source_images", false);
    set("minimax_director_ui", "");
    return values;
}

test("repairs a legacy array that omitted lora_config", () => {
    const node = makeCurrentAdvancedNode();
    const values = makeCurrentAdvancedValues(node);
    const loraIndex = node.widgets.findIndex((widget) => widget.name === "lora_config");
    values.splice(loraIndex, 1);
    values.splice(values.length - 3, 0, "Performance");
    const config = { widgets_values: values };

    assert.equal(migrateAdvancedWidgetValues(node, config), true);
    assert.equal(config.widgets_values[loraIndex], "[]");
    assert.equal(config.widgets_values[node.widgets.findIndex((widget) => widget.name === "steps")], 4);
    assert.deepEqual(config.widgets_values.slice(-4), ["Performance", true, false, ""]);
});

test("repairs a legacy array that omitted the base sampling widgets", () => {
    const node = makeCurrentAdvancedNode();
    const values = makeCurrentAdvancedValues(node);
    const stepsIndex = node.widgets.findIndex((widget) => widget.name === "steps");
    values.splice(stepsIndex, 5);
    values.splice(values.length - 3, 0, ...Array(5).fill("Performance"));
    const config = { widgets_values: values };

    assert.equal(migrateAdvancedWidgetValues(node, config), true);
    assert.deepEqual(config.widgets_values.slice(stepsIndex, stepsIndex + 5), [25, "res_multistep", "simple", 12, 3]);
    assert.equal(config.widgets_values[node.widgets.findIndex((widget) => widget.name === "lora_config")], "[]");
    assert.deepEqual(config.widgets_values.slice(-4), ["Performance", true, false, ""]);
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
