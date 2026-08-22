const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");

const source = fs.readFileSync(path.join(__dirname, "../web/js/minimax_refine.js"), "utf8");
const start = source.indexOf("const FOLLOW_DIRECTOR_ASPECT");
const end = source.indexOf("\nfunction readMode", start);
const helpers = Function(`${source.slice(start, end)}\nreturn {normalizeAspect, isFollowAspect, isScaleByAspect, isCustomAspect};`)();

test("normalizes legacy Chinese aspect values to English labels", () => {
    const cases = [
        ["跟随导演台", "Follow Director"],
        ["按倍数", "Scale by multiplier"],
        ["1:1 (方形)", "1:1 (Square)"],
        ["2:3 (竖版照片)", "2:3 (Portrait photo)"],
        ["3:2 (横版照片)", "3:2 (Landscape photo)"],
        ["3:4 (竖版标准)", "3:4 (Portrait standard)"],
        ["4:3 (标准)", "4:3 (Standard)"],
        ["9:16 (竖屏)", "9:16 (Portrait)"],
        ["16:9 (宽屏)", "16:9 (Widescreen)"],
        ["21:9 (超宽)", "21:9 (Ultrawide)"],
        ["自定义", "Custom"],
    ];
    for (const [legacy, expected] of cases) {
        assert.equal(helpers.normalizeAspect(legacy), expected, legacy);
    }
});

test("classifies English aspect modes without changing their meaning", () => {
    assert.equal(helpers.isFollowAspect("Follow Director"), true);
    assert.equal(helpers.isScaleByAspect("Scale by multiplier"), true);
    assert.equal(helpers.isCustomAspect("Custom"), true);
    assert.equal(helpers.isCustomAspect("16:9 (Widescreen)"), false);
});
