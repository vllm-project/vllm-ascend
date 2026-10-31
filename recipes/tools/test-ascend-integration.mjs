// Self-contained fixtures exercise the official assembler and renderer offline.
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";
import { spawnSync } from "node:child_process";
import { parseArgs } from "node:util";
import { after, test } from "node:test";
import { assertInstallationAdapter } from "./ascend-checks.mjs";
import { run } from "./validate-ascend.mjs";

const { values } = parseArgs({ options: { renderer: { type: "string" } } });
assert.ok(values.renderer, "Pass --renderer PATH.");
const renderer = path.resolve(values.renderer);
const require = createRequire(path.join(renderer, "package.json"));
const yaml = require("js-yaml");
const cli = fileURLToPath(new URL("validate-ascend.mjs", import.meta.url));
const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "ascend-recipe-test-"));
const write = (file, object) => { fs.mkdirSync(path.dirname(file), { recursive: true }); fs.writeFileSync(file, yaml.dump(object)); };
const coreDir = path.join(tmp, "core");
const empty = path.join(tmp, "empty");
fs.mkdirSync(empty);
const core = { meta: { hardware: { b200: "verified" } }, model: { model_id: "test/Fixture" },
  compatible_strategies: ["single_node_tp"], variants: { default: { precision: "bf16", vram_minimum_gb: 1, supported_hardware: ["b200"] } } };
write(path.join(coreDir, "models/test/Fixture.yaml"), core);
const sources = yaml.load(fs.readFileSync(path.join(renderer, "sources.yaml"), "utf8"));
const overrides = Object.keys(sources.channels[sources.default_channel]).filter((name) => name !== "vllm-ascend")
  .map((name) => `${name}=${name === "vllm" ? coreDir : empty}`);
const overlay = { meta: { hardware: { ascend_910c: "verified" } }, variants: { ascend: {
  model_id: "test/Ascend", precision: "int8", vram_minimum_gb: 1, supported_hardware: ["ascend_910c"], tp: 4,
  hardware_overrides: { ascend_910c: { docker_image: "registry.example/custom:1", install: { pip: false } } },
} } };
let next = 0;
function fixture(change = () => {}) {
  const recipes = path.join(tmp, `source-${next++}`);
  const doc = structuredClone(overlay);
  change(doc);
  write(path.join(recipes, "overlays/test/Fixture.yaml"), doc);
  return recipes;
}
const config = (extra = {}) => ({ renderer, recipes: fixture(), override: overrides, ...extra });
const hasCode = (report, code) => report.findings.some((finding) => finding.code === code);
after(() => {
  assert.equal(path.dirname(tmp), path.resolve(os.tmpdir()));
  assert.ok(path.basename(tmp).startsWith("ascend-recipe-test-"));
  fs.rmSync(tmp, { recursive: true, force: true });
});

test("valid fixture uses the official assembler and has no semantic errors", async () => {
  const report = await run(config());
  assert.equal(report.summary.scenarios, 1);
  assert.equal(report.summary.blocking, 0);
  assert.equal(report.provenance.vllm.local, true);
});

test("inherited Rust frontend selection is preserved and disclosed in coverage", async () => {
  const recipes = fixture((o) => { o.variants.ascend.extra_env = { VLLM_USE_RUST_FRONTEND: "1" }; });
  const report = await run(config({ recipes }));
  assert.equal(report.results[0].workers[0].env.VLLM_USE_RUST_FRONTEND, "1");
  assert.match(report.coverage, /Python frontend requested/);
  assert.ok(report.limitations.some((text) => text.includes("VLLM_USE_RUST_FRONTEND")));
});
test("missing effective pip configuration is an error independent of image", async () => {
  const recipes = fixture((o) => { delete o.variants.ascend.hardware_overrides.ascend_910c.install; });
  const report = await run(config({ recipes }));
  assert.ok(hasCode(report, "ASC002"));
  assert.equal(report.summary.blocking, 1);
});
test("same findings in the target snapshot remain visible but do not block", async () => {
  const recipes = fixture((o) => { delete o.variants.ascend.hardware_overrides.ascend_910c.install; });
  const report = await run(config({ recipes, "baseline-recipes": recipes }));
  assert.equal(report.summary.existing_errors, 1);
  assert.equal(report.summary.new_errors, 0);
  assert.equal(report.summary.blocking, 0);
});
test("changed invalid image evidence blocks even when the rule code already existed", async () => {
  const bad = (image) => fixture((o) => { o.variants.ascend.hardware_overrides.ascend_910c.docker_image = image; });
  const report = await run(config({ recipes: bad("vllm/vllm-openai:nightly"), "baseline-recipes": bad("vllm/vllm-openai:latest") }));
  assert.equal(report.summary.new_errors, 1);
  assert.equal(report.summary.resolved_errors, 1);
  assert.equal(report.summary.blocking, 1);
});
test("fixed errors are reported as resolved", async () => {
  const baseline = fixture((o) => { delete o.variants.ascend.hardware_overrides.ascend_910c.install; });
  const report = await run(config({ "baseline-recipes": baseline }));
  assert.equal(report.summary.errors, 0);
  assert.equal(report.summary.resolved_errors, 1);
});
test("missing baseline cannot silently become a clean baseline", async () => {
  await assert.rejects(run(config({ "baseline-recipes": path.join(tmp, "missing") })), /ENOENT/);
});
test("new model absent from a valid empty baseline is fully checked", async () => {
  const bad = fixture((o) => { delete o.variants.ascend.hardware_overrides.ascend_910c.install; });
  const report = await run(config({ recipes: bad, "baseline-recipes": empty }));
  assert.equal(report.summary.new_errors, 1);
});
test("unmatched selection cannot produce a passing zero-coverage report", async () => {
  assert.ok(hasCode(await run(config({ variant: "typo" })), "SOURCE_EMPTY"));
});
test("duplicate YAML keys fail official strict assembly", async () => {
  const recipes = fixture();
  fs.writeFileSync(path.join(recipes, "overlays/test/Fixture.yaml"), "variants: {}\nvariants: {}\n");
  await assert.rejects(run(config({ recipes })), /duplicated mapping key/);
});
test("orphaned overlays fail official strict assembly", async () => {
  const recipes = fixture();
  fs.renameSync(path.join(recipes, "overlays/test/Fixture.yaml"), path.join(recipes, "overlays/test/Orphan.yaml"));
  await assert.rejects(run(config({ recipes })), /no recipe test\/Orphan/);
});
test("overlay extension is independent from the core recipe extension", async () => {
  const recipes = fixture();
  fs.renameSync(path.join(recipes, "overlays/test/Fixture.yaml"), path.join(recipes, "overlays/test/Fixture.yml"));
  assert.equal((await run(config({ recipes }))).summary.blocking, 0);
});
test("inherited RECIPES overrides cannot replace the candidate source", async () => {
  const previous = process.env.RECIPES_OVERRIDES;
  try {
    process.env.RECIPES_OVERRIDES = `vllm-ascend=${empty}`;
    assert.equal((await run(config())).summary.scenarios, 1);
  } finally {
    if (previous === undefined) delete process.env.RECIPES_OVERRIDES;
    else process.env.RECIPES_OVERRIDES = previous;
  }
});
test("caller cannot override candidate source through the local source override option", async () => {
  await assert.rejects(run(config({ override: [`vllm-ascend=${empty}`] })), /Invalid source override/);
});
test("UI installation semantics changes are detected but unrelated text is allowed", () => {
  const ui = fs.readFileSync(path.join(renderer, "src/components/recipes/CommandBuilder.jsx"), "utf8");
  assert.doesNotThrow(() => assertInstallationAdapter(ui + "\n// unrelated comment\n"));
  assert.throws(() => assertInstallationAdapter(ui.replace("hwInstall?.pip === false", "hwInstall?.pip === true")), /semantics changed/);
});
test("CLI saves reports and uses exit 0, 1 and 2 for success, findings and bad input", () => {
  const recipes = fixture();
  const args = ["--disable-warning=MODULE_TYPELESS_PACKAGE_JSON", cli, "--renderer", renderer, "--recipes", recipes,
    ...overrides.flatMap((value) => ["--override", value])];
  const env = { ...process.env };
  delete env.GITHUB_STEP_SUMMARY;
  const output = path.join(tmp, "report.json");
  assert.equal(spawnSync(process.execPath, [...args, "--report", output], { encoding: "utf8", env }).status, 0);
  assert.ok(fs.existsSync(output.replace(/\.json$/, ".md")));
  const doc = structuredClone(overlay);
  delete doc.variants.ascend.hardware_overrides.ascend_910c.install;
  write(path.join(recipes, "overlays/test/Fixture.yaml"), doc);
  assert.equal(spawnSync(process.execPath, args, { encoding: "utf8", env }).status, 1);
  assert.equal(spawnSync(process.execPath, [cli], { encoding: "utf8", env }).status, 2);
  const failure = path.join(tmp, "failure.json");
  assert.equal(spawnSync(process.execPath, [...args, "--baseline-recipes", path.join(tmp, "missing-baseline"),
    "--report", failure], { encoding: "utf8", env }).status, 2);
  assert.equal(JSON.parse(fs.readFileSync(failure, "utf8")).status, "tool-error");
});
test("temporary channels are cleaned without changing renderer sources or public output", async () => {
  const sourceBefore = fs.readFileSync(path.join(renderer, "sources.yaml"), "utf8");
  await run(config());
  assert.equal(fs.readFileSync(path.join(renderer, "sources.yaml"), "utf8"), sourceBefore);
  const leftovers = fs.readdirSync(path.join(renderer, ".content")).filter((name) => /^ascend-(head|base)-/.test(name));
  assert.deepEqual(leftovers, []);
});
test("identical baseline/head paths do not allow an empty candidate selection", async () => {
  const recipes = fixture();
  await assert.rejects(run(config({ recipes, "baseline-recipes": recipes, model: "test/Typo" })), /No matching overlays/);
});
test("symbolic links in source trees are rejected rather than silently skipped", async () => {
  const recipes = fixture();
  const target = path.join(tmp, "linked-directory");
  fs.mkdirSync(target);
  fs.symlinkSync(target, path.join(recipes, "overlays/link"), "junction");
  await assert.rejects(run(config({ recipes })), /Symbolic links/);
});
test("PD expectations are read independently from candidate and baseline snapshots", async () => {
  const pdCore = path.join(tmp, "pd-core");
  write(path.join(pdCore, "models/test/Fixture.yaml"), { ...core, compatible_strategies: ["pd_cluster"] });
  const make = (connector) => {
    const recipes = fixture((o) => { o.variants.ascend.extra_env = {
      ASCEND_RT_VISIBLE_DEVICES: Array.from({ length: 16 }, (_, i) => i).join(","),
    }; });
    fs.mkdirSync(path.join(recipes, "validation"));
    fs.writeFileSync(path.join(recipes, "validation/pd-expectations.json"), JSON.stringify({
      "test/Fixture": { ascend_910c: { ascend: { connector, router: "vllm-router" } } },
    }));
    return recipes;
  };
  const report = await run(config({ recipes: make("NixlConnector"), "baseline-recipes": make("MooncakeConnectorV1"),
    override: overrides.map((value) => value.startsWith("vllm=") ? `vllm=${pdCore}` : value) }));
  assert.equal(report.summary.errors, 0);
  assert.equal(report.summary.resolved_errors, 2);
});
test("invalid PD expectation structure fails instead of silently disabling its checks", async () => {
  const recipes = fixture();
  fs.mkdirSync(path.join(recipes, "validation"));
  fs.writeFileSync(path.join(recipes, "validation/pd-expectations.json"), "null");
  await assert.rejects(run(config({ recipes })), /PD expectations must/);
});
test("device evidence changes are new errors even when the generic finding message is unchanged", async () => {
  const bad = (ids) => fixture((o) => { o.variants.ascend.extra_env = { ASCEND_RT_VISIBLE_DEVICES: ids }; });
  const report = await run(config({ recipes: bad("1,1,2,3"), "baseline-recipes": bad("0,0,2,3") }));
  assert.ok(report.comparison.blocking.some((finding) => finding.code === "ASC022"));
});
