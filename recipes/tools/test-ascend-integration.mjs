// Self-contained fixtures exercise the official assembler and renderer offline.
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { createRequire } from "node:module";
import { fileURLToPath, pathToFileURL } from "node:url";
import { spawnSync } from "node:child_process";
import { parseArgs } from "node:util";
import { after, test } from "node:test";
import { assertInstallationAdapter, installationFor } from "./ascend-checks.mjs";
import { run } from "./validate-ascend.mjs";

const { values } = parseArgs({ options: { renderer: { type: "string" } } });
assert.ok(values.renderer, "Pass --renderer PATH.");
const renderer = path.resolve(values.renderer);
const synthesis = await import(pathToFileURL(path.join(renderer, "src/lib/command-synthesis.js")).href);
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

function presetConfig({ pd = false, points = false, active = true, badRole = false } = {}) {
  const presetCore = path.join(tmp, `preset-core-${next++}`);
  write(path.join(presetCore, "models/test/Fixture.yaml"), { ...structuredClone(core),
    model: { model_id: "test/Fixture", architecture: "moe", default_frontend: "rust" },
    features: { spec_decoding: { args: ["--speculative-config", '{"method":"mtp","num_speculative_tokens":1}'] } },
    opt_in_features: ["spec_decoding"], compatible_strategies: [pd ? "pd_cluster" : "single_node_tp"] });
  const recipes = fixture((o) => {
    o.presets = { tuned: { hardware: "ascend_910c", strategy: pd ? "pd_cluster" : "single_node_tp",
      variants: ["ascend"], default: active, image: { vllm: "registry.example/preset:2" },
      select: { frontend: "python", features: { spec_decoding: true }, kv_offload: "off",
        ...(pd ? { pd_router: "ascend", pd_connector: "mooncake_ascend" } : {}) } } };
    if (pd) {
      o.variants.ascend.supported_frontends = ["python"];
      o.presets.tuned.roles = Object.fromEntries(["prefill", "decode"].map((role) => [role, {
        nodes: 1, parallelism: "dep", tp: 1, pp: 1,
        json_args: { "--kv-transfer-config": { MooncakeConnectorV2: {
          kv_connector: "MooncakeHybridConnector", ...(badRole && role === "prefill" ? { kv_role: "kv_consumer" } : {}),
        } } },
      }]));
    }
    if (points) o.presets.tuned.points = {
      low: { default: true, image: { vllm: "registry.example/point:low" },
        json_args: { "--speculative-config": { num_speculative_tokens: 2 } } },
      high: { image: { vllm: "registry.example/point:high" },
        json_args: { "--speculative-config": { num_speculative_tokens: 5 } } },
    };
  });
  return config({ recipes, override: overrides.map((value) => value.startsWith("vllm=") ? `vllm=${presetCore}` : value) });
}

test("default preset applies its image, feature selection and frontend before checking", async () => {
  const report = await run(presetConfig());
  const checked = report.results.find((r) => r.preset === "tuned");
  assert.ok(checked, "The default preset must be checked, not only baseline");
  assert.equal(checked.installation.image, "registry.example/preset:2");
  assert.equal(checked.frontend, "python");
  assert.notEqual(checked.workers[0].env.VLLM_USE_RUST_FRONTEND, "1");
  assert.ok(checked.workers[0].argv.includes('{"method":"mtp","num_speculative_tokens":1}'));
});

test("opt-in PD preset is checked with Ascend router and its edited Mooncake connector", async () => {
  const report = await run(presetConfig({ pd: true, active: false }));
  const checked = report.results.find((r) => r.preset === "tuned");
  assert.ok(checked, "A non-default preset must not be omitted");
  assert.equal(checked.router, "ascend-proxy");
  for (const worker of checked.workers) {
    const i = worker.argv.indexOf("--kv-transfer-config");
    const kv = JSON.parse(worker.argv[i + 1]);
    assert.equal(kv.kv_connector, "MooncakeHybridConnector");
    assert.equal(kv.kv_role, worker.role === "prefill" ? "kv_producer" : "kv_consumer");
    assert.ok(worker.argv.includes('{"method":"mtp","num_speculative_tokens":1}'));
  }
  assert.equal(checked.findings.filter((f) => f.severity === "error").length, 0);
});

test("every declared preset point is checked with its own image and JSON edits", async () => {
  const report = await run(presetConfig({ points: true, active: false }));
  const points = report.results.filter((r) => r.preset === "tuned");
  assert.deepEqual(points.map((r) => r.point).sort(), ["high", "low"]);
  for (const point of points) {
    assert.equal(point.installation.image, `registry.example/point:${point.point}`);
    const argv = point.workers[0].argv;
    assert.equal(JSON.parse(argv[argv.indexOf("--speculative-config") + 1]).num_speculative_tokens,
      point.point === "low" ? 2 : 5);
  }
});

test("an invalid KV role in an opt-in preset still blocks without external expectations", async () => {
  const report = await run(presetConfig({ pd: true, active: false, badRole: true }));
  assert.ok(report.comparison.blocking.some((f) => f.preset === "tuned" && f.code === "ASC014"));
});

test("the default multi-node preset keeps its declared node count", async () => {
  const options = presetConfig();
  const corePath = path.join(options.override.find((v) => v.startsWith("vllm=")).slice(5), "models/test/Fixture.yaml");
  const doc = yaml.load(fs.readFileSync(corePath, "utf8"));
  doc.compatible_strategies = ["multi_node_tp"];
  write(corePath, doc);
  const overlayPath = path.join(options.recipes, "overlays/test/Fixture.yaml");
  const patch = yaml.load(fs.readFileSync(overlayPath, "utf8"));
  patch.presets.tuned.strategy = "multi_node_tp";
  patch.presets.tuned.select.nodes = 4;
  write(overlayPath, patch);
  const report = await run(options);
  const checked = report.results.find((r) => r.preset === "tuned");
  assert.ok(checked);
  assert.equal(checked.nodes, 4);
  assert.equal(checked.workers.length, 4);
  assert.ok(checked.workers[3].argv.includes("3"));
});

test("legacy baseline PD expectations do not override an opt-in preset connector", async () => {
  const options = presetConfig({ pd: true, active: false });
  fs.mkdirSync(path.join(options.recipes, "validation"));
  fs.writeFileSync(path.join(options.recipes, "validation/pd-expectations.json"), JSON.stringify({
    "test/Fixture": { ascend_910c: { ascend: { connector: "NixlConnector", router: "vllm-router" } } },
  }));
  const report = await run(options);
  assert.ok(report.results.some((r) => r.preset === "tuned"));
  assert.equal(report.comparison.blocking.length, 0);
});

test("a hardware-wide preset omits inferred NVIDIA-only variants", async () => {
  const options = presetConfig();
  const file = path.join(options.recipes, "overlays/test/Fixture.yaml");
  const patch = yaml.load(fs.readFileSync(file, "utf8"));
  delete patch.presets.tuned.variants;
  write(file, patch);
  const report = await run(options);
  assert.equal(report.summary.blocking, 0);
  assert.deepEqual(report.results.filter((r) => r.preset === "tuned").map((r) => r.variant), ["ascend"]);
});

test("a strategy-wide preset omits inferred strategies unsupported on Ascend", async () => {
  const options = presetConfig();
  const corePath = path.join(options.override.find((v) => v.startsWith("vllm=")).slice(5), "models/test/Fixture.yaml");
  const doc = yaml.load(fs.readFileSync(corePath, "utf8"));
  doc.compatible_strategies.push("multi_node_tp");
  write(corePath, doc);
  const file = path.join(options.recipes, "overlays/test/Fixture.yaml");
  const patch = yaml.load(fs.readFileSync(file, "utf8"));
  delete patch.presets.tuned.strategy;
  patch.strategy_hardware = { multi_node_tp: { npu: "unsupported" } };
  write(file, patch);
  const report = await run(options);
  assert.equal(report.summary.blocking, 0);
  assert.deepEqual(report.results.filter((r) => r.preset === "tuned").map((r) => r.strategy), ["single_node_tp"]);
});

test("a preset explicitly targeting an incompatible variant still blocks", async () => {
  const options = presetConfig();
  const file = path.join(options.recipes, "overlays/test/Fixture.yaml");
  const patch = yaml.load(fs.readFileSync(file, "utf8"));
  patch.presets.tuned.variants = ["default", "ascend"];
  write(file, patch);
  const report = await run(options);
  assert.ok(report.comparison.blocking.some((f) => f.preset === "tuned" && f.variant === "default" && f.code === "SOURCE_RENDER"));
});

test("image and pip fallbacks are retained as warnings without failing the gate", async () => {
  const recipes = fixture((o) => { o.variants.ascend.hardware_overrides.ascend_910c = {}; });
  const report = await run(config({ recipes }));
  assert.deepEqual(report.findings.map((f) => [f.code, f.severity]), [["ASC001", "warning"], ["ASC002", "warning"]]);
  assert.equal(report.summary.blocking, 0);
  assert.equal(report.results[0].status, "warning");
});
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

function unrestrictedVariantConfig(declareA5 = false) {
  const unrestrictedCore = path.join(tmp, "unrestricted-core");
  write(path.join(unrestrictedCore, "models/test/Fixture.yaml"), {
    ...structuredClone(core), variants: { default: { precision: "bf16", vram_minimum_gb: 1 } },
  });
  const recipes = fixture((o) => {
    o.variants.default = { hardware_overrides: structuredClone(overlay.variants.ascend.hardware_overrides) };
    if (declareA5) o.meta.hardware.ascend_950dt = "supported";
  });
  return config({ recipes, override: overrides.map((value) => value.startsWith("vllm=") ? `vllm=${unrestrictedCore}` : value) });
}

test("undeclared restricted hardware does not create fallback errors for an inherited variant", async () => {
  const report = await run(unrestrictedVariantConfig());
  assert.deepEqual(report.results.filter((r) => r.variant === "default").map((r) => r.hardware), ["ascend_910c"]);
  assert.equal(report.summary.blocking, 0);
});

test("declared restricted hardware still reports missing Ascend installation settings as warnings", async () => {
  const report = await run(unrestrictedVariantConfig(true));
  assert.deepEqual(report.findings.map((f) => [f.hardware, f.code, f.severity]),
    [["ascend_950dt", "ASC001", "warning"], ["ascend_950dt", "ASC002", "warning"]]);
  assert.equal(report.summary.blocking, 0);
});

test("an explicit hardware filter cannot enable an undeclared restricted profile", async () => {
  const report = await run({ ...unrestrictedVariantConfig(), hardware: "ascend_950dt" });
  assert.ok(hasCode(report, "SOURCE_EMPTY"));
});

function standaloneFixture(change = () => {}) {
  const recipes = fixture();
  const doc = { ...structuredClone(core), meta: { hardware: { ascend_910c: "verified" } },
    model: { model_id: "test/AscendOnly" }, variants: structuredClone(overlay.variants) };
  change(doc);
  write(path.join(recipes, "models/test/AscendOnly.yaml"), doc);
  return recipes;
}

test("standalone Ascend recipes and overlays are both checked", async () => {
  const report = await run(config({ recipes: standaloneFixture() }));
  assert.equal(report.summary.models, 2);
  assert.equal(report.summary.scenarios, 2);
  assert.equal(report.summary.blocking, 0);
  assert.ok(report.inputs.find((input) => input.model === "test/AscendOnly").recipe_sha256);
});

test("standalone recipe errors block and existing baseline errors remain visible", async () => {
  const recipes = standaloneFixture((doc) => {
    doc.variants.ascend.hardware_overrides.ascend_910c.install = { docker: false, pip: false };
  });
  const report = await run(config({ recipes, "baseline-recipes": standaloneFixture() }));
  assert.equal(report.summary.new_errors, 1);
  assert.equal(report.comparison.blocking[0].model, "test/AscendOnly");
  assert.equal(report.comparison.blocking[0].code, "ASC004");
  const unchanged = await run(config({ recipes, "baseline-recipes": recipes }));
  assert.equal(unchanged.summary.existing_errors, 1);
  assert.equal(unchanged.summary.blocking, 0);
});

test("a standalone-only source does not need an overlays directory", async () => {
  const recipes = standaloneFixture();
  fs.rmSync(path.join(recipes, "overlays"), { recursive: true });
  const report = await run(config({ recipes }));
  assert.equal(report.summary.models, 1);
  assert.equal(report.results[0].model, "test/AscendOnly");
  assert.equal(report.summary.blocking, 0);
});

test("inherited Rust frontend selection is preserved", async () => {
  const recipes = fixture((o) => { o.variants.ascend.extra_env = { VLLM_USE_RUST_FRONTEND: "1" }; });
  const report = await run(config({ recipes }));
  assert.equal(report.results[0].workers[0].env.VLLM_USE_RUST_FRONTEND, "1");
});
test("missing effective pip configuration warns independently of image", async () => {
  const recipes = fixture((o) => { delete o.variants.ascend.hardware_overrides.ascend_910c.install; });
  const report = await run(config({ recipes }));
  assert.ok(hasCode(report, "ASC002"));
  assert.equal(report.summary.warnings, 1);
  assert.equal(report.summary.blocking, 0);
});

test("hardware pip command replaces the inherited generic installation", async () => {
  const recipes = fixture((o) => {
    o.variants.ascend.hardware_overrides.ascend_910c.install.pip = { command: "uv pip install vllm-ascend" };
  });
  const report = await run(config({ recipes }));
  assert.equal(report.summary.errors, 0);
  assert.equal(report.results[0].installation.pipCommand, "uv pip install vllm-ascend");
  assert.equal(report.results[0].installation.defaultPip, false);
});

test("variant installation can disable pip or supply its Ascend command", async () => {
  for (const pip of [false, { command: "uv pip install vllm-ascend" }]) {
    const recipes = fixture((o) => {
      delete o.variants.ascend.hardware_overrides.ascend_910c.install;
      o.variants.ascend.install = { pip };
    });
    const report = await run(config({ recipes }));
    assert.equal(report.summary.errors, 0);
    assert.equal(report.results[0].installation.pipEnabled, pip !== false);
  }
});

test("hardware pip command re-enables a variant's disabled pip tab", async () => {
  const recipes = fixture((o) => {
    o.variants.ascend.install = { pip: false };
    o.variants.ascend.hardware_overrides.ascend_910c.install.pip = { command: "uv pip install vllm-ascend" };
  });
  const report = await run(config({ recipes }));
  assert.equal(report.summary.errors, 0);
  assert.equal(report.results[0].installation.pipEnabled, true);
});

test("hardware Docker opt-out permits a pip-only configuration without an image", async () => {
  const recipes = fixture((o) => {
    const hw = o.variants.ascend.hardware_overrides.ascend_910c;
    delete hw.docker_image;
    hw.install = { docker: false, pip: { command: "uv pip install vllm-ascend" } };
  });
  const report = await run(config({ recipes }));
  assert.equal(report.summary.errors, 0);
  assert.equal(report.results[0].installation.dockerEnabled, false);
});

test("hardware profile installation is resolved by the shared renderer helpers", () => {
  const profile = { brand: "Huawei", generation: "npu", gpu_count: 8,
    install: { pip: { command: "uv pip install vllm-ascend" } } };
  const recipe = { model: { install: { pip: false } }, variants: { ascend: {} } };
  const meta = synthesis.computeDockerMeta(recipe, recipe.variants.ascend, profile, "ascend_910c");
  const install = installationFor(recipe, "ascend", "ascend_910c", meta, synthesis, profile);
  assert.equal(install.pipEnabled, true);
  assert.equal(install.pipCommand, "uv pip install vllm-ascend");
});
test("same findings in the target snapshot remain visible but do not block", async () => {
  const recipes = fixture((o) => { delete o.variants.ascend.hardware_overrides.ascend_910c.install; });
  const report = await run(config({ recipes, "baseline-recipes": recipes }));
  assert.equal(report.comparison.existing.filter((f) => f.code === "ASC002" && f.severity === "warning").length, 1);
  assert.equal(report.summary.new_errors, 0);
  assert.equal(report.summary.blocking, 0);
});
test("changed fallback image evidence is a new warning even when the rule code already existed", async () => {
  const bad = (image) => fixture((o) => { o.variants.ascend.hardware_overrides.ascend_910c.docker_image = image; });
  const report = await run(config({ recipes: bad("vllm/vllm-openai:nightly"), "baseline-recipes": bad("vllm/vllm-openai:latest") }));
  assert.equal(report.comparison.additions.filter((f) => f.code === "ASC001" && f.severity === "warning").length, 1);
  assert.equal(report.comparison.resolved.filter((f) => f.code === "ASC001").length, 1);
  assert.equal(report.summary.blocking, 0);
});
test("fixed installation warnings are reported as resolved", async () => {
  const baseline = fixture((o) => { delete o.variants.ascend.hardware_overrides.ascend_910c.install; });
  const report = await run(config({ "baseline-recipes": baseline }));
  assert.equal(report.summary.errors, 0);
  assert.equal(report.comparison.resolved.filter((f) => f.code === "ASC002").length, 1);
});
test("missing baseline cannot silently become a clean baseline", async () => {
  await assert.rejects(run(config({ "baseline-recipes": path.join(tmp, "missing") })), /ENOENT/);
});
test("new model absent from a valid empty baseline is fully checked", async () => {
  const bad = fixture((o) => { delete o.variants.ascend.hardware_overrides.ascend_910c.install; });
  const report = await run(config({ recipes: bad, "baseline-recipes": empty }));
  assert.equal(report.comparison.additions.filter((f) => f.code === "ASC002" && f.severity === "warning").length, 1);
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
test("missing shared installation APIs fail instead of reverting to stale UI rules", () => {
  assert.doesNotThrow(() => assertInstallationAdapter(synthesis));
  assert.throws(() => assertInstallationAdapter({ ...synthesis, resolveInstall: undefined }), /resolveInstall/);
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
  assert.equal(spawnSync(process.execPath, args, { encoding: "utf8", env }).status, 0);
  assert.equal(spawnSync(process.execPath, [...args, "--warnings-as-errors"], { encoding: "utf8", env }).status, 1);
  doc.variants.ascend.extra_env = { ASCEND_RT_VISIBLE_DEVICES: "0,0" };
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
  await assert.rejects(run(config({ recipes, "baseline-recipes": recipes, model: "test/Typo" })), /No matching recipes or overlays/);
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
