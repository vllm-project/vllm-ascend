#!/usr/bin/env node
// Runs after the official schema/API check; never executes recipe commands.
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { createRequire } from "node:module";
import { fileURLToPath, pathToFileURL } from "node:url";
import { execFileSync } from "node:child_process";
import { createHash, randomUUID } from "node:crypto";
import { parseArgs } from "node:util";
import { assertInstallationAdapter, checkScenario, installationFor } from "./ascend-checks.mjs";

const DEFAULT_RECIPES = fileURLToPath(new URL("../", import.meta.url));
const hash = (text) => createHash("sha256").update(text).digest("hex");
const identity = (f) => JSON.stringify([f.model, f.hardware, f.variant, f.strategy, f.role, f.code, f.severity, f.message, f.evidence]);
export function compareFindings(current, baseline = null, warningsAsErrors = false) {
  const previous = new Set((baseline || []).map(identity));
  const now = new Set(current.map(identity));
  const additions = current.filter((f) => !previous.has(identity(f)));
  return { additions, existing: current.filter((f) => previous.has(identity(f))),
    resolved: (baseline || []).filter((f) => !now.has(identity(f))),
    blocking: current.filter((f) => f.code.startsWith("SOURCE_")
      || (!previous.has(identity(f)) && (f.severity === "error" || warningsAsErrors))) };
}
function yamlFiles(dir) {
  if (!fs.existsSync(dir)) return [];
  return fs.readdirSync(dir, { withFileTypes: true }).sort((a, b) => a.name.localeCompare(b.name))
    .flatMap((e) => {
      if (e.isSymbolicLink()) throw new Error(`Symbolic links are not supported in checked source trees: ${path.join(dir, e.name)}`);
      return e.isDirectory() ? yamlFiles(path.join(dir, e.name))
        : e.isFile() && /\.ya?ml$/.test(e.name) ? [path.join(dir, e.name)] : [];
    });
}
function indexedYaml(dir) {
  const files = new Map();
  for (const file of yamlFiles(dir)) {
    const id = path.relative(dir, file).replaceAll(path.sep, "/").replace(/\.ya?ml$/, "");
    if (files.has(id)) throw new Error(`Duplicate model ID: ${id}`);
    files.set(id, file);
  }
  return files;
}
function workersFor(result) {
  if (result.deployType === "pd_cluster") return ["prefill", "decode"].map((role) => ({ role, ...result[role] }));
  if (result.deployType === "multi_node") return [{ role: "head", argv: result.headArgv, env: result.env },
    ...(result.workerArgvs || []).map((argv, i) => ({ role: `worker-${i + 1}`, argv, env: result.env }))];
  return [{ role: "server", argv: result.argv, env: result.env }];
}
function normalize(value) {
  if (value instanceof Date) return value.toISOString().slice(0, 10);
  if (Array.isArray(value)) return value.map(normalize);
  return value && typeof value === "object"
    ? Object.fromEntries(Object.entries(value).map(([k, v]) => [k, normalize(v)])) : value;
}

function inspectSource(recipes, content, tools, options, allowEmpty = false) {
  const { readYaml, synth, taxonomy, strategies } = tools;
  const expectationFile = options["pd-expectations"] || path.join(recipes, "validation/pd-expectations.json");
  const expectations = fs.existsSync(expectationFile) ? JSON.parse(fs.readFileSync(expectationFile, "utf8")) : {};
  if (options["pd-expectations"] && !fs.existsSync(expectationFile)) throw new Error("Explicit PD expectation file is missing");
  if (!expectations || typeof expectations !== "object" || Array.isArray(expectations)) throw new Error("PD expectations must be a model-keyed object");
  for (const hardware of Object.values(expectations)) {
    if (!hardware || typeof hardware !== "object" || Array.isArray(hardware)) throw new Error("Invalid PD expectation hardware map");
    for (const variants of Object.values(hardware)) {
      if (!variants || typeof variants !== "object" || Array.isArray(variants)) throw new Error("Invalid PD expectation variant map");
      for (const expected of Object.values(variants)) {
        if (!expected || typeof expected.connector !== "string" || !expected.connector.trim()
          || typeof expected.router !== "string" || !expected.router.trim()
          || Object.keys(expected).some((key) => !["connector", "router"].includes(key))) throw new Error("PD expectation requires connector and router strings only");
      }
    }
  }
  const snapshot = { inputs: [], results: [], findings: [], skipped: [] };
  if (yamlFiles(path.join(recipes, "models")).length) throw new Error("Full recipes under models/ are not covered; only overlays are supported.");
  const overlays = indexedYaml(path.join(recipes, "overlays"));
  const assembled = indexedYaml(path.join(content, "models"));
  for (const [hfId, file] of overlays) {
    if (options.model && options.model !== hfId) continue;
    const assembledFile = assembled.get(hfId);
    if (!assembledFile) throw new Error(`No assembled recipe for ${hfId}`);
    const recipe = readYaml(assembledFile);
    snapshot.inputs.push({ model: hfId, overlay_sha256: hash(fs.readFileSync(file)), assembled_sha256: hash(fs.readFileSync(assembledFile)) });
    const before = snapshot.results.length;
    const sourceError = (code, message) => snapshot.findings.push({ model: hfId, code, severity: "error", message,
      fix: "Review the declared hardware/strategy selection and renderer compatibility." });
    for (const [variantKey, variant] of Object.entries(recipe.variants || {})) {
      if (options.variant && options.variant !== variantKey) continue;
      for (const hw of synth.listCompatibleHardware(taxonomy.hardware_profiles, variant, recipe)) {
        if (options.hardware && options.hardware !== hw) continue;
        const profile = taxonomy.hardware_profiles[hw];
        if (profile.brand !== "Huawei" || profile.generation !== "npu") continue;
        for (const strategyName of synth.effectiveCompatibleStrategies(recipe)) {
          if (options.strategy && options.strategy !== strategyName) continue;
          if (!synth.isStrategySupportedOnHardware(recipe, strategyName, profile, hw)) continue;
          const strategy = strategies[strategyName];
          if (!strategy) { sourceError("SOURCE_STRATEGY", `Unknown strategy ${strategyName}`); continue; }
          const context = { model: hfId, hardware: hw, variant: variantKey, strategy: strategyName };
          if (!synth.isStrategyReachable(recipe, strategyName, strategy.deploy_type, profile.gpu_count, hw)
            || (strategy.deploy_type === "single_node" && !synth.fitsSingleNode(profile, variant, hw))) {
            snapshot.skipped.push({ ...context, reason: "Renderer resource/reachability gate" }); continue;
          }
          const minNodes = Math.max(1, synth.nodesForStrategy(recipe, strategyName, profile.gpu_count, hw),
            Math.ceil(synth.variantVramMinimumGb(variant, hw) / (profile.vram_gb || Infinity)));
          const nodes = strategy.deploy_type === "multi_node" ? Math.max(2, minNodes) : strategy.deploy_type === "pd_cluster" ? 2 : 1;
          if (nodes > synth.MAX_NODES) { snapshot.skipped.push({ ...context, reason: "Exceeds renderer node limit" }); continue; }
          const pdNodes = strategy.deploy_type === "pd_cluster" ? {
            prefill: Math.max(minNodes, synth.pdRoleNodes(recipe, "prefill", profile, hw) ?? 1),
            decode: Math.max(minNodes, synth.pdRoleNodes(recipe, "decode", profile, hw) ?? 1),
          } : null;
          const result = synth.resolveCommand(recipe, variantKey, strategyName, hw, [], strategies, taxonomy,
            [], nodes, pdNodes, {}, null, null, "python", null, "http", null, "baseline");
          const workers = workersFor(result);
          if (!workers.length || workers.some((w) => !Array.isArray(w.argv) || !w.argv.length)) {
            sourceError("SOURCE_RENDER", `No structured worker argv for ${hw}/${variantKey}/${strategyName}`); continue;
          }
          const installation = installationFor(recipe, variantKey, hw, synth.computeDockerMeta(recipe, variant, profile, hw));
          const findings = checkScenario({ profile, deployType: result.deployType, installation, workers,
            router: result.orchestrator, expectedPD: expectations[hfId]?.[hw]?.[variantKey] });
          snapshot.results.push({ ...context, nodes, pdNodes, installation, workers, findings,
            status: findings.some((f) => f.severity === "error") ? "fail" : findings.length ? "warning" : "pass" });
          snapshot.findings.push(...findings.map((f) => ({ ...context, ...f })));
        }
      }
    }
    if (snapshot.results.length === before) sourceError("SOURCE_EMPTY", "Selected model produced zero checked Huawei configurations.");
  }
  if (!snapshot.inputs.length && !allowEmpty) throw new Error("No matching overlays; zero checks cannot pass.");
  return snapshot;
}

export async function run(options) {
  const renderer = path.resolve(options.renderer);
  const recipes = path.resolve(options.recipes || DEFAULT_RECIPES);
  if (!fs.statSync(recipes).isDirectory()) throw new Error("--recipes must be a directory");
  const baseline = options["baseline-recipes"] ? path.resolve(options["baseline-recipes"]) : null;
  if (baseline && !fs.statSync(baseline).isDirectory()) throw new Error("Baseline recipe directory is missing");
  const sha = execFileSync("git", ["-C", renderer, "rev-parse", "HEAD"], { encoding: "utf8" }).trim();
  const dirty = execFileSync("git", ["-C", renderer, "status", "--porcelain", "--untracked-files=no"], { encoding: "utf8" }).trim();
  if (dirty) throw new Error("Use a renderer checkout without tracked modifications.");
  assertInstallationAdapter(fs.readFileSync(path.join(renderer, "src/components/recipes/CommandBuilder.jsx"), "utf8"));
  const require = createRequire(path.join(renderer, "package.json"));
  const yaml = require("js-yaml");
  const readYaml = (file) => normalize(yaml.load(fs.readFileSync(file, "utf8")));
  const sources = readYaml(path.join(renderer, "sources.yaml"));
  const original = sources.channels?.[sources.default_channel];
  if (original?.["vllm-ascend"]?.role !== "overlay" || !original["vllm-ascend"].scope?.brands?.includes("Huawei")) {
    throw new Error("Renderer must register the Huawei-scoped vllm-ascend overlay source.");
  }
  const lock = JSON.parse(fs.readFileSync(path.join(renderer, "sources.lock.json"), "utf8"));
  const overrides = {};
  for (const override of options.override || []) {
    const equal = override.indexOf("=");
    const name = override.slice(0, equal);
    if (equal < 1 || !original[name] || name === "vllm-ascend") throw new Error(`Invalid source override: ${override}`);
    overrides[name] = path.resolve(override.slice(equal + 1));
  }
  const synth = await import(pathToFileURL(path.join(renderer, "src/lib/command-synthesis.js")).href);
  const taxonomy = readYaml(path.join(renderer, "taxonomy.yaml"));
  const strategies = Object.fromEntries(["strategies", "kv_store"].flatMap((dir) =>
    yamlFiles(path.join(renderer, dir)).map((file) => { const spec = readYaml(file); return [spec.name, spec]; })));
  const tools = { readYaml, synth, taxonomy, strategies };
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "ascend-recipe-check-"));
  const id = randomUUID();
  const channels = { [`ascend-head-${id}`]: recipes, ...(baseline ? { [`ascend-base-${id}`]: baseline } : {}) };
  const config = { default_channel: Object.keys(channels)[0], channels: {} };
  const pins = {};
  for (const [channel, candidate] of Object.entries(channels)) {
    config.channels[channel] = Object.fromEntries(Object.entries(original).map(([name, spec]) => [name,
      name === "vllm-ascend" ? { ...spec, local: candidate }
        : overrides[name] ? { ...spec, local: overrides[name] }
          : spec.local ? { ...spec, local: path.resolve(renderer, spec.local) } : spec]));
    pins[channel] = lock[sources.default_channel] || {};
  }
  fs.writeFileSync(path.join(tmp, "sources.yaml"), yaml.dump(config));
  fs.writeFileSync(path.join(tmp, "sources.lock.json"), JSON.stringify(pins));
  const env = Object.fromEntries(Object.entries(process.env).filter(([key]) => !key.startsWith("RECIPES_")));
  Object.assign(env, { RECIPES_SOURCES: path.join(tmp, "sources.yaml"), CI: "true" });
  const execute = (argv) => {
    try { execFileSync(process.execPath, argv, { cwd: renderer, env, encoding: "utf8", stdio: ["ignore", "pipe", "pipe"], maxBuffer: 16 * 1024 * 1024 }); }
    catch (error) { throw new Error(`Official source preparation failed: ${error.stderr || error.stdout || error.message}`); }
  };
  try {
    execute(["scripts/sync-sources.mjs", "--channel", Object.keys(channels)[0]]);
    const snapshots = [];
    const provenance = [];
    for (const [channel, candidate] of Object.entries(channels)) {
      execute(["scripts/assemble-content.mjs", "--channel", channel, "--strict"]);
      const content = path.join(renderer, ".content", channel);
      snapshots.push(inspectSource(candidate, content, tools, options, snapshots.length > 0));
      provenance.push(JSON.parse(fs.readFileSync(path.join(content, "provenance.json"), "utf8")).sources);
    }
    const head = snapshots[0];
    const comparison = compareFindings(head.findings, baseline ? snapshots[1].findings : null, options["warnings-as-errors"]);
    const report = { version: 1, created_at: new Date().toISOString(), renderer_sha: sha,
      mode: baseline ? "no-new-errors" : "strict", provenance: provenance[0], baseline_provenance: provenance[1] || null,
      coverage: "Huawei overlays; baseline preset, features off, Python frontend requested with inherited environment preserved; minimum multi-node topology. No model execution.",
      limitations: ["No full recipes, tuned presets, feature combinations or exhaustive topology enumeration.",
        "Inherited VLLM_USE_RUST_FRONTEND may still enable Rust; frontend compatibility is not verified.",
        "Multi-node/PD local device capacity and runtime/package/image compatibility are not verified.",
        "Known errors retained from the PR target branch remain errors, but do not block no-new-errors mode."],
      ...head, comparison };
    report.summary = { models: head.inputs.length, scenarios: head.results.length,
      passed: head.results.filter((r) => r.status === "pass").length,
      errors: head.findings.filter((f) => f.severity === "error").length,
      warnings: head.findings.filter((f) => f.severity === "warning").length,
      new_errors: comparison.additions.filter((f) => f.severity === "error").length,
      existing_errors: comparison.existing.filter((f) => f.severity === "error").length,
      resolved_errors: comparison.resolved.filter((f) => f.severity === "error").length,
      blocking: comparison.blocking.length };
    return report;
  } finally {
    // Only the UUID channels created above; never remove latest/validate.
    for (const channel of Object.keys(channels)) fs.rmSync(path.join(renderer, ".content", channel), { recursive: true, force: true });
    fs.rmSync(tmp, { recursive: true, force: true });
  }
}

const safeCell = (value) => String(value ?? "—").replaceAll("|", "\\|").replace(/[\r\n]/g, " ").replaceAll("<", "&lt;");
export function markdownReport(report, limit = Infinity) {
  const lines = ["# Ascend recipe configuration checks", "", `Renderer: \`${report.renderer_sha}\`; mode: **${report.mode}**.`, "",
    `Models: ${report.summary.models}; configurations: ${report.summary.scenarios}; new errors: ${report.summary.new_errors}; existing errors: ${report.summary.existing_errors}; blocking: ${report.summary.blocking}.`, "",
    "A successful no-new-errors check does not mean every configuration is valid or NPU-tested.", "",
    "| Category | Model | Hardware / variant / strategy | Rule | Finding |", "| --- | --- | --- | --- | --- |"];
  let shown = 0;
  for (const [category, findings] of [["New", report.comparison.additions], ["Existing", report.comparison.existing], ["Resolved", report.comparison.resolved]]) {
    for (const f of findings) {
      if (shown++ >= limit) continue;
      lines.push(`| ${category} | ${safeCell(f.model)} | ${safeCell([f.hardware, f.variant, f.strategy].filter(Boolean).join(" / "))} | ${safeCell(f.code)} | ${safeCell(f.message)} |`);
    }
  }
  if (shown > limit) lines.push("", `Showing ${limit} findings; download the configuration report artifact for the full list.`);
  lines.push("", "## Scope", "", report.coverage, "", ...report.limitations.map((text) => `- ${text}`), "");
  return lines.join("\n");
}

async function main() {
  const { values } = parseArgs({ options: {
    renderer: { type: "string" }, recipes: { type: "string" }, "baseline-recipes": { type: "string" },
    model: { type: "string" }, hardware: { type: "string" }, variant: { type: "string" }, strategy: { type: "string" },
    report: { type: "string" }, "pd-expectations": { type: "string" }, override: { type: "string", multiple: true },
    "warnings-as-errors": { type: "boolean", default: false }, help: { type: "boolean", default: false },
  } });
  if (values.help) { console.log("validate-ascend.mjs --renderer PATH [--recipes PATH] [--baseline-recipes PATH] [--report FILE.json] [--model org/model] [--hardware ID] [--variant KEY] [--strategy KEY] [--pd-expectations FILE] [--override source=directory] [--warnings-as-errors]"); return; }
  if (!values.renderer) throw new Error("--renderer is required");
  if (values.report && !/\.json$/i.test(values.report)) throw new Error("--report must end in .json");
  let report;
  try { report = await run(values); }
  catch (error) {
    if (values.report) {
      const file = path.resolve(values.report);
      fs.mkdirSync(path.dirname(file), { recursive: true });
      fs.writeFileSync(file, JSON.stringify({ version: 1, status: "tool-error", error: error.message }, null, 2) + "\n");
      fs.writeFileSync(file.replace(/\.json$/i, ".md"), `# Ascend recipe check failed\n\nInput/tool error: ${safeCell(error.message)}\n`);
    }
    throw error;
  }
  const markdown = markdownReport(report);
  if (values.report) {
    const file = path.resolve(values.report);
    fs.mkdirSync(path.dirname(file), { recursive: true });
    fs.writeFileSync(file, JSON.stringify(report, null, 2) + "\n");
    fs.writeFileSync(file.replace(/\.json$/i, ".md"), markdown);
  }
  if (process.env.GITHUB_STEP_SUMMARY) fs.appendFileSync(process.env.GITHUB_STEP_SUMMARY, markdownReport(report, 40));
  for (const finding of report.comparison.additions) {
    console.log(`[${finding.severity.toUpperCase()} ${finding.code}] ${finding.model} / ${finding.hardware || "source"} / ${finding.variant || "—"} / ${finding.strategy || "—"} / ${finding.role || "—"}: ${finding.message}`);
    console.log(`  Fix: ${finding.fix}`);
  }
  console.log(`SUMMARY ${JSON.stringify(report.summary)}`);
  process.exitCode = report.summary.blocking ? 1 : 0;
}
if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  main().catch((error) => { console.error(`[INPUT/TOOL ERROR] ${error.message}`); process.exitCode = 2; });
}
