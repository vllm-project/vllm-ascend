// Temporary migration check. Run from the pinned recipes renderer checkout.
// Remove this adapter once the official channel reads the Ascend source.
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import fs from "node:fs";
import { createRequire } from "node:module";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

const renderer = process.cwd();
const source = path.dirname(fileURLToPath(import.meta.url));
const yaml = createRequire(path.join(renderer, "package.json"))("js-yaml");
const readYaml = (file) => yaml.load(fs.readFileSync(file, "utf8"));
const temporary = fs.mkdtempSync(path.join(os.tmpdir(), "ascend-recipes-"));
const channel = path.basename(temporary);
const assembled = path.join(renderer, ".content", channel);
const split = path.join(temporary, "split");

// A local override or another active channel must not change the baseline.
const env = { ...process.env, CI: "true" };
for (const key of ["RECIPES_OVERRIDES", "RECIPES_CHANNEL", "RECIPES_CONTENT_DIR", "RECIPES_PUBLIC_DIR"]) {
  delete env[key];
}
env.RECIPES_SOURCES = path.join(temporary, "sources.yaml");

function run(script, ...args) {
  execFileSync(process.execPath, [path.join(renderer, "scripts", script), ...args], {
    cwd: renderer, env, stdio: "inherit",
  });
}

function yamlFiles(dir, prefix = "") {
  return fs.readdirSync(dir, { withFileTypes: true }).flatMap((entry) => {
    const relative = path.join(prefix, entry.name);
    if (entry.isDirectory()) return yamlFiles(path.join(dir, entry.name), relative);
    return /\.ya?ml$/.test(entry.name) ? [relative] : [];
  }).sort();
}

try {
  const config = readYaml(path.join(renderer, "sources.yaml"));
  const upstreamChannel = config.default_channel;
  const upstreamSources = structuredClone(config.channels[upstreamChannel]);
  for (const spec of Object.values(upstreamSources)) {
    if (spec.local) spec.local = path.resolve(renderer, spec.local);
  }
  const lock = JSON.parse(fs.readFileSync(path.join(renderer, "sources.lock.json"), "utf8"));
  fs.writeFileSync(env.RECIPES_SOURCES, yaml.dump({
    default_channel: channel, channels: { [channel]: upstreamSources },
  }));
  fs.writeFileSync(path.join(temporary, "sources.lock.json"), JSON.stringify({
    [channel]: lock[upstreamChannel],
  }));

  run("sync-sources.mjs");
  run("assemble-content.mjs", "--strict");
  run("split-overlay.mjs", "--brands", "Huawei", "--source", "vllm-ascend",
    "--guide-heading", "Ascend", "--models", path.join(assembled, "models"), "--out", split);

  // During bootstrap, only content already merged in the pinned sources is
  // admitted. This also detects an accidentally omitted recipe or parameter.
  const expected = path.join(split, "vllm-ascend", "overlays");
  const actual = path.join(source, "overlays");
  const models = path.join(source, "models");
  assert.ok(!fs.existsSync(models) || yamlFiles(models).length === 0,
    "Complete recipes are not part of this merged-overlay bootstrap");
  const files = yamlFiles(expected);
  assert.deepEqual(yamlFiles(actual), files, "Imported overlay file set differs from the merged sources");
  for (const file of files) {
    assert.deepEqual(readYaml(path.join(actual, file)), readYaml(path.join(expected, file)),
      `${file}: content differs from the merged source`);
  }
  console.log(`Verified ${files.length} overlays against the merged source snapshots.`);

  fs.writeFileSync(env.RECIPES_SOURCES, yaml.dump({
    default_channel: channel,
    channels: { [channel]: {
      baseline: { local: path.join(split, "core"), role: "core" },
      "vllm-ascend": { local: source, role: "overlay", scope: { brands: ["Huawei"] } },
    } },
  }));
  fs.writeFileSync(path.join(temporary, "sources.lock.json"), "{}");
  // Use the same strict assembly/schema steps as validate-source.mjs, but
  // retain this run's unique channel instead of its fixed "validate" output.
  run("assemble-content.mjs", "--strict");
  env.RECIPES_CONTENT_DIR = assembled;
  env.RECIPES_PUBLIC_DIR = path.join(temporary, "public");
  run("build-recipes-api.mjs");
} finally {
  // Both paths are created by this run; never remove a caller-supplied path.
  assert.equal(path.dirname(temporary), path.resolve(os.tmpdir()));
  assert.equal(path.dirname(assembled), path.join(renderer, ".content"));
  fs.rmSync(temporary, { recursive: true, force: true });
  fs.rmSync(assembled, { recursive: true, force: true });
}
