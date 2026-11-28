// Source-owned v0.32.0rc tutorial baselines and read-only configuration checks.
import assert from "node:assert/strict";
import { execFileSync, spawnSync } from "node:child_process";
import { mkdirSync, mkdtempSync, readdirSync, readFileSync, rmSync, statSync, writeFileSync } from "node:fs";
import { createRequire } from "node:module";
import { tmpdir } from "node:os";
import path from "node:path";
import { test } from "node:test";
import { pathToFileURL } from "node:url";
import { parseArgs } from "node:util";

const { values } = parseArgs({ options: { renderer: { type: "string" }, report: { type: "string" } } });
assert.ok(values.renderer, "Pass --renderer PATH.");
const renderer = path.resolve(values.renderer);
const { resolvePoolDeployment, validatePoolDeployment, validatePoolProfile } = await import(
  pathToFileURL(path.join(renderer, "src/lib/pool-deployment.js")).href
);
const yaml = createRequire(path.join(renderer, "package.json"))("js-yaml");
const read = (name) => yaml.load(readFileSync(new URL(name, import.meta.url), "utf8"));
const taxonomy = yaml.load(readFileSync(path.join(renderer, "taxonomy.yaml"), "utf8"));
const minimax = read("../overlays/MiniMaxAI/MiniMax-M3.1-Flash.yaml");
const qwen = read("../overlays/Qwen/Qwen3.8-Flash-Next.yaml");
const cases = [
  [minimax, "ascend_a3_mooncake_pool", 3, 16, 19100, [19200,19201,19202,19203,19300,19301,19302,19303], 19080],
  [minimax, "ascend_950dt_mooncake_pool", 3, 8, 31050, [31060,31060], 19080],
  [qwen, "ascend_a2_memcache_pool", 2, 8, 25568, [25569], 25570],
  [qwen, "ascend_a3_memcache_pool", 1, 16, 25568, [25569], 25570],
  [qwen, "ascend_950dt_memcache_pool", 1, 8, 8247, [8248], 8249],
];
const flag = (argv, key) => argv[argv.indexOf(key) + 1];
const modelDir = "/data/site's model weights";
const deviceFor = (counter) => "/dev/nvme" + counter + "n1";
const addresses = { p: "192.0.2.1", d: "192.0.2.2", d1: "192.0.2.11", d2: "192.0.2.12" };

for (const [overlay, key, hostCount, deviceCount, pPort, dPorts, proxyPort] of cases) {
  for (const mode of ["new", "reuse"]) {
    test(key + " " + mode, () => {
      const preset = overlay.presets[key];
      const backend = overlay === minimax ? "mooncake" : "memcache";
      const profile = read("../pool_profiles/" + backend + ".yaml");
      const recipe = { ...overlay, _poolProfiles: { ["vllm-ascend/" + backend]: profile } };
      const hardware = taxonomy.hardware_profiles[preset.hardware];
      assert.deepEqual(validatePoolProfile(profile), []);
      assert.deepEqual(validatePoolDeployment(profile, preset.deployment, hardware), []);
      const inputs = {
        master_address: addresses.p, meta_address: addresses.p, config_address: addresses.p, metrics_address: addresses.p,
        meta_port: 15000, config_port: 16000, metrics_port: 18000, force_new_disk: false,
        // Fixed filenames belong to the preset and cannot be redirected by supplied values.
        prefill_config: "other-local.conf", decode_config: "other-local.conf",
      };
      for (const host of preset.deployment.hosts) {
        const local = { address: addresses[host], nic: "eth1", weights_host_dir: modelDir, config_host_dir: "/cfg/" + host,
          ssd_host_dir: "/ssd/" + host, device: deviceFor(1), device_2: deviceFor(2) };
        for (const [field, value] of Object.entries(local)) inputs["hosts." + host + "." + field] = value;
      }
      const plan = resolvePoolDeployment({ recipe, presetKey: key, mode, inputs });
      assert.deepEqual(plan.missing, []);
      assert.deepEqual(plan.errors, []);
      assert.equal(plan.hosts.length, hostCount);
      assert.equal(plan.steps.filter((s) => s.id.startsWith("container/")).length, hostCount);
      const positions = new Map(plan.steps.map((s, i) => [s.id, i]));
      for (const step of plan.steps) {
        assert.ok(step.depends_on.every((id) => positions.has(id) && positions.get(id) < positions.get(step.id)), step.id);
        assert.ok(!step.command.includes("{{"), step.id);
      }
      for (const step of plan.steps.filter((s) => s.id.startsWith("container/"))) {
        assert.ok(step.argv.includes(modelDir + ":/models/checkpoint:ro"), step.command);
      }
      assert.equal(plan.steps.filter((s) => /^start-(master|meta)\//.test(s.id)).length, mode === "new" ? 1 : 0);
      if (mode === "reuse") {
        assert.ok(!plan.steps.some((s) => /^new-disk|^second-disk|^configure-meta|^write-config|^configure-storage|^copy-templates/.test(s.id)));
      }
      assert.ok(plan.steps.some((s) => s.id.startsWith("shared-prefix/") && s.command.includes("/v1/chat/completions")));
      assert.ok(plan.steps.some((s) => s.id.startsWith("ssd-restore/") && s.check.includes("SSD read/restore")));
      const [prefill] = plan.launches.filter((p) => p.role === "prefill");
      const decode = plan.launches.filter((p) => p.role === "decode");
      const proxy = plan.launches.find((p) => p.role === "proxy");
      assert.equal(flag(prefill.argv, "--port"), String(pPort));
      assert.deepEqual(decode.map((p) => Number(flag(p.argv, "--port"))), dPorts);
      assert.equal(flag(proxy.argv, "--port"), String(proxyPort));
      const decoderPorts = proxy.argv.indexOf("--decoder-ports") + 1;
      assert.deepEqual(proxy.argv.slice(decoderPorts, decoderPorts + dPorts.length).map(Number), dPorts);
      for (const w of [prefill, ...decode]) {
        assert.equal(w.argv.filter((a) => a === "--kv-transfer-config").length, 1);
        assert.equal(w.env.VLLM_USE_V2_MODEL_RUNNER, "1");
        assert.equal(w.env.PYTHONHASHSEED, "0");
        assert.deepEqual(w.devices, w.env.ASCEND_RT_VISIBLE_DEVICES.split(",").map(Number));
        const config = JSON.parse(flag(w.argv, "--kv-transfer-config"));
        const role = w.role === "prefill" ? "kv_producer" : "kv_consumer";
        const [direct, cache] = config.kv_connector_extra_config.connectors;
        assert.deepEqual([config.kv_role, direct.kv_role, cache.kv_role], [role, role, role]);
        assert.equal(direct.kv_connector, backend === "mooncake" ? "MooncakeConnectorV1" : "MooncakeConnectorV2");
        assert.equal(cache.kv_connector, "AscendStoreConnector");
        assert.equal(cache.kv_connector_extra_config.backend, backend);
      }
      for (const host of plan.hosts) {
        const ids = [prefill, ...decode].filter((p) => p.host === host).flatMap((p) => p.devices);
        assert.equal(ids.length, deviceCount);
        assert.equal(new Set(ids).size, deviceCount);
      }
      if (backend === "memcache") {
        const a5 = key === "ascend_950dt_memcache_pool";
        const localFiles = a5 ? ["mmc-local-p.conf", "mmc-local-d.conf"] : ["mmc-local.conf", "mmc-local.conf"];
        assert.deepEqual([prefill, decode[0]].map((w) => w.env.MMC_LOCAL_CONFIG_PATH),
          localFiles.map((file) => "/opt/recipe-pool/" + file));
        assert.ok(plan.inputs.every((field) => !/^(local_configs|prefill_config|decode_config)$/.test(field.key.split(".").at(-1))));
        if (mode === "reuse") assert.ok(!plan.inputs.some((field) => field.key.endsWith(".local_services")));
        const steps = mode === "new" ? ["copy-templates/", "configure-storage/", "validate-local/"] :
          ["validate-local/", "reuse-disk-policy/"];
        const guards = plan.steps.filter((step) => step.id.startsWith("reuse-disk-policy/"));
        assert.equal(guards.length, mode === "reuse" ? hostCount : 0);
        function prerequisites(id) {
          const direct = plan.steps[positions.get(id)].depends_on;
          return new Set(direct.flatMap((dep) => [dep, ...prerequisites(dep)]));
        }
        for (const worker of [prefill, ...decode]) {
          const beforeStart = prerequisites("start-" + worker.role + "/" + worker.id);
          for (const guard of guards) assert.ok(beforeStart.has(guard.id), worker.id + " requires " + guard.id);
        }
        for (const prefix of steps) {
          for (const host of plan.hosts) {
            const command = plan.steps.find((step) => step.id === prefix + host).command;
            const argumentsLine = command.split("\n").find((line) => /^python3? - /.test(line));
            assert.ok(argumentsLine.includes(localFiles.join(" ")), argumentsLine);
            assert.ok(!argumentsLine.includes("other-local.conf"), argumentsLine);
          }
        }
      }
      if (key === "ascend_a3_mooncake_pool") {
        assert.deepEqual(decode.map((p) => flag(p.argv, "--data-parallel-rank")), ["0","1","2","3","4","5","6","7"]);
        assert.ok(plan.steps.find((s) => s.id === "start-d4/decode4").depends_on.includes("d1-coordinator/decode0"));
      }
      if (key === "ascend_a3_memcache_pool") {
        assert.equal(flag(prefill.argv, "--tensor-parallel-size"), "2");
        assert.equal(flag(prefill.argv, "--pipeline-parallel-size"), "2");
        assert.equal(flag(prefill.argv, "--data-parallel-size"), "2");
        assert.ok(plan.steps.find((s) => s.id === "container/p").argv.includes("/dev/nvme2n1:/dev/recipe-cache1"));
      }
      if (key === "ascend_950dt_memcache_pool") {
        assert.equal(prefill.env.MMC_LOCAL_CONFIG_PATH, "/opt/recipe-pool/mmc-local-p.conf");
        assert.equal(decode[0].env.MMC_LOCAL_CONFIG_PATH, "/opt/recipe-pool/mmc-local-d.conf");
        assert.ok(plan.steps.find((s) => s.id === "start-decode/decode").depends_on.includes("prefill-ready/prefill"));
      }
    });
  }
}

const diskPolicy = "ubsio.standalone.force_new_disk";
const approved = "# " + diskPolicy + " = true\n; " + diskPolicy + " = true\n\t" + diskPolicy + " = false  \n";
const unsafePolicies = [
  null,                                  // The role configuration file is absent.
  "# " + diskPolicy + " = false\n",      // Only a comment, so the package default applies.
  ...["true", "", "auto", "0", "false # unverified"].map((value) => diskPolicy + " = " + value + "\n"),
  diskPolicy + " = true\n" + approved,    // A dictionary parser would hide the unsafe first entry.
  approved + diskPolicy + " = true\n",
  approved + diskPolicy + " = false\n",
  approved + diskPolicy + ": true\n",
];

for (const key of ["ascend_a2_memcache_pool", "ascend_a3_memcache_pool", "ascend_950dt_memcache_pool"]) {
  test(key + " executes the read-only reuse guard on the files each worker opens", (t) => {
    const root = mkdtempSync(path.join(tmpdir(), "memcache reuse "));
    t.after(() => rmSync(root, { recursive: true, force: true }));
    // Resolve the fixture's path in the Bash environment that runs the generated command.
    const shellRoot = execFileSync("bash", ["-c", "pwd -P"], { cwd: root, encoding: "utf8" }).trim();
    const inputs = {};
    for (const host of qwen.presets[key].deployment.hosts) {
      mkdirSync(path.join(root, host));
      inputs["hosts." + host + ".config_dir"] = shellRoot + "/" + host;
    }
    const recipe = { ...qwen, _poolProfiles: { "vllm-ascend/memcache": read("../pool_profiles/memcache.yaml") } };
    const plan = resolvePoolDeployment({ recipe, presetKey: key, mode: "reuse", inputs });
    const guards = plan.steps.filter((step) => step.id.startsWith("reuse-disk-policy/"));
    const containers = new Set(plan.launches.filter((w) => ["prefill", "decode"].includes(w.role)).map((w) => w.container));
    assert.deepEqual(guards.map((step) => step.container).sort(), [...containers].sort());
    for (const guard of guards) {
      const dir = path.join(root, guard.host);
      const workers = plan.launches.filter((w) => w.container === guard.container && ["prefill", "decode"].includes(w.role));
      const files = [...new Set(workers.map((w) => path.posix.basename(w.env.MMC_LOCAL_CONFIG_PATH)))];
      assert.ok(files.length > 0);
      const snapshot = () => readdirSync(dir).sort().map((name) => {
        const file = path.join(dir, name);
        return [name, readFileSync(file), statSync(file).mtimeMs];
      });
      function check(status, file) {
        const before = snapshot();
        const run = spawnSync("bash", ["-c", guard.command], { encoding: "utf8" });
        assert.ifError(run.error);
        assert.equal(run.status, status, guard.id + ": " + run.stderr);
        if (file) assert.ok(run.stderr.includes(file), run.stderr);
        assert.deepEqual(snapshot(), before, "The guard changed the configuration on " + guard.host);
      }
      for (const name of files) writeFileSync(path.join(dir, name), approved);
      check(0);
      for (const name of files) {
        const file = path.join(dir, name);
        for (const policy of unsafePolicies) {
          if (policy === null) rmSync(file);
          else writeFileSync(file, policy);
          check(1, name);
        }
        writeFileSync(file, approved);
      }
    }
  });
}

if (values.report) {
  test("the assembled source report checked every published pool process", () => {
    const report = JSON.parse(readFileSync(values.report, "utf8"));
    for (const [overlay, key, , , pPort, dPorts, proxyPort] of cases) {
      const rows = report.results.filter((row) => row.preset === key);
      assert.equal(rows.length, 1, key);
      const [row] = rows;
      const backend = overlay === minimax ? "mooncake" : "memcache";
      assert.equal(row.pool?.backend, backend, key);
      assert.equal(row.pool.profile, "vllm-ascend/" + backend);
      assert.equal(row.router, "ascend-proxy");
      assert.equal(flag(row.pool.proxy.argv, "--port"), String(proxyPort));
      assert.deepEqual(row.workers.map((w) => Number(flag(w.argv, "--port"))), [pPort, ...dPorts]);
      assert.equal(new Set(row.workers.map((w) => w.placement)).size, row.workers.length);
      for (const worker of row.workers) {
        assert.ok(worker.placement);
        const kv = JSON.parse(flag(worker.argv, "--kv-transfer-config"));
        assert.equal(kv.kv_connector, "MultiConnector");
        assert.equal(kv.kv_connector_extra_config.connectors[1].kv_connector_extra_config.backend, backend);
      }
      assert.ok(!row.findings.some((f) => f.severity === "error"), key);
    }
  });
}
