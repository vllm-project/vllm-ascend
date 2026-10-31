import assert from "node:assert/strict";
import { test } from "node:test";
import { checkScenario, installationFor } from "./ascend-checks.mjs";
import { compareFindings } from "./validate-ascend.mjs";

const profile = { brand: "Huawei", generation: "npu", gpu_count: 8 };
const recipe = { model: {}, variants: { ascend: { hardware_overrides: {
  a5: { docker_image: "registry.example/team/custom:1", install: { pip: false } },
} } } };
function scenario(patch = {}) {
  return { profile, deployType: "single_node", installation: {
    dockerEnabled: true, pipEnabled: false, image: "registry.example/team/custom:1",
    imageSource: "variants.ascend.hardware_overrides.a5.docker_image",
  }, workers: [{ role: "server", argv: ["vllm", "serve", "org/model", "--tensor-parallel-size", "4"], env: {} }], ...patch };
}
const codes = (value) => checkScenario(value).map((x) => x.code);

test("custom NPU image does not require an ascend repository or a5 suffix", () => {
  assert.deepEqual(codes(scenario()), []);
});
test("missing image is an error even when generation produced a default image", () => {
  assert.ok(codes(scenario({ installation: { dockerEnabled: true, pipEnabled: false,
    image: "vllm/vllm-openai:latest", imageSource: "renderer-default" } })).includes("ASC001"));
});
test("explicit NVIDIA parent image inheritance is rejected", () => {
  assert.ok(codes(scenario({ installation: { dockerEnabled: true, pipEnabled: false,
    image: "custom/nvidia-build:v1", imageSource: "model.docker_image.nvidia" } })).includes("ASC001"));
});
test("pip-only configuration does not require a docker image", () => {
  assert.deepEqual(codes(scenario({ installation: { dockerEnabled: false, pipEnabled: true,
    pipCommand: "pip install ./vllm-ascend", defaultPip: false } })), []);
});
test("missing pip override is rejected independently from a valid image", () => {
  assert.ok(codes(scenario({ installation: { dockerEnabled: true, pipEnabled: true,
    defaultPip: true, image: "custom/ascend:1", imageSource: "explicit" } })).includes("ASC002"));
});
test("explicit CUDA wheel index is rejected", () => {
  assert.ok(codes(scenario({ installation: { dockerEnabled: false, pipEnabled: true,
    defaultPip: false, pipCommand: "pip install vllm --index-url https://wheels.vllm.ai/nightly/cu130" } })).includes("ASC003"));
});
test("disabling both installation methods is rejected", () => {
  assert.ok(codes(scenario({ installation: { dockerEnabled: false, pipEnabled: false } })).includes("ASC004"));
});
test("exact hardware pip false matches current renderer UI precedence", () => {
  assert.equal(installationFor(recipe, "ascend", "a5", { image: "custom:1" }).pipEnabled, false);
});
test("hardware pip command is not treated as overriding the renderer main pip command", () => {
  const r = structuredClone(recipe);
  r.variants.ascend.hardware_overrides.a5.install = { pip: { command: "pip install vllm-ascend" } };
  assert.equal(installationFor(r, "ascend", "a5", { image: "custom:1" }).defaultPip, true);
});
test("PD without reviewed expectation is not silently accepted as Ascend-ready", () => {
  assert.ok(codes(scenario({ deployType: "pd_cluster" })).includes("ASC010"));
});
test("NIXL may be explicitly expected; it is not NVIDIA-only by name", () => {
  const workers = ["prefill", "decode"].map((role) => ({ role, env: {}, argv: [
    "--kv-transfer-config", JSON.stringify({ kv_connector: "NixlConnector", kv_role: role === "prefill" ? "kv_producer" : "kv_consumer" }),
  ] }));
  assert.deepEqual(codes(scenario({ deployType: "pd_cluster", workers, router: "vllm-router",
    expectedPD: { connector: "NixlConnector", router: "vllm-router" } })), []);
});
test("PD default connector is compared with reviewed expectation", () => {
  const workers = ["prefill", "decode"].map((role) => ({ role, env: {}, argv: [
    "--kv-transfer-config", JSON.stringify({ kv_connector: "NixlConnector", kv_role: "kv_both" }),
  ] }));
  assert.ok(codes(scenario({ deployType: "pd_cluster", workers, router: "vllm-router",
    expectedPD: { connector: "MooncakeConnectorV1", router: "vllm-router" } })).includes("ASC011"));
});
test("invalid KV JSON gives actionable error", () => {
  assert.ok(codes(scenario({ deployType: "pd_cluster", workers: [{ role: "prefill", env: {}, argv: ["--kv-transfer-config", "oops"] }],
    expectedPD: { connector: "MooncakeConnectorV1", router: "vllm-router" }, router: "vllm-router" })).includes("ASC012"));
});
test("PD CUDA-only device pinning is detected", () => {
  assert.ok(codes(scenario({ deployType: "pd_cluster", workers: [{ role: "prefill", argv: [],
    env: { CUDA_VISIBLE_DEVICES: "0,1,2,3" } }] })).includes("ASC020"));
});
test("conflicting static device lists are detected", () => {
  assert.ok(codes(scenario({ workers: [{ role: "server", argv: [], env: {
    CUDA_VISIBLE_DEVICES: "0,1,2,3", ASCEND_RT_VISIBLE_DEVICES: "0,1",
  } }] })).includes("ASC021"));
});
test("symbolic device lists are reported as not statically checked", () => {
  const findings = checkScenario(scenario({ workers: [{ role: "server", argv: [], env: { ASCEND_RT_VISIBLE_DEVICES: "$NPU_IDS" } }] }));
  assert.ok(findings.some((x) => x.code === "ASC023" && x.severity === "warning"));
});
test("too few selected NPUs for single-node TP is rejected", () => {
  assert.ok(codes(scenario({ workers: [{ role: "server", argv: ["--tensor-parallel-size=4"],
    env: { ASCEND_RT_VISIBLE_DEVICES: "0,1" } }] })).includes("ASC024"));
});
test("GPU-compatible flag names and NCCL env are not blacklisted", () => {
  assert.deepEqual(codes(scenario({ workers: [{ role: "server", argv: ["--gpu-memory-utilization", "0.9",
    "--compilation-config", '{"cudagraph_mode":"FULL_DECODE_ONLY"}'], env: { NCCL_SOCKET_IFNAME: "eth0" } }] })), []);
});
test("known incompatible FLASHINFER backend is rejected", () => {
  assert.ok(codes(scenario({ workers: [{ role: "server", argv: ["--attention-backend=FLASHINFER"], env: {} }] })).includes("ASC030"));
});

test("explicit FlashInfer MoE backends are rejected in supported flag forms", () => {
  for (const argv of [["--moe-backend", "flashinfer_trtllm"], ["--moe_backend=flashinfer_cutlass"]]) {
    assert.ok(codes(scenario({ workers: [{ role: "server", argv, env: {} }] })).includes("ASC030"));
  }
  assert.deepEqual(codes(scenario({ workers: [{ role: "server", argv: ["--moe-backend", "triton"], env: {} }] })), []);
});

test("changed FlashInfer MoE backend evidence remains a new blocking error", () => {
  const findings = (backend) => checkScenario(scenario({ workers: [{ role: "server",
    argv: ["--moe-backend", backend], env: {} }] }));
  assert.equal(compareFindings(findings("flashinfer_cutlass"), findings("flashinfer_trtllm")).blocking.length, 1);
});
test("disabled FLASHINFER env does not falsely fail", () => {
  assert.deepEqual(codes(scenario({ workers: [{ role: "server", argv: [], env: { VLLM_USE_FLASHINFER_MOE_FP8: "0" } }] })), []);
});
test("an inherited Docker install command cannot pull a known NVIDIA image unnoticed", () => {
  const value = scenario();
  value.installation.dockerCommand = "docker pull vllm/vllm-openai:latest";
  assert.ok(codes(value).includes("ASC005"));
});
test("an explicit generic pip command is not mistaken for an Ascend override", () => {
  const value = scenario({ installation: { dockerEnabled: false, pipEnabled: true,
    defaultPip: false, pipCommand: "uv pip install -U vllm --torch-backend auto" } });
  assert.ok(codes(value).includes("ASC002"));
});
test("last accepted parallelism alias takes precedence over earlier long flags", () => {
  const value = scenario({ workers: [{ role: "server", argv: [
    "--tensor-parallel-size", "4", "-tp", "32",
  ], env: {} }] });
  assert.ok(codes(value).includes("ASC024"));
});
test("null, arrays and scalar KV JSON are configuration errors rather than tool exceptions", () => {
  for (const json of ["null", "[]", "1", '"connector"']) {
    const value = scenario({ deployType: "pd_cluster", workers: [{ role: "prefill", argv: ["--kv-transfer-config", json], env: {} }] });
    assert.ok(codes(value).includes("ASC012"), json);
  }
});
test("nonpositive or unresolved parallelism does not silently pass the capacity check", () => {
  const worker = (value) => [{ role: "server", argv: ["--tensor-parallel-size", value], env: {} }];
  assert.ok(codes(scenario({ workers: worker("0") })).includes("ASC025"));
  const findings = checkScenario(scenario({ workers: worker("$TP") }));
  assert.ok(findings.some((f) => f.code === "ASC025" && f.severity === "warning"));
});

test("baseline comparison keeps existing debt visible without blocking unrelated PR changes", () => {
  const finding = { model: "test/model", hardware: "a5", variant: "ascend", strategy: "single_node_tp",
    code: "ASC001", severity: "error", message: "image A" };
  const compared = compareFindings([finding], [finding]);
  assert.equal(compared.existing.length, 1);
  assert.equal(compared.blocking.length, 0);
});
test("new or changed error evidence blocks while resolved errors disappear", () => {
  const finding = { code: "ASC001", severity: "error", message: "image A" };
  const changed = { ...finding, message: "image B" };
  const compared = compareFindings([changed], [finding]);
  assert.equal(compared.blocking.length, 1);
  assert.equal(compared.resolved.length, 1);
});
test("without a baseline all errors block; warnings can explicitly be made blocking", () => {
  const findings = [{ code: "X", severity: "error", message: "error" }, { code: "Y", severity: "warning", message: "warning" }];
  assert.equal(compareFindings(findings).blocking.length, 1);
  assert.equal(compareFindings(findings, null, true).blocking.length, 2);
});
test("zero-coverage and source invariant errors cannot be grandfathered by a baseline", () => {
  const findings = [{ code: "SOURCE_EMPTY", severity: "error", message: "zero checked" }];
  assert.equal(compareFindings(findings, findings).blocking.length, 1);
});
test("symbolic DP does not downgrade a separately invalid PP size", () => {
  const findings = checkScenario(scenario({ workers: [{ role: "server", env: {},
    argv: ["--pipeline-parallel-size", "0", "--data-parallel-size", "$DP"] }] }));
  assert.ok(findings.some((f) => f.code === "ASC025" && f.severity === "error"));
});
test("parallel sizes follow decimal integer token syntax rather than Number coercion", () => {
  for (const token of ["2.0", "0x2", "2e0"]) {
    assert.ok(codes(scenario({ workers: [{ role: "server", env: {}, argv: ["--pipeline-parallel-size", token] }] })).includes("ASC025"), token);
  }
});
