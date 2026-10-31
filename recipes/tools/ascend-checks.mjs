// CPU-only semantic checks. Never executes recipe shell commands.
import { createHash } from "node:crypto";

// Guard the JSX-only installation resolver until the renderer exports one.
// Other renderer changes are allowed; changes to these semantics need review.
export function assertInstallationAdapter(source) {
  const text = source.replaceAll("\r\n", "\n");
  const block = text.slice(text.indexOf("function InstallBlock("));
  const sections = [
    [text, "  const hwInstall =", "  const effectiveInstallMode =", "668c2e58b17c2788abe2a9ef6d5bd7ea0446fa068dbd6ab21c5bc2937c61ea01"],
    [block, "  const install = recipe.model?.install || {};", "  const [open, setOpen]", "d04be1258218b6ff81e5885e906aa72d65ad5b0d3a56570389e2895d30006ba9"],
    [block, "  const nightlyRequired =", "  // A CUDA map", "e375afe4c095df6ec378926d3bd797bf60da7b046f70d0bf8b37e51b2947c04c"],
    [block, "  const pipCudaTag =", "  const pipNote =", "a51f62ae340d3cb373665bd720cb4fc307263faa20dc75259a787cd9a2defb20"],
    [block, "  const defaultDockerCmd =", "  const defaultDockerNote =", "9b74e89c78484d0a82215ccf24c5b2e32f552d4f1047c40e327aae23c09b566a"],
  ];
  for (const [body, start, end, expected] of sections) {
    const first = body.indexOf(start);
    const last = body.indexOf(end, first);
    const digest = first < 0 || last < first ? null
      : createHash("sha256").update(body.slice(first, last).trim()).digest("hex");
    if (digest !== expected) throw new Error(`Renderer installation semantics changed at ${start.trim()}; review the adapter before updating its fingerprint.`);
  }
}
export function installationFor(recipe, variantKey, hardwareId, meta) {
  const variant = recipe.variants?.[variantKey] || {};
  const install = recipe.model?.install || {};
  const hwInstall = variant.hardware_overrides?.[hardwareId]?.install
    || recipe.hardware_overrides?.[hardwareId]?.install || {};
  const exact = variant.hardware_overrides?.[hardwareId]?.docker_image;
  let imageSource = "renderer-default";
  if (typeof exact === "string" && exact) {
    imageSource = `variants.${variantKey}.hardware_overrides.${hardwareId}.docker_image`;
  } else {
    for (const [prefix, value] of [[`variants.${variantKey}.docker_image`, variant.docker_image],
      ["model.docker_image", recipe.model?.docker_image]]) {
      if (typeof value === "string" && value) { imageSource = prefix; break; }
      if (value?.nvidia) { imageSource = `${prefix}.nvidia`; break; }
      if (value?.cu130 || value?.cu129) { imageSource = `${prefix}.cuda_map`; break; }
    }
  }
  return {
    dockerEnabled: install.docker !== false,
    pipEnabled: install.pip !== false && hwInstall.pip !== false,
    pipCommand: install.pip?.command || null,
    defaultPip: !install.pip?.command,
    nightly: recipe.model?.nightly_required === true || variant.nightly_required === true,
    image: meta.image,
    imageSource,
    dockerCommand: install.docker?.command || null,
  };
}

function flagValue(argv, name) {
  let found;
  const aliases = { "-tp": "--tensor-parallel-size", "-dp": "--data-parallel-size", "-pp": "--pipeline-parallel-size" };
  for (let i = 0; i < argv.length; i++) {
    const equal = argv[i].indexOf("=");
    const rawKey = equal < 0 ? argv[i] : argv[i].slice(0, equal);
    const key = rawKey.startsWith("--") ? rawKey.replaceAll("_", "-") : aliases[rawKey] || rawKey;
    if (key === name) found = equal < 0 ? argv[i + 1] : argv[i].slice(equal + 1);
  }
  return found;
}

function staticDevices(value) {
  if (value == null) return null;
  const text = String(value).trim();
  if (/[$`]/.test(text)) return null;
  if (!/^\d+(?:\s*,\s*\d+)*$/.test(text)) return [];
  return text.split(",").map(Number);
}

export function checkScenario(scenario) {
  const { installation: install, workers = [], profile, deployType, expectedPD, router } = scenario;
  const findings = [];
  const add = (code, message, fix, severity = "error", role = null) => {
    const worker = workers.find((w) => w.role === role);
    const argv = worker?.argv || [];
    const env = worker?.env || {};
    const evidence = code === "ASC001" ? [install.image, install.imageSource]
      : ["ASC002", "ASC003"].includes(code) ? [install.pipCommand, install.defaultPip, install.nightly]
        : code === "ASC004" ? [install.dockerEnabled, install.pipEnabled]
          : code === "ASC005" ? install.dockerCommand
            : code.startsWith("ASC01") ? [expectedPD || null, router,
              worker ? flagValue(argv, "--kv-transfer-config") : workers.map((w) => flagValue(w.argv, "--kv-transfer-config"))]
              : ["ASC020", "ASC021", "ASC022", "ASC023"].includes(code) ? [env.ASCEND_RT_VISIBLE_DEVICES, env.CUDA_VISIBLE_DEVICES, profile.gpu_count]
                : ["ASC024", "ASC025"].includes(code) ? [flagValue(argv, "--tensor-parallel-size"), flagValue(argv, "--data-parallel-size"),
                  flagValue(argv, "--pipeline-parallel-size"), env.ASCEND_RT_VISIBLE_DEVICES, profile.gpu_count]
                  : code === "ASC030" ? [flagValue(argv, "--attention-backend"), flagValue(argv, "--moe-backend"), env.VLLM_USE_FLASHINFER_MOE_FP8] : null;
    findings.push({ code, severity, role, message, fix, evidence });
  };
  if (!install.dockerEnabled && !install.pipEnabled) {
    add("ASC004", "Both Docker and pip installation are disabled.", "Enable one supported installation method.");
  }
  if (install.dockerEnabled) {
    if (!install.image?.trim() || install.imageSource === "renderer-default"
      || /\.nvidia$|\.cuda_map$/.test(install.imageSource)
      || /(?:^|\/)vllm\/vllm-openai(?=[:@]|$)|nvcr\.io\/nvidia\//.test(install.image)) {
      add("ASC001", `NPU image resolves to ${install.image || "<empty>"} from ${install.imageSource}.`,
        "Pin an Ascend image in the variant's exact hardware override, or disable Docker for a pip-only recipe.");
    }
  }
  const genericExplicitPip = /--torch-backend(?:=|\s+)auto\b/.test(install.pipCommand || "")
    && !/vllm[-_]ascend|torch[-_]npu/.test(install.pipCommand || "");
  if (install.pipEnabled && (install.defaultPip || genericExplicitPip)) {
    add("ASC002", install.nightly
      ? "NPU pip tab falls back to the CUDA nightly wheel index."
      : "NPU pip tab falls back to generic vLLM --torch-backend auto installation.",
    "Set install.pip: false on the exact hardware, or supply an effective Ascend main pip command. Dependencies do not override it.");
  }
  if (install.pipEnabled && /(?:wheels\.vllm\.ai|download\.pytorch\.org)[^\s]*\/cu\d+|--torch-backend(?:=|\s+)cu\d+/.test(install.pipCommand || "")) {
    add("ASC003", "The effective pip command selects CUDA wheels.", "Use an Ascend installation flow or disable the pip tab.");
  }
  if (install.dockerEnabled && /\bdocker\s+pull\s+(?:vllm\/vllm-openai(?=[:@\s]|$)|nvcr\.io\/nvidia\/)/.test(install.dockerCommand || "")) {
    add("ASC005", "The custom Docker installation instruction pulls a known NVIDIA image despite the NPU selection.",
      "Override the shared installation instruction or use the default pull of the selected Ascend image.");
  }
  if (deployType === "pd_cluster") {
    if (!expectedPD?.connector || !expectedPD?.router) {
      add("ASC010", "PD connector/router has no explicit static expectation; renderer defaults are not Ascend validation.",
        "Review the tutorial and record the expected connector/router with --pd-expectations; this is not an NPU pass record.");
    } else if (router !== expectedPD.router) {
      add("ASC013", `PD router is ${router}; expected ${expectedPD.router}.`, "Fix the recipe or the upstream renderer; do not change expectations merely to silence the check.");
    }
    for (const worker of workers) {
      let config;
      try { config = JSON.parse(flagValue(worker.argv, "--kv-transfer-config")); }
      catch { add("ASC012", "Missing or invalid --kv-transfer-config JSON.", "Provide a valid PD KV transfer configuration.", "error", worker.role); continue; }
      if (!config || typeof config !== "object" || Array.isArray(config) || typeof config.kv_connector !== "string") {
        add("ASC012", "KV transfer config must be a non-null object with a connector name.", "Provide a valid PD KV transfer configuration.", "error", worker.role); continue;
      }
      if (expectedPD?.connector && config.kv_connector !== expectedPD.connector) {
        add("ASC011", `PD connector is ${config.kv_connector}; expected ${expectedPD.connector}.`, "Check the generated configuration against the Ascend tutorial.", "error", worker.role);
      }
      const expectedRole = worker.role === "prefill" ? "kv_producer" : "kv_consumer";
      if (![expectedRole, "kv_both"].includes(config.kv_role)) {
        add("ASC014", `Invalid ${worker.role} KV role: ${config.kv_role}.`, `Use ${expectedRole} or the connector's documented kv_both mode.`, "error", worker.role);
      }
    }
  }
  for (const worker of workers) {
    const argv = worker.argv || [];
    const env = worker.env || {};
    const ascend = staticDevices(env.ASCEND_RT_VISIBLE_DEVICES);
    const cuda = staticDevices(env.CUDA_VISIBLE_DEVICES);
    if (deployType === "pd_cluster" && env.CUDA_VISIBLE_DEVICES != null && env.ASCEND_RT_VISIBLE_DEVICES == null) {
      add("ASC020", "PD pins CUDA devices without an Ascend device selection.", "Check the NPU role allocation and renderer platform branch.", "error", worker.role);
    }
    if (ascend && cuda && ascend.join(",") !== cuda.join(",")) {
      add("ASC021", "Ascend and CUDA device lists disagree.", "Review the generated PD device allocation; do not assume CUDA pinning selects NPUs.", "error", worker.role);
    }
    if (env.ASCEND_RT_VISIBLE_DEVICES != null && ascend === null) {
      add("ASC023", "Ascend device selection is symbolic; IDs and capacity were not statically checked.", "Validate resolved device IDs on the target host.", "warning", worker.role);
    }
    if (ascend && (!ascend.length || new Set(ascend).size !== ascend.length
      || ascend.some((id) => id >= profile.gpu_count))) {
      add("ASC022", `Invalid, duplicated or out-of-range NPU list: ${env.ASCEND_RT_VISIBLE_DEVICES}.`, "Use distinct node-local NPU IDs within the selected hardware profile.", "error", worker.role);
    }
    if (deployType === "single_node") {
      const sizes = ["--tensor-parallel-size", "--data-parallel-size", "--pipeline-parallel-size"]
        .map((flag) => [flag, flagValue(argv, flag) ?? "1"]);
      const symbolic = (value) => /[$`]/.test(String(value));
      const unresolved = sizes.some(([, value]) => symbolic(value));
      const invalid = sizes.filter(([, value]) => !symbolic(value)).some(([, value]) =>
        !/^\+?\d+$/.test(String(value).trim()) || !Number.isSafeInteger(Number(value)) || Number(value) <= 0);
      if (invalid) add("ASC025", "Parallelism contains an invalid positive decimal integer.",
        "Resolve TP, DP and PP to positive decimal integers before deployment.", "error", worker.role);
      if (unresolved) add("ASC025", "Parallelism is symbolic and not statically checked.",
        "Resolve TP, DP and PP before deployment.", "warning", worker.role);
      const needed = sizes.reduce((product, [, value]) => product * Number(value), 1);
      const available = ascend?.length ?? profile.gpu_count;
      if (Number.isFinite(needed) && needed > available) {
        add("ASC024", `Single-node TP×DP×PP needs ${needed} NPUs; ${available} selected.`, "Align parallel sizes and visible NPUs.", "error", worker.role);
      }
    }
    const backend = flagValue(argv, "--attention-backend");
    const moeBackend = flagValue(argv, "--moe-backend");
    if (backend?.toUpperCase().includes("FLASHINFER")
      || moeBackend?.toUpperCase().includes("FLASHINFER")
      || /^(1|true|yes)$/i.test(String(env.VLLM_USE_FLASHINFER_MOE_FP8 || ""))) {
      add("ASC030", "Generated NPU command retains a FLASHINFER backend setting.", "Move GPU-only settings out of shared base configuration or override them through a supported layer.", "error", worker.role);
    }
  }
  return findings;
}
