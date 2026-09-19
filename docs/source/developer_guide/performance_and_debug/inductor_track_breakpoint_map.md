# Inductor Track Breakpoint Map

The **inductor compile-backend track** (`compilation_config.backend='inductor'`, i.e. `-cc.backend inductor`) routes `torch.compile` through vLLM's built-in Inductor adaptor into torch_npu's `triton_experimental` backend. The generated Triton kernels are then compiled by **triton-ascend** and **bishengir-compile** (AscendNPU-IR) down to NPU binaries.

When this track breaks it usually does **not** raise: the engine silently falls back to another track, the backend never activates, or an operator deoptimizes to an extern kernel. The only symptoms are "performance is wrong" or "numbers are wrong". This page is the operating manual for pinpointing where the chain broke: the pipeline boundaries, the log chain, backend-activation verification, artifact reading, a four-question triage tree, and a living list of known breakpoints.

Every code reference below is a **grep anchor** (a log message, a symbol name, or an env-var name), never a bare line number — line numbers drift across versions, `grep -rn "<anchor>" <repo>` does not.

## 0. The Pipeline and Its Boundaries

```text
vllm engine (vllm serve / LLM)
 └─ torch.compile -> @support_torch_compile -> vllm compilation backend
     ├─ inductor track: compilation_config.backend='inductor' (-cc.backend inductor)
     │    -> vllm InductorAdaptor -> torch._inductor.compile_fx
     │       -> torch_npu npu_backend loader -> triton_experimental (lazy activation)
     │          -> codegen: output_code.py (wrapper + embedded Triton kernel source)
     │             -> triton.compile()  ── produces .ttir ──────── boundary A
     │                -> triton-ascend: Triton IR -> Ascend MLIR (.ttadapter) ── boundary B
     │                   -> bishengir-compile (AscendNPU-IR): MLIR -> NPU binary (.npubin)
     │                      -> NPU execution
     └─ legacy tracks: npugraph_ex / fusion_pass (AscendCompiler,
        vllm_ascend/compilation/compiler_interface.py)
```

| Layer | Component | Artifact at its output boundary | Typical failure signal |
|---|---|---|---|
| vLLM compilation | `vllm/compilation/backends.py` | per-compile-range graph handles | track never activates; unexpected cache hits |
| Platform config | `vllm_ascend/platform.py` | derived `compilation_config` + env carriers | mode forced to NONE, silently inert track |
| Dynamo / AOT-autograd | torch | AOT joint graph | graph break, `fake tensor` errors (backend-agnostic) |
| Inductor + triton_experimental | `torch/_inductor`, `torch_npu/_inductor/triton_experimental` | `output_code.py`, then `.ttir` (boundary A) | `MissingOperator`, wrong fusion/layout, missing npu import anchor |
| triton-ascend | triton-ascend | `.ttadapter` (boundary B) | `[ConvertTritonIRToLinalgIR]` |
| bishengir-compile | AscendNPU-IR | `.npubin` | `[ConvertLinalgRToBinary]` |
| NPU execution | device | — | wrong numbers / crash / kernel-count anomaly |

Each artifact boundary is also a component boundary for evidence gathering: to attribute a failure, find the last artifact that is still correct and the first signal that is wrong.

## 1. Log Chain of One Cold Start (grep anchors)

| # | Grep anchor (log message) | Printed by | What seeing it means |
|---|---|---|---|
| 1 | `Platform plugin .* is activated` | `vllm/platforms/__init__.py` | out-of-tree platform plugin loaded; `current_platform` is the Ascend one |
| 2 | `NPU does not support compilation mode` | `vllm_ascend/platform.py` | mode outside the supported set was forced to NONE — **the whole compile chain is dead from here on**; no inductor logs will follow. Also check `Compilation disabled, using eager mode` (enforce eager) |
| 3 | `Inductor compile-backend track enabled: backend=inductor` | `vllm_ascend/platform.py` | early platform hook applied the track defaults (also prints the initial cudagraph mode) |
| 4 | `Using InductorAdaptor` / `Using InductorStandaloneAdaptor` / `Using custom backend` | `vllm/compilation/backends.py` | DEBUG level (needs `VLLM_LOGGING_LEVEL=DEBUG`). `InductorAdaptor` = the inductor track; `custom backend` = a legacy AscendCompiler track |
| 5 | `Compiling a graph for compile range` / `Cache the graph of compile range` / `Directly load the compiled graph(s)` | `vllm/compilation/backends.py` | subgraphs really entered the compiler; the cold-compile vs cache-hit fork (always read this row when chasing cold-start behavior) |
| 6 | `Inductor compile-backend track active: cudagraph_mode=.* (final)` | `vllm_ascend/platform.py` | the **effective** cudagraph mode after all preset/downgrade steps; use it to tell whether the `-O` preset landed PIECEWISE / FULL_AND_PIECEWISE or a downgrade chain fired |
| 7 | `torch.compile took .* s in total` | `vllm/compilation/monitor.py` | compilation finished, with wall time |

Notes:

- `triton_experimental` itself is silent by default — there is **no log at activation** (see the next section for how to verify it). For torch-side Inductor logs use `TORCH_LOGS=+inductor`; for the DEBUG rows above use `VLLM_LOGGING_LEVEL=DEBUG`.
- Warnings that say the track is inert: `VLLM_USE_BREAKABLE_CUDAGRAPH wins over compilation_config.backend='inductor'`, `TORCH_COMPILE_DISABLE=1 detected`, `VLLM_USE_AOT_COMPILE=1 is explicitly set` — see the guard table in section 7.

## 2. Is the Backend Really Active? (Intent vs Fact)

torch_npu exposes three entries for selecting `npu_backend`: a per-compile option (`torch.compile(fn, options={"npu_backend": ...})`), the global `torch._inductor.config.npu_backend`, and the env var `TORCHINDUCTOR_NPU_BACKEND`. All converge on `_load_backend()` in `torch_npu/_inductor/__init__.py`. **Under vLLM the env var is the only carrier that matters**: vLLM compiles pieces via `compile_fx`, not via the `torch.compile` wrapper, so a per-compile option never applies. `vllm_ascend/platform.py` sets `TORCHINDUCTOR_NPU_BACKEND=triton_experimental` before workers are spawned so they inherit it.

Activation is **lazy and logless**: `_activate()` in `torch_npu/_inductor/triton_experimental/__init__.py` runs at the first Inductor compilation; importing the package alone does nothing. "The env var is set" does not mean "the backend is active".

Two levels of evidence:

- **Intent level** — read the in-process registry after a compile:

  ```python
  from torch_npu.utils._dynamo import _InductorNpuRegistry
  print(_InductorNpuRegistry._loaded_backend)  # expect "triton_experimental" AFTER a compile
  ```

  Read it **after** the compile call: the per-compile option temporarily rewrites the process env and restores it on exit, so reading the env can lie. The registry (grep anchor `class _InductorNpuRegistry` in `torch_npu/utils/_dynamo.py`) is the trustworthy probe.

- **Fact level** — machine evidence from the generated code:

  ```bash
  grep -rl npu_triton_heuristics "$TORCHINDUCTOR_CACHE_DIR" --include=output_code.py | wc -l
  # non-zero = the triton_experimental backend produced kernels
  ```

  Caveat: when running under a vLLM engine, the upstream Inductor adaptor **hard-redirects** `TORCHINDUCTOR_CACHE_DIR` into the vLLM compile cache (an `inductor_cache/` directory; set in `initialize_cache` in `vllm/compilation/compiler_interface.py`). Search there, not in your pre-set env value.

## 3. Reading output_code.py (Evidence Judgement)

`output_code.py` is Inductor's final codegen product for one FX subgraph: a wrapper function (buffer allocation, call sequence) plus embedded Triton kernel source. Two places it lands:

- `TORCH_COMPILE_DEBUG=1` -> `<TORCH_COMPILE_DEBUG_DIR>/torch_compile_debug/<aot_id>/output_code.py`;
- the Inductor artifact cache (FxGraphCache serialization) -> `<TORCHINDUCTOR_CACHE_DIR>/<xx>/<hash>.debug/output_code.py` (under vLLM: inside the `inductor_cache/` redirect above).

The fastest judgement is the import block at the top of the file:

```python
from torch._inductor.runtime.triton_heuristics import start_graph, end_graph
from torch_npu._inductor.triton_experimental import npu_triton_heuristics   # <- the anchor
from torch_npu._inductor.triton_experimental import get_current_raw_stream as get_raw_stream
```

| What you grep in output_code.py | Judgement |
|---|---|
| `import npu_triton_heuristics` (torch_npu path) | the subgraph went through **triton_experimental** (injected by its wrapper codegen; no other backend emits it) |
| `@npu_triton_heuristics.pointwise(...)` and friends | each kernel is managed by the NPU autotuner (upstream prefix would be `triton_heuristics.`) |
| `extern_kernels.aclnnXXX(...)` | the operator deoptimized to an aclnn extern kernel — correct, but fusion is cut there (a performance concern, not a correctness one) |
| opaque custom-op launches (e.g. `moe_forward`, `mla_forward`, `_C_ascend.*` / `torch.ops.*` calls) | fused AscendC ops dispatched as opaque extern; their internals are invisible in output_code — expert/grouped compute inside them will not appear as Triton kernels even when everything is correct |
| only buffer allocations and a return, no kernel | the subgraph degenerated to a pure copy / splitting boundary — normal |
| no output_code.py generated at all | codegen never ran: the track is inert or a different track won (see Q2 below) |

## 4. Triage Decision Tree (Four Questions)

Start from the symptom "performance wrong / numbers wrong / I suspect the inductor track is not used at all" and walk down; each question ends in a layer verdict.

```text
Q1  Does the terminal show "Compiling a graph for compile range"?
 no  -> the config layer broke. Check "NPU does not support compilation mode",
        enforce-eager / -O0, and the guard warnings in section 7.
        [verdict: vllm-ascend config layer]
 yes |
     Q2  Was output_code.py generated (under the vLLM compile cache inductor_cache/)?
      no  -> inductor codegen never ran. Find which adaptor won
            ("Using InductorAdaptor" vs "Using custom backend", DEBUG level).
            Legacy tracks show exactly this picture.
            [verdict: routing layer]
      yes |
          Q3  Does output_code.py contain npu_triton_heuristics?
           no  -> the backend did not activate. Walk the three-entry
                 verification in section 2; most likely lazy-activation
                 timing (section 6, entry 1) or the value never reached
                 the worker process.
                 [verdict: torch_npu activation]
           yes |
               Q4  Is the kernel source itself right?
                 (fusion layout / extern_kernels share / grid shape)
                 -> apply the section 3 table and count extern kernels.
                    Wrong numbers: compare end-to-end greedy generations
                    against the eager track first.
                    Compile errors below codegen (MLIRCompilationError, ...):
                    go to section 5 artifact boundaries.
                    [verdict: below codegen]
```

## 5. Below Codegen: Artifact Boundaries A and B

For failures that come from the compiler stack under codegen (unit tests, single-kernel reproductions), classify the error signal first:

| Signal (grep the log) | Shape | Owner layer |
|---|---|---|
| `MissingOperatorWith(out)?Decomp` | compile error in Inductor | lowering / decomposition coverage (triton_experimental override lists in `torch_npu/_inductor/triton_experimental`) |
| `[ConvertTritonIRToLinalgIR]` | compile error downstream of boundary A | triton-ascend |
| `[ConvertLinalgRToBinary]` | compile error downstream of boundary B | bishengir-compile (AscendNPU-IR) |
| no compile error, but wrong numbers / crash / kernel-count anomaly | runtime | execution layer; bisect with the artifacts below |
| `TorchRuntimeError` / `fake tensor call` / graph break | dynamo / operator semantics | before any backend — reproduce on another backend first; if it fails there too, it is not a backend bug |

Artifacts and where they land:

| Artifact | Boundary | Location |
|---|---|---|
| `.ttir` | into triton-ascend — **boundary A** | `$TRITON_CACHE_DIR/<hash>/` (or the kernel-dump dir when a dump is forced) |
| `.ttadapter` | triton-ascend output / bishengir input — **boundary B** | same directory |
| `.npubin` | bishengir-compile output | same directory |

Attribution rule of thumb: an error raised before `triton.compile()` completes belongs to Inductor/triton_experimental; a `.ttir` that looks right but that triton-ascend fails to compile belongs to triton-ascend; a `.ttadapter` that looks right but that bishengir fails or miscompiles belongs to AscendNPU-IR. For numerics, do not assume eager execution is the reference truth: CANN eager transcendental accuracy can be worse than an algorithmic lowering — arbitrate against CPU fp64 as an independent reference before blaming the compiled path.

## 6. Known Breakpoints and Traps (Living List)

Each entry: symptom (grep anchor) -> root cause -> fix. Append a new entry whenever a breakpoint is root-caused.

1. **Lazy activation timing.** Symptom: `both a fallback and a decomp for same op: aten._to_copy.default`, raised from `make_fallback` (anchor `def make_fallback` in `torch/_inductor/lowering.py`). Custom entry points that reach Inductor before the first compile: env-channel backend activation happens at the first Inductor compilation, later than AOT decomposition, so the lazy assertion fires first. Fix: `import torch_npu._inductor` (triggers `_load_backend`) before compiling.

2. **Decompositions dropped by a compile_fx wrapper.** A wrapper whose signature receives `decompositions` but calls `aot_autograd(fw_compiler=...)` without forwarding them (anchor `def compile_fx` in `vllm_ascend/compilation/compiler_interface.py`) leaves `aten._to_copy` residues and amplifies entry 1. Fix: `aot_autograd(fw_compiler=..., decompositions=select_decomp_table())`.

3. **Illegal Inductor config key.** Symptom: `torch._inductor.config.<key> does not exist`, raised by `ConfigModule.patch`, which only accepts keys that already exist in the config. Cause: a pass-manager key (for example `graph_fusion_manager`) flowing into `config_patches` / `inductor_compile_config`. Fix: custom pass keys must never reach the Inductor config; mount passes on a real key such as `post_grad_custom_post_pass` instead.

4. **depyf is incompatible with torch 2.13 on this track (hard blocker).** Symptom: a crash whose stack ends in depyf's `patched_load_by_key_path` meeting the newer torch `codecache` kwarg `set_sys_modules`, at Triton-kernel load time. Setting `VLLM_DEBUG_DUMP_PATH` makes vLLM mount depyf, so merely setting that env var kills the run. Fix: never set `VLLM_DEBUG_DUMP_PATH` on the inductor track; use the `TORCH_COMPILE_DEBUG` leg (section 7.2) or upgrade depyf.

5. **Legacy-track AOT-cache crash.** Symptom: `AOTAutogradCache: expected OutputCode` (npugraph_ex): a legacy track's GraphModule-returning inner compiler cannot satisfy the AOT autograd cache's OutputCode contract when torch enables AOT by default; the legacy track disables that cache explicitly (see `enable_autograd_cache` in `vllm_ascend/compilation/compiler_interface.py`). Seeing this on what should be an inductor run means routing picked the wrong track — go back to Q2.

6. **Counters read zero from the driver process.** vLLM workers compile in their own (spawned) processes: `compilation_counter` and the pattern `match_table` updated inside a worker are invisible in the driver. Fix: assert in the compiling process (`compilation_counter.expect(...)`), or use artifact-level evidence (the section 3 grep). Anchor: `get_match_table` in `vllm/compilation/passes/vllm_inductor_pass.py`.

7. **`torch._dynamo.mark_dynamic` is a sticky tensor property.** Reusing input tensors across test cases carries the previous case's dynamic annotations into the next one; the failure (a `ConstraintViolationError` about a tensor specialized as a constant) points at the polluted case while the root cause is the previous one. Fix: fresh tensors per case.

8. **Informational attributes are not switches.** `compilation_config.oot_compiler` is written dynamically by the platform code and is not a vLLM `CompilationConfig` field; seeing it in a log does not mean vLLM consumed it. The track switch is `compilation_config.backend` (`-cc.backend inductor`), plus the guards below.

## 7. Track Observability Affordances

### 7.1 Config guards

Explicitly-set controls that silently change what the track compiles produce `warning_once` lines (never fatal, except the two hard `ValueError` rows):

| Condition | Grep anchor | Meaning |
|---|---|---|
| `VLLM_USE_BREAKABLE_CUDAGRAPH` set | `wins over compilation_config.backend='inductor'` | upstream forced mode to NONE; the track is inert. Set `VLLM_USE_BREAKABLE_CUDAGRAPH=0` to use the track |
| `TORCH_COMPILE_DISABLE=1` | `TORCH_COMPILE_DISABLE=1 detected` | upstream already disabled compilation; the track has no effect on this engine |
| `VLLM_USE_AOT_COMPILE=1` | `VLLM_USE_AOT_COMPILE=1 is explicitly set` | it runs, but reloading the saved AOT artifact on a later start is unverified on this track |
| user `TORCHINDUCTOR_NPU_BACKEND` != `triton_experimental` | `keeping user TORCHINDUCTOR_NPU_BACKEND` | your backend value is kept and used |
| `VLLM_USE_STANDALONE_COMPILE=1` / `VLLM_USE_MEGA_AOT_ARTIFACT=1` | `does not support` | hard `ValueError` — unadapted compile paths |
| `enforce_eager=True` / `-O0` | *(no error — upstream semantics: mode forced to NONE, track inert)* | hard `ValueError` — compilation fully disabled |

### 7.2 One-flag dump trigger

A non-empty `-cc.debug_dump_path` (for example `--compilation-config '{"debug_dump_path": "/tmp/dump"}'`) turns on `TORCH_COMPILE_DEBUG` via `os.environ.setdefault` in `vllm_ascend/platform.py` (grep anchor `compilation_config.debug_dump_path`), so Inductor drops `output_code` artifacts. The log line that confirms it: `compilation_config.debug_dump_path is set; TORCH_COMPILE_DEBUG=`.

Two deliberate non-actions, so you do not "fix" them: cache-dir envs are not touched (upstream `initialize_cache` hard-redirects `TORCHINDUCTOR_CACHE_DIR` into the vLLM compile cache anyway — the on-track artifact home is the `inductor_cache/` directory), and `VLLM_DEBUG_DUMP_PATH` is never set because it would mount depyf (section 6, entry 4).

### 7.3 Reading the pattern match_table (quantization fusion passes)

Each fusion pass under `vllm_ascend/compilation/passes/` (`norm_quant_fusion_pass.py`, `qknorm_rope_fusion_pass.py`, `muls_add_pass.py`) increments `VllmPatternMatcherPass.match_table[pass_name]` — pass names `rmsnorm_quant`, `qknorm_rope`, `muls_add`. Read it via `get_match_table()` from `vllm/compilation/passes/vllm_inductor_pass.py`.

Reading method:

- Read from a **cold compile** in the **compiling process** (spawn caveat, section 6 entry 6).
- All-zero hits on a model you expect to be fused means pattern mismatch. Diagnose from the `Source Nodes:` comments in `output_code.py`: (a) which op form was actually emitted — e.g. a custom-op variant with a baked-in bias versus the `bias=None` dynamic variant; (b) intermediate `view` / wrapper nodes between producer ops that the pattern must match through (easy to miss in a static code walk because they can be folded into extern wrappers); (c) matcher dtype coverage — integer dtype enums are a common blind spot.

### 7.4 Quick failure-triage table

| Symptom | First check |
|---|---|
| crash on `expected OutputCode` | legacy track + AOT autograd cache (section 6, entry 5); on an intended inductor run it means routing picked the wrong track |
| crash mentioning `patched_load_by_key_path` / `set_sys_modules` | `VLLM_DEBUG_DUMP_PATH` is set (depyf, section 6 entry 4) — unset it |
| `match_table` empty / counters read 0 | spawn semantics: the worker owns compilation — assert in-process or use artifact evidence (section 6, entry 6) |
| default `-O` run did not land FULL_AND_PIECEWISE | grep `Inductor compile-backend track active: cudagraph_mode=.* (final)` and distinguish the `-O` preset chain from a downgrade chain |
| no inductor logs at all | mode forced to NONE (`NPU does not support compilation mode`) or a section 7.1 guard fired — start at Q1 of the decision tree |

## 8. Keeping This Map Alive

This page is a living document, reviewed alongside the code it describes:

- Append an entry to section 6 whenever a breakpoint is root-caused (symptom grep anchor + owner layer + fix), and update sections 1–5 and 7 when routing, guards, or dump behavior change.
- Anchors only: log strings, symbols, env-var names, file paths — never bare line numbers. If an anchor stops matching after a version bump, fix the anchor in the same change.
