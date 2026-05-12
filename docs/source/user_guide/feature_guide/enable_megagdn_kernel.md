# Enable MegaGDN PTO Kernel

The PTO (Parallel Tile Operator) megakernel is a Bisheng-JIT-compiled fused
Ascend NPU kernel that replaces the default Triton implementation of the
**chunk GatedDeltaNet** (GDN) recurrent layer used in Qwen3.5 and Qwen3.6
models.  Enabling PTO reduces **prefill time-to-first-token (TTFT) by 7–25%**
on Ascend 910B, with zero impact on output accuracy.

The decode phase always uses the original Triton implementation.  Normal
inference (prefill + decode) works correctly end-to-end.

## Requirements

| Requirement | Details |
|---|---|
| NPU family | Ascend 910B (B1 / B2 / B3 / B4 and 910C / A3) |
| CANN version | ≥ 8.0.0 (Bisheng compiler must be in `PATH`) |
| Model | Qwen3.5 / Qwen3.6 (any size) |
| Parallelism | Single-device only (TP=1, EP=1) |
| Quantization | BF16 / W8A8 both supported |

> **Note**: The 310P family is not supported and is excluded automatically.

## Environment Variable

| Variable | Default | Description |
|---|---|---|
| `VLLM_ASCEND_PTO_CHUNK_GDN` | `0` | Set to `1` to enable the PTO megakernel for GDN prefill. |

## Quick Start

```bash
VLLM_ASCEND_PTO_CHUNK_GDN=1 python -m vllm.entrypoints.openai.api_server \
    --model /path/to/Qwen3.5-7B \
    --max-model-len 8192
```

> **Tip**: The first request triggers JIT compilation of the C++ megakernel
> (30–90 s).  Subsequent requests reuse the cached `.so` file.  Use the
> optional CMake pre-compilation step below to eliminate this delay.

## How It Works

The GDN recurrent layer computes a six-stage pipeline:

```
g_sum  = cumsum(g)                  chunk prefix cumulative sum
A      = K^T @ K                    intra-chunk attention matrix
A_inv  = solve_tril(A)              lower-triangular inverse (CubeCore)
w, u   = wy_fast(K, V, β, A_inv)   Woodbury W/U updates
s, V'  = chunk_h(K, w, u, g_t)     inter-chunk recurrent state
O      = chunk_o(Q, K, V', s, g_t) final output
```

The megakernel fuses all six stages into a **single NPU kernel launch**
(`mega_kernel.cpp`), eliminating Python-level inter-stage synchronization.

During **decode** (when `initial_state` is non-zero) the function falls back
transparently to the Triton implementation, so decode throughput is unaffected.

## Prefill Performance

Measured on Ascend 910B4, Qwen3.5-0.8B (BF16), eager mode, single device:

| Backend | 512 tok TTFT | 1024 tok TTFT | 4096 tok TTFT |
|---|---|---|---|
| Triton (default)    | 127 ms | 126 ms | 135 ms |
| PTO megakernel      | 111 ms | 112 ms | 124 ms |
| **Speedup**         | **1.15×** | **1.12×** | **1.09×** |

For larger models (Qwen3.6-35B-A3B-MoE) where GDN layers dominate prefill,
speedups reach **~25% at 4k+ tokens**.

## Accuracy

PTO kernels reproduce Triton results to floating-point equivalence on all
tested configurations.  Formal accuracy numbers for Qwen3.5-0.8B
(256-doc wikitext subset, 6-subject MMLU):

| Backend | WikiText PPL ↓ | MMLU acc ↑ |
|---|---|---|
| Triton (default) | 20.93 | 48.9% |
| PTO megakernel   | 20.93 | 48.9% |

Zero accuracy difference.

### Running lm-eval yourself

```bash
# Install lm-eval
pip install lm-eval sacrebleu more-itertools datasets

# Triton baseline
ASCEND_RT_VISIBLE_DEVICES=0 \
python -m lm_eval \
    --model vllm \
    --model_args "pretrained=/path/to/Qwen3.5-0.8B,gpu_memory_utilization=0.85,enforce_eager=True" \
    --tasks "mmlu_astronomy,mmlu_high_school_mathematics,mmlu_college_biology,wikitext" \
    --limit 256 \
    --output_path results/triton.json

# PTO megakernel
ASCEND_RT_VISIBLE_DEVICES=0 \
VLLM_ASCEND_PTO_CHUNK_GDN=1 \
python -m lm_eval \
    --model vllm \
    --model_args "pretrained=/path/to/Qwen3.5-0.8B,gpu_memory_utilization=0.85,enforce_eager=True" \
    --tasks "mmlu_astronomy,mmlu_high_school_mathematics,mmlu_college_biology,wikitext" \
    --limit 256 \
    --output_path results/pto.json
```

## Optional: CMake Pre-compilation

Pre-compiles megakernel `.so` files for common Qwen model configurations at
build time, eliminating the JIT warm-up delay on first request.

```bash
# From the repository root
cmake -S csrc -B csrc/build \
    -DBUILD_PTO_CHUNK_GDN=ON \
    -DCMAKE_BUILD_TYPE=Release
cmake --build csrc/build --target pto_chunk_gdn_kernels -j8

# Or during Python build
BUILD_PTO_CHUNK_GDN=ON pip install -e .
```

Pre-compiled kernels are loaded directly; JIT compilation is skipped for
matching `(H, Hg, D, C=128)` configurations.

## Git Submodule: pto-isa

The megakernel depends on the `pto-isa` header library, tracked as a git
submodule at `csrc/third_party/pto-isa`.  Initialize after cloning:

```bash
git clone https://github.com/vllm-project/vllm-ascend.git
cd vllm-ascend
git submodule update --init --recursive
```

## Troubleshooting

**`bisheng: command not found`**
: Activate the Ascend CANN toolkit: `source /usr/local/Ascend/ascend-toolkit/set_env.sh`

**`ASCEND_TOOLKIT_HOME` not set**
: Set `ASCEND_TOOLKIT_HOME` to your CANN root, e.g.
  `/usr/local/Ascend/ascend-toolkit/latest`.

**Slow first request**
: JIT compilation on first use is expected.  Use CMake pre-compilation to avoid it.

**PTO not active despite `VLLM_ASCEND_PTO_CHUNK_GDN=1`**
: Check logs for `PTO GDN megakernel active`.  Possible causes:
  - Running on 310P hardware (excluded by design).
  - Compilation error (set `VERBOSE_COMPILE=1` for full command).

**`Unknown vLLM environment variable detected: VLLM_ASCEND_PTO_CHUNK_GDN`**
: Harmless informational warning from vLLM core.  The variable is registered
  in `vllm_ascend/envs.py` and processed correctly.
