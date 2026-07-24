# Optimization and Tuning

This guide aims to help users improve vLLM Ascend performance at the system level. It includes OS configuration, library optimization, deployment guide, and so on. Any feedback is welcome.

## Preparation

Run the container:

```bash
# Update DEVICE according to your device (/dev/davinci[0-7])
export DEVICE=/dev/davinci0
# Update the cann base image
export IMAGE=m.daocloud.io/quay.io/ascend/cann:{{ cann_image_tag }}
docker run --rm \
--name performance-test \
--shm-size=1g \
--device $DEVICE \
--device /dev/davinci_manager \
--device /dev/devmm_svm \
--device /dev/hisi_hdc \
-v /usr/local/dcmi:/usr/local/dcmi \
-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
-v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
-v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
-v /etc/ascend_install.info:/etc/ascend_install.info \
-v /root/.cache:/root/.cache \
-it $IMAGE bash
```

Configure your environment:

```bash
# Configure the mirror
echo "deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu-ports/ jammy main restricted universe multiverse" > /etc/apt/sources.list && \
echo "deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu-ports/ jammy main restricted universe multiverse" >> /etc/apt/sources.list && \
echo "deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu-ports/ jammy-updates main restricted universe multiverse" >> /etc/apt/sources.list && \
echo "deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu-ports/ jammy-updates main restricted universe multiverse" >> /etc/apt/sources.list && \
echo "deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu-ports/ jammy-backports main restricted universe multiverse" >> /etc/apt/sources.list && \
echo "deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu-ports/ jammy-backports main restricted universe multiverse" >> /etc/apt/sources.list && \
echo "deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu-ports/ jammy-security main restricted universe multiverse" >> /etc/apt/sources.list && \
echo "deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu-ports/ jammy-security main restricted universe multiverse" >> /etc/apt/sources.list

# Install os packages
apt update && apt install wget gcc g++ libnuma-dev git vim -y
```

Install vLLM and vLLM Ascend:

```bash
# Install necessary dependencies
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip install modelscope pandas datasets gevent sacrebleu rouge_score pybind11 pytest

# Configure this var to speed up model download
export VLLM_USE_MODELSCOPE=True
```

Please follow the [Installation Guide](https://docs.vllm.ai/projects/ascend/en/latest/installation.html) to make sure vLLM and vLLM Ascend are installed correctly.

!!! note

    Make sure your vLLM and vLLM Ascend are installed after your Python configuration is completed, because these packages will build binary files using python in current environment. If you install vLLM and vLLM Ascend before completing section 1.1, the binary files will not use the optimized python.

## Optimizations

### 1. Memory Allocator Optimization

#### 1.1. jemalloc

**jemalloc** is a memory allocator that improves performance for multi-threaded scenarios and can reduce memory fragmentation. jemalloc uses a local thread memory manager to allocate variables, which can avoid lock competition between threads and can hugely optimize performance.

```bash
# Install jemalloc
sudo apt update
sudo apt install libjemalloc2

# Configure jemalloc
export LD_PRELOAD=/usr/lib/"$(uname -i)"-linux-gnu/libjemalloc.so.2:$LD_PRELOAD
```

#### 1.2. TCMalloc

**TCMalloc (Thread Caching Malloc)** is a universal memory allocator that improves overall performance while ensuring low latency by introducing a multi-level cache structure, reducing mutex contention and optimizing large object processing flow. Find more [details](https://www.hiascend.com/document/detail/zh/Pytorch/700/ptmoddevg/trainingmigrguide/performance_tuning_0068.html).

```bash
# Install tcmalloc
sudo apt update
sudo apt install libgoogle-perftools4 libgoogle-perftools-dev

# Get the location of libtcmalloc.so*
find /usr -name libtcmalloc.so*

# Make the priority of tcmalloc higher
# The <path> is the location of libtcmalloc.so we get from the upper command
# Example: "$LD_PRELOAD:/usr/lib/aarch64-linux-gnu/libtcmalloc.so"
export LD_PRELOAD="$LD_PRELOAD:<path>"

# Verify your configuration
# The path of libtcmalloc.so will be contained in the result if your configuration is valid
ldd `which python`
```

### 2. `torch_npu` Optimization

Some performance tuning features in `torch_npu` are controlled by environment variables. Some features and their related environment variables are shown below.

Memory optimization:

```bash
# Upper limit of memory block splitting allowed (MB): Setting this parameter can prevent large memory blocks from being split.
export PYTORCH_NPU_ALLOC_CONF="max_split_size_mb:250"
```

or

```bash
# When operators on the communication stream have dependencies, they all need to be ended before being released for reuse. The logic of multi-stream reuse is to release the memory on the communication stream in advance so that the computing stream can be reused.
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
```

Scheduling optimization:

```bash
# Optimize operator delivery queue. This will affect the memory peak value, and may degrade if the memory is tight.
export TASK_QUEUE_ENABLE=2

# This will greatly improve the CPU bottleneck model and ensure the same performance for the NPU bottleneck model.
export CPU_AFFINITY_CONF=1
```

### 3. CANN Optimization

#### 3.1. HCCL Optimization

There are some performance tuning features in HCCL, which are controlled by environment variables.

You can configure HCCL to use "AIV" mode to optimize performance by setting the environment variable shown below. In "AIV" mode, the communication is scheduled by AI vector core directly with RoCE, instead of being scheduled by AI CPU.

```bash
export HCCL_OP_EXPANSION_MODE="AIV"
```

Plus, there are more features for performance optimization in specific scenarios, which are shown below.

- `HCCL_INTRA_ROCE_ENABLE`: Use RDMA link instead of SDMA link between two 8Ps as the mesh interconnect link. Find more [details](https://www.hiascend.com/document/detail/zh/Pytorch/600/ptmoddevg/trainingmigrguide/performance_tuning_0044.html).
- `HCCL_RDMA_TC`: Use this var to configure traffic class of RDMA NIC. Find more [details](https://www.hiascend.com/document/detail/zh/Pytorch/600/ptmoddevg/trainingmigrguide/performance_tuning_0045.html).
- `HCCL_RDMA_SL`: Use this var to configure service level of RDMA NIC. Find more [details](https://www.hiascend.com/document/detail/zh/Pytorch/600/ptmoddevg/trainingmigrguide/performance_tuning_0046.html).
- `HCCL_BUFFSIZE`: Use this var to control the cache size for sharing data between two NPUs. Find more [details](https://www.hiascend.com/document/detail/zh/Pytorch/600/ptmoddevg/trainingmigrguide/performance_tuning_0047.html).

### 4. Kernel Optimization

This section describes operating system–level optimizations applied on the host machine (bare metal or Kubernetes node) to improve performance stability, latency, and throughput for inference workloads.

!!! note

    These settings must be applied on the host OS and with root privileges. Not inside containers.

#### 4.1 Set CPU Frequency Governor to `performance`

```shell
echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
```

Purpose

- Forces all CPU cores to run under the `performance` governor
- Disables dynamic frequency scaling (e.g., `ondemand`, `powersave`)

Benefits

- Keeps CPU cores at maximum frequency
- Reduces latency jitter
- Improves predictability for inference workloads

#### 4.2 Disable Swap Usage

```shell
sysctl -w vm.swappiness=0
```

Purpose

- Minimizes the kernel’s tendency to swap memory pages to disk

Benefits

- Prevents severe latency spikes caused by swapping
- Improves stability for large in-memory models

Notes

- For inference workloads, swap can introduce second-level latency
- Recommended values are `0` or `1`

#### 4.3 Disable Automatic NUMA Balancing

```shell
sysctl -w kernel.numa_balancing=0
```

Purpose

- Disables the kernel’s automatic NUMA page migration mechanism

Benefits

- Prevents background memory page migrations
- Reduces unpredictable memory access latency
- Improves performance stability on NUMA systems

Recommended For

- Multi-socket servers
- Ascend / NPU deployments with explicit NUMA binding
- Systems with manually managed CPU and memory affinity

#### 4.4 Increase Scheduler Migration Cost

```shell
sysctl -w kernel.sched_migration_cost_ns=50000
```

Purpose

- Increases the cost for the scheduler to migrate tasks between CPU cores

Benefits

- Reduces frequent thread migration
- Improves CPU cache locality
- Lowers latency jitter for inference workloads
  
Parameter Details

- Unit: nanoseconds (ns)
- Typical recommended range: 50000–100000
- Higher values encourage threads to stay on the same CPU core

## 6. vLLM Ascend Software Tuning

Sections 1–5 above focus on the host OS, CANN/HCCL, and `torch_npu`. This section explains **how to tune vLLM Ascend itself**: which class of problem you are solving, which option addresses it, and what usually goes wrong if you enable something in the wrong scenario.

Model-specific commands and benchmark numbers stay in [model deployment tutorials](../../tutorials/models/index.md). Option semantics and compatibility matrices are in the [Feature Guide](../../user_guide/feature_guide/index.md) and [Additional Configuration](../../user_guide/configuration/additional_config.md).

<a id="ascend-tuning-workflow"></a>

(ascend-tuning-workflow)=

### 6.1 Tuning workflow

Use the same workflow for every change:

1. **Define the scenario and metrics** — throughput (tokens/s), TTFT, TPOT, max concurrency, memory headroom, and whether traffic is prefill-heavy or decode-heavy.
2. **Classify the primary bottleneck** — use [Section 6.2](#ascend-tuning-bottleneck) (one dominant category is enough to start).
3. **Change one tuning option at a time** — keep topology, weights, and unrelated flags fixed; compare A/B on the same benchmark.
4. **Re-measure and record constraints** — note mutual exclusions (for example, `enforce_eager` vs graph mode, FlashComm1 vs small batches, batch invariant vs some Ascend options).

<a id="ascend-tuning-bottleneck"></a>

(ascend-tuning-bottleneck)=

### 6.2 Classify the primary bottleneck

Match your symptoms to **one** row below. That row tells you which subsection in [Section 6.3](#ascend-tuning-options) to read first.

| Symptoms (what you see in metrics or profiling) | Bottleneck class | Read first |
|-------------------------------------------------|------------------|------------|
| Decode QPS is low while batch size is already moderate; CPU scheduler or Python host path is hot; per-token latency varies a lot step to step | **Runtime overhead** | [6.3.1 Runtime overhead (graph and scheduling)](#ascend-tuning-runtime) |
| Adding TP ranks does not scale throughput; large prefill steps spend much time in HCCL; very long prompts OOM or have high TTFT | **Parallelism and communication** | [6.3.2 Parallelism and communication](#ascend-tuning-parallelism) |
| Linear layers show high MTE time or are clearly memory-bound; changing quant format or weight layout moves the needle | **Compute and memory bandwidth** | [6.3.3 Compute and memory bandwidth](#ascend-tuning-compute) |
| Raising `max-num-seqs` or `max-num-batched-tokens` causes OOM; throughput stays flat because each long prompt is split into many small prefill steps; scheduling feels memory- or token-cap bound (if APC is on but prompts still pay full prefill, check prefix overlap and APC settings—not only batch caps) | **Capacity and batching** | [6.3.4 Capacity and batching](#ascend-tuning-capacity) |

<a id="ascend-tuning-options"></a>

(ascend-tuning-options)=

### 6.3 Tuning options by bottleneck class

Each table uses the same columns:

- **Option** — name in docs or config.
- **Addresses** — which cost it reduces.
- **Mechanism** — what the stack actually does (one sentence).
- **How to enable** — `Default` or explicit flag/CLI.
- **When not to use** — common mistakes.

<a id="ascend-tuning-runtime"></a>

(ascend-tuning-runtime)=

#### 6.3.1 Runtime overhead (graph and scheduling)

**Bottleneck class:** framework dispatch, graph capture mismatch, and CPU-side scheduling overhead.

Graph-mode rows below apply to the [**V1 engine**](../../user_guide/feature_guide/graph_mode.md) today.

| Option | Addresses | Mechanism | How to enable | When not to use |
|--------|-----------|-----------|---------------|-----------------|
| [**AddRMSNormQuant fusion**](../../user_guide/configuration/additional_config.md) | Extra memory traffic around norm + quant | Fuses add, RMSNorm, and quant into fewer kernels | **Default** via `fuse_norm_quant` in `ascend_compilation_config` when compile is active | `enforce_eager=True` or graph/compile disabled; Ascend 310P (pass not applied); turning off passes without a reason |
| [**QKNorm–Rope fusion**](../../user_guide/configuration/additional_config.md) | Kernel launch and memory traffic around QK norm + RoPE | Fuses QK norm and RoPE when shapes match (for example, `head_dim == 128`) | **Default** where supported (`fuse_qknorm_rope`) when compile is active; set `false` if Triton is unavailable | Same as AddRMSNormQuant for eager/compile; models or dtypes the pass does not support |
| [**Graph mode (ACLGraph; Npugraph_ex on FULL paths)**](../../user_guide/feature_guide/graph_mode.md) | Per-step launch and Python scheduling overhead | ACLGraph captures and replays the execution graph; on `FULL` / `FULL_DECODE_ONLY`, Npugraph_ex rewrites the FX graph **before** capture (it does not replace ACLGraph). Default `FULL_AND_PIECEWISE` uses ACLGraph without that Npugraph_ex path | `--compilation-config` (for example, `"cudagraph_mode": "FULL_DECODE_ONLY"`) | Non–V1 engine; `enforce_eager=True`; some context-parallel + `FULL` combinations — see [Graph Mode Guide](../../user_guide/feature_guide/graph_mode.md) |
| [**`cudagraph_capture_sizes`**](https://docs.vllm.ai/en/latest/design/cuda_graphs/) | Padding waste when a step’s token count falls between captured buckets | Each entry is a **batch token count** (`num_tokens` after bucketing) for which a graph is captured; runtime steps round **up** to the next listed size (or fall back to eager if above the max) | List sizes in `--compilation-config` from profiling or scheduler logs; see [ACL Graph — capture sizes](../Design_Documents/ACL_Graph.md#capture-sizes-and-bucketing) | Misaligned buckets; with [FlashComm1 (FC1)](#ascend-tuning-parallelism), entries must be **multiples of TP** (filtered at init). [SP Pass](../../user_guide/feature_guide/sequence_parallelism.md) also requires token counts aligned to TP—see that guide |
| [**`--async-scheduling`**](https://docs.vllm.ai/en/latest/configuration/engine_args/#async-scheduling-no-async-scheduling) | CPU becomes the limiter at high concurrency | Overlaps scheduling work with NPU execution | CLI flag (`--async-scheduling`) | Check interaction with speculative decoding, pipeline parallel, and batch-invariant mode |

**Example — graph mode:**

```bash
vllm serve Qwen/Qwen3-8B \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}'
```

<a id="ascend-tuning-parallelism"></a>

(ascend-tuning-parallelism)=

#### 6.3.2 Parallelism and communication

**Bottleneck class:** tensor-parallel collectives, long-sequence partitioning, and weights spread across ranks.

**FC1/SP and PCP/DCP solve different problems—do not use one as a substitute for the other.** FC1 and SP Pass target TP Norm comm (FC1: custom ops, often eager or graph; SP Pass: compile-time, graph only). PCP/DCP target sequence length. Some model guides still forbid certain combinations (for example, FC1 with PCP/DCP on specific architectures).

| Option | Addresses | Mechanism | How to enable | When not to use |
|--------|-----------|-----------|---------------|-----------------|
| [**FlashComm1 (FC1)**](../../user_guide/feature_guide/sequence_parallelism.md#difference-between-sp-and-flash-comm-v1) | Large AllReduce + RMSNorm (+ quant) phases inside a **TP team** | Custom ops implement **ReduceScatter → local RMSNorm (+ quant) → AllGather** instead of one AllReduce on full hidden states; active only when scheduled **token count exceeds a runtime threshold** (dense models: platform often gates above ~1000 tokens per step—confirm in profiling; MoE paths differ) | [`enable_flashcomm1`](../../user_guide/configuration/additional_config.md) | Small batches (below threshold, may be slower than baseline); vision-language models (use SP Pass instead); on **dense** workloads FC1 may also enable Matmul–ReduceScatter fusion when tokens exceed the threshold—**MoE + FC1** usually does not follow the same Matmul–RS path; combinations your model doc marks unsupported |
| [**Sequence parallelism (SP, Pass)**](../../user_guide/feature_guide/sequence_parallelism.md) | Same collective pattern as FC1, for **VL** graphs | Compile-time pass rewrites AllReduce+RMSNorm to RS/AG; requires graph mode | `--compilation-config '{"pass_config": {"enable_sp": true}}'` | Non-VL models today; quantized SP still limited |
| [**Context parallel (PCP / DCP)**](../../user_guide/feature_guide/context_parallel.md) | Memory and time on **very long sequences** | Splits sequence dimension across ranks for prefill and/or decode | `--prefill-context-parallel-size`, `--decode-context-parallel-size`; design note: [context parallel](../Design_Documents/context_parallel.md) | Short-context workloads; orthogonal to FC1 — pick for length, not for TP Norm comm |
| [**`enable_matmul_allreduce`**](../../user_guide/configuration/additional_config.md) | Matmul immediately followed by AllReduce under TP | **Custom-op** fusion (separate from the MatmulAllReduceAddRMSNorm **compiler pass** on some Npugraph_ex graphs) | `--additional-config '{"enable_matmul_allreduce": true}'` | Default **off**; enable only when your model/hardware guide recommends it |
| [**FlashComm2**](../../user_guide/feature_guide/layer_sharding.md#flashcomm2-enabled) | Memory and comm around **output projection (`o_proj`)** | Shards `o_proj` with [`enable_flashcomm2_parallel_size`](../../user_guide/configuration/additional_config.md) and adjusts comm domains; not a replacement for FC1 | `--additional-config '{"enable_flashcomm2_parallel_size": N}'` (`N > 0`) | Treating it as “FC1 version 2” |
| [**Fine-grained TP**](../../user_guide/feature_guide/Fine_grained_TP.md) | Uneven comm across modules (lm_head, MLP, embedding) | Different TP widths per module | [`finegrained_tp_config`](../../user_guide/configuration/additional_config.md) in `--additional-config` | Before basic TP size is sane |
| [**Layer sharding**](../../user_guide/feature_guide/layer_sharding.md) | Full-layer weights too large (often PD prefill node) | Shards selected linear layers across ranks | [`layer_sharding`](../../user_guide/configuration/additional_config.md) in `--additional-config` (PD prefill / P role per feature guide) | Outside [PD-disaggregated prefill (P node)](../../user_guide/feature_guide/layer_sharding.md); colocated single-node jobs with enough memory and no layer-shard need |

**Suggested order:** (1) fix TP/EP/PP/PD topology → (2) long context → try PCP/DCP → (3) VL → SP Pass in graph mode → (4) non-VL TP workloads → FC1 when batch tokens are large enough (dense models: often ~1000+; MoE may apply without that cap—see your model guide) → (5) set `cudagraph_capture_sizes` to the **padded token counts** you see in graph-mode steps (multiples of TP when [FlashComm1 (FC1)](#ascend-tuning-parallelism) is on).

**Example — FlashComm1:**

```bash
vllm serve Qwen/Qwen3-32B \
  --tensor-parallel-size 4 \
  --additional-config '{"enable_flashcomm1": true}'
```

<a id="ascend-tuning-compute"></a>

(ascend-tuning-compute)=

#### 6.3.3 Compute and memory bandwidth

**Bottleneck class:** weight movement (MTE), layout-friendly math on Cube, and MoE dispatch/combine.

| Option | Addresses | Mechanism | How to enable | When not to use |
|--------|-----------|-----------|---------------|-----------------|
| [**Weight prefetch**](../../user_guide/feature_guide/weight_prefetch.md) | MTE stalls before large linear layers | A side pipeline issues **CMO** prefetches into L2 while vector ops (RMSNorm, SwiGLU, etc.) run on another pipeline | [`weight_prefetch_config`](../../user_guide/configuration/additional_config.md) in `--additional-config` | **Throughput** serving only: prefetch uses vector time to hide DMA—wrong `prefetch_ratio` hurts latency and can slow the model; confirm overlap on the profiling timeline before raising ratios; keep off for low-latency SLOs; graph path should be stable first; MLP `down` prefetch requires **sequence parallel to be active** ([SP Pass](../../user_guide/feature_guide/sequence_parallelism.md) or [FlashComm1](../../user_guide/feature_guide/sequence_parallelism.md#difference-between-sp-and-flash-comm-v1)—see the prefetch guide) |
| [**`weight_nz_mode`**](../../user_guide/configuration/additional_config.md) | Cube efficiency on weight layout | Stores weights in **FRACTAL_NZ** when allowed (`0` off, `1` quant only, `2` aggressive) | `--additional-config '{"weight_nz_mode": 1}'` (default **1**) | Expecting it to fix HCCL bottlenecks |
| [**Quantization (W8A8, W4A8, …)**](../../user_guide/feature_guide/quantization.md) | Arithmetic and collective volume | Lower precision kernels and smaller transfers | Model-specific quant flags and weights | Without accuracy validation; MoE paths may need [`enable_mlapo`](../../user_guide/configuration/additional_config.md) or [`enable_fused_mc2`](../../user_guide/configuration/additional_config.md) per model guide |
| [**`enable_mlapo`** (MoE)](../../user_guide/configuration/additional_config.md) | DeepSeek-class W8A8 layer execution | Layer-wise adaptive parallel layout; trades **more NPU memory** for speed | Default **on**; disable via `--additional-config '{"enable_mlapo": false}'` | When memory is tighter than latency |
| [**`enable_fused_mc2`** (MoE)](../../user_guide/configuration/additional_config.md) | MoE dispatch/combine overhead | Replaces default ALLTOALL+MC2 with fused operators under strict constraints | `--additional-config '{"enable_fused_mc2": 1}'` or `2` | Wrong PD role, EP size, or MTP dtype — see [Large-scale EP](../../user_guide/feature_guide/large_scale_ep.md) |
| [**EPLB** (MoE)](../../user_guide/feature_guide/eplb_swift_balancer.md) | Hot experts on fixed EP layout | Rebalances expert placement from heat maps | [`eplb_config`](../../user_guide/configuration/additional_config.md) in `--additional-config` | Small EP jobs where overhead dominates |

<a id="ascend-tuning-capacity"></a>

(ascend-tuning-capacity)=

#### 6.3.4 Capacity and batching

**Bottleneck class:** token and sequence admission per step, prefill chunking, and KV storage placement.

| Option | Addresses | Mechanism | How to enable | When not to use |
|--------|-----------|-----------|---------------|-----------------|
| [**`max-num-batched-tokens`**](https://docs.vllm.ai/en/stable/configuration/optimization/) | Throughput vs peak memory per scheduler step | Caps total tokens in one forward pass | CLI (`vllm serve` / [engine args](https://docs.vllm.ai/en/latest/configuration/engine_args/)) | Too low → many [chunked prefill](https://docs.vllm.ai/en/stable/configuration/optimization/#chunked-prefill) rounds per long prompt, throughput collapses; too high → OOM |
| [**`max-num-seqs`**](https://docs.vllm.ai/en/latest/configuration/engine_args/) | Concurrent sequences vs KV footprint | Caps number of active sequences | CLI | Raising it without memory headroom |
| [**Chunked prefill**](https://docs.vllm.ai/en/stable/configuration/optimization/#chunked-prefill) | Long single prompts blocking the batch | Splits prefill into multiple forward passes under the token cap | vLLM scheduler (often on by default) | `max-num-batched-tokens` so small that each chunk is tiny — looks like “low QPS” in [Section 6.2](#ascend-tuning-bottleneck) |
| [**Prefix caching (APC)**](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/) | Repeated prompt prefixes | Reuses KV blocks for shared prefixes | vLLM prefix-cache settings | Expecting gain when prompts do not share prefixes |
| [**PD / EPD disaggregation**](../../user_guide/feature_guide/epd_disaggregation.md) | Competing TTFT and TPOT on one pool | Runs prefill and decode on separate engines | `kv_transfer_config`, roles; see also [disaggregated prefill](../Design_Documents/disaggregated_prefill.md) | Simple single-node latency tests without PD infra |
| [**KV offload**](../../user_guide/feature_guide/kv_cache_cpu_offload.md) / [**KV pool**](../../user_guide/feature_guide/kv_pool.md) | KV exceeds device memory | Moves or shares KV across CPU/remote store | Per feature guide; pool design: [KV Cache Pool](../Design_Documents/KV_Cache_Pool_Guide.md) | Before batch limits are tuned |
| [**Dynamic batch**](../../user_guide/feature_guide/dynamic_batch.md) / [**`enable_balance_scheduling`**](../../user_guide/configuration/additional_config.md) | Uneven load across time or ranks | SLO-driven chunk sizing or Ascend balance scheduler | `--SLO_limits_for_dynamic_batch` (see [dynamic batch](../../user_guide/feature_guide/dynamic_batch.md)); balance scheduling via `--additional-config` | Until baseline batching is stable |

Set `max-num-batched-tokens` and `max-num-seqs` to a **stable, non-OOM** point before tuning graph capture lists or FlashComm1.
