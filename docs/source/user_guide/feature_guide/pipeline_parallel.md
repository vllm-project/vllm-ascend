# Pipeline Parallelism

Pipeline Parallelism (PP) partitions a model's Transformer hidden layers into
consecutive pipeline stages. Each stage is executed by one Tensor Parallel (TP)
group, or by one worker when the TP size is `1`, and sends intermediate
activations to the next stage. For example, `--pipeline-parallel-size 2`
divides the hidden layers between two pipeline stages.

Input tokens → PP stage 0 (earlier hidden layers) → intermediate activations →
PP stage 1 (later hidden layers, final norm, LM head, and sampling) → output
tokens.

This guide uses the following terms:

| Term | Meaning |
| --- | --- |
| PP stage | A logical partition containing a consecutive range of model layers. |
| PP rank | A worker's stage index within its PP communication group. Workers with the same stage index and different TP ranks jointly execute one stage. |
| TP group | The workers that apply TP together within one PP stage. |
| Node | A physical server. One node can contain multiple PP stages or TP groups. |

!!! note

    Not every model implementation supports PP. Before deployment, check the
    **Pipeline Parallel** column in [Supported Models](../support_matrix/supported_models.md).

For a first deployment, begin with **Quick Start**. Before production, review
**Configure Layer Partitioning** and the compatibility limitations. Use the
advanced scenarios and performance analysis when the deployment needs
additional features or tuning.

## Quick Start

The following examples show the minimum options for starting a basic PP
deployment. Review the topology, compatibility, and performance sections
before using the configuration in production.

### Before You Begin

Verify the following before starting the service:

- The visible NPU count satisfies the selected TP/PP/DP topology.
- All nodes use the same vLLM, vLLM Ascend, model weights, and Python environment.
- All nodes can access the same model path or equivalent local copies.
- For an MP deployment, every node can reach the head node's master address
  and port.
- HCCL and, for a Ray deployment, the Ray resource view are healthy.

### Single-Node Deployment

The following example demonstrates the minimum PP syntax on two NPUs. Use this
topology only when layer partitioning is required:

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1

vllm serve /path/to/model \
    --tensor-parallel-size 1 \
    --pipeline-parallel-size 2 \
    --trust-remote-code
```

Increase `--tensor-parallel-size` when each stage should use a TP group. For
example, `TP4 PP2` requires eight visible NPUs.

### Two-Node Deployment with Multiprocessing

The multiprocessing (MP) backend can start a PP deployment across two nodes
without a Ray cluster. The following example uses two 8-NPU nodes, keeps TP
inside each node, and places one PP stage on each node. Replace
`<HEAD_NODE_IP>` with an address that the worker node can reach and
`<MASTER_PORT>` with an unused TCP port. Use the same values on both nodes.

On the head node, use node rank `0`:

```bash
vllm serve /path/to/model \
    --distributed-executor-backend mp \
    --tensor-parallel-size 8 \
    --pipeline-parallel-size 2 \
    --nnodes 2 \
    --node-rank 0 \
    --master-addr <HEAD_NODE_IP> \
    --master-port <MASTER_PORT> \
    --trust-remote-code
```

On the worker node, use node rank `1` and add `--headless`:

```bash
vllm serve /path/to/model \
    --distributed-executor-backend mp \
    --tensor-parallel-size 8 \
    --pipeline-parallel-size 2 \
    --nnodes 2 \
    --node-rank 1 \
    --master-addr <HEAD_NODE_IP> \
    --master-port <MASTER_PORT> \
    --headless \
    --trust-remote-code
```

Only the head node starts the API server. Both nodes must use identical model,
TP, PP, `--nnodes`, `--master-addr`, and `--master-port` values. Assign each
node a unique `--node-rank` from `0` to `--nnodes - 1`, and set the required
HCCL and communication environment variables before running either command.

### Multi-Node Deployment with Ray

After starting the Ray cluster, run `vllm serve` on the head node. For two
8-NPU nodes, keep TP inside each node and use PP between the nodes:

```bash
vllm serve /path/to/model \
    --distributed-executor-backend ray \
    --tensor-parallel-size 8 \
    --pipeline-parallel-size 2 \
    --trust-remote-code
```

Start Ray only after setting the required communication environment variables
on every node. The complete cluster setup is documented in
[Ray Distributed](../../tutorials/features/ray.md).

## When to Use PP

PP is useful in the following situations:

- A model does not fit in one TP group and must be divided further by layer.
- A multi-node deployment should keep high-frequency TP collectives inside each
  node while sending activations between nodes through PP.
- A Prefill node in a P/D-disaggregated deployment needs more devices for model
  capacity or long-context prefill.

PP also introduces activation transfers between stages and pipeline bubbles.
Use a TP-only deployment as the baseline when the model fits in one node, then
add PP when model capacity or the network topology requires it.

| Deployment goal | Recommended starting point |
| --- | --- |
| The model fits in one node | Start with TP only and benchmark before adding PP. |
| The model spans multiple nodes | Use TP within a node and PP across nodes. |
| Multiple serving replicas are required | Build one TP/PP replica first, then add DP only on a supported model, operator, and network topology. |
| Long-context Prefill needs PP tuning | Start with PP, then evaluate [Dynamic Chunked Pipeline Parallel](dynamic_chunk_pipeline_parallel.md). |

## Configure Layer Partitioning

Layer partitioning determines which range of target-model hidden layers each PP
stage executes. If the target model has `N` hidden layers and the PP size is
`P`, the stage selected by PP rank `i` uses a half-open interval:

```text
PP rank i: [start_layer_i, end_layer_i)
```

Only the target model's Transformer hidden layers participate in this
partitioning.

### Automatic Partitioning

By default, vLLM distributes hidden layers as evenly as possible. When the layer
count is not divisible by the PP size, vLLM assigns the remaining layers
starting from the second-to-last stage and proceeding toward lower PP ranks.
This avoids adding more hidden layers to the last stage, which normally also
runs the final norm, LM head, logits, and sampling. With more than two stages,
small remainders are placed on middle stages to reduce pressure on the input
and output stages.

For a target model with 61 hidden layers:

```text
PP2: [31, 30]
PP rank 0: [0, 31)
PP rank 1: [31, 61)

PP4: [15, 15, 16, 15]
PP rank 0: [0, 15)
PP rank 1: [15, 30)
PP rank 2: [30, 46)
PP rank 3: [46, 61)
```

Use automatic partitioning first. Customize it only when measurements show a
stage-level memory or latency imbalance.

### Custom Partitioning

Set `VLLM_PP_LAYER_PARTITION` to the number of target hidden layers executed by
each PP stage, in PP-rank order:

```bash
export VLLM_PP_LAYER_PARTITION="32,29"

vllm serve /path/to/61-layer-model \
    --tensor-parallel-size 8 \
    --pipeline-parallel-size 2 \
    --trust-remote-code
```

In this example:

```text
PP rank 0: [0, 32)
PP rank 1: [32, 61)
```

The partition must satisfy all of the following rules:

- Use one positive integer per PP rank, separated by commas.
- The number of entries must equal `--pipeline-parallel-size`.
- The sum must equal the target model's hidden-layer count.
- Do not include embeddings, the final norm, the LM head, or draft-model layers.
- Use the same value for every worker in one service instance.
- On Ray clusters, set the value on every node before starting Ray.
- Restart the service after changing the value.

An invalid entry count or layer sum causes startup to fail.

### Tune Stage Balance

Stage balance is workload-dependent. The last stage often needs extra memory
and compute for the final norm, LM head, logits, sampling, and possibly a local
speculative-decoding drafter.

Use the following tuning loop:

1. Start with automatic partitioning.
2. Record peak memory and step latency for the workers in every PP stage under
   a representative workload.
3. If the last stage is the bottleneck, move one target hidden layer at a time
   from the last stage to a preceding stage with a lower PP rank, for example
   from `[31,30]` to `[32,29]`.
4. Repeat the accuracy and performance test after every change.

Moving too many layers can merely transfer the bottleneck to another stage, so
do not select a partition using memory capacity alone.

## Advanced Scenarios

### PP with Speculative Decoding

`VLLM_PP_LAYER_PARTITION` partitions only the target model. Draft-model layers
are not included in its layer sum, and their weight naming and layer layout
depend on the specific speculative-decoding implementation.

In the current Model Runner V1 local MTP and EAGLE proposer path, the drafter is
loaded on the last PP stage rather than partitioned across PP stages. Therefore,
the last stage can contain:

- Its target-model hidden layers.
- The target model's output-side modules.
- The local drafter and its supporting modules.

If the last rank is constrained by memory or latency, reduce its target hidden
layers with a custom partition. `draft_tensor_parallel_size` controls only the
draft model's TP size; it does not define a draft-model PP partition.

Speculative-decoding support varies by model, method, and model runner. See
[Speculative Decoding](speculative_decoding.md) and the relevant model tutorial
before enabling it with PP.

### PP in P/D-Disaggregated Serving

PP is commonly enabled on the Prefill node, while the Decode node remains
`PP1`. When using `MooncakeConnectorV1`, both the Prefill and Decode
configurations must describe the actual Prefill topology.

| Setting | Prefill node | Decode node |
| --- | --- | --- |
| `--pipeline-parallel-size` | Actual Prefill PP size, for example `2`. | Currently `1`. |
| `prefill.pp_size` | Actual Prefill PP size. | The same Prefill PP size. |
| `prefill.pp_layer_partition` | Custom Prefill partition, if used. | The same custom Prefill partition. |
| `decode.pp_size` | `1` | `1` |
| `VLLM_PP_LAYER_PARTITION` | Set when the Prefill node uses a custom partition. | Do not copy the Prefill value to a `PP1` Decode process. |

For a 61-layer Prefill model using `TP8 PP2` and the custom partition
`[32,29]`, include the following topology in `kv_connector_extra_config` on
both sides:

```json
{
  "prefill": {
    "dp_size": 1,
    "tp_size": 8,
    "pp_size": 2,
    "pp_layer_partition": "32,29"
  },
  "decode": {
    "dp_size": 1,
    "tp_size": 1,
    "pp_size": 1
  }
}
```

The Prefill process must also export:

```bash
export VLLM_PP_LAYER_PARTITION="32,29"
```

If automatic partitioning is used, omit `pp_layer_partition` on both sides.
The current Mooncake connector requires the Prefill TP size to be greater than
or equal to the Decode TP size and to be an integer multiple of it.

For complete P/D startup commands, see
[PD Disaggregation with Mooncake](../../tutorials/features/pd_disaggregation_mooncake_multi_node.md).
For long-context Prefill optimization, see
[Dynamic Chunked Pipeline Parallel](dynamic_chunk_pipeline_parallel.md).

## Performance Characteristics: PP Compared with TP

PP and TP distribute model execution in different ways, so neither strategy is
universally faster. TP partitions supported operations within Transformer
layers. TP workers participate in every layer and use collective communication
to combine partial results, while components that are not partitioned can
remain replicated. PP assigns consecutive layers to different stages and
transfers intermediate tensors between adjacent stages. In a hybrid topology,
TP communication remains inside each PP stage, while PP communication connects
the stages.

The primary performance advantage of PP over a large cross-node TP group is
**communication locality**, rather than faster computation within an individual
layer.

| Aspect | Tensor Parallelism | Pipeline Parallelism | Performance implication |
| --- | --- | --- | --- |
| Model partition | Every TP rank participates in every layer and holds either a shard or a replica of each component. | Each PP stage executes only its assigned layers; TP can additionally shard those layers. | PP can extend model capacity without creating a very large TP group. |
| Communication | Collective operations can occur inside many Transformer layers. | Intermediate tensors are transferred between adjacent PP stages. | Keeping TP inside each node usually reduces the frequency of cross-node communication. |
| Single-request latency | All TP workers contribute to the same layer concurrently. | A request traverses the PP stages in sequence. | TP usually favors latency-sensitive, low-concurrency workloads when the TP interconnect is fast. |
| Steady-state throughput | A large TP group can lose efficiency to collective latency and smaller per-rank matrix operations. | Concurrent scheduler batches formed from independent requests or Prefill chunks can keep different stages occupied. | PP can improve throughput when the pipeline has enough work and the stages are balanced. |
| Memory | Weights are sharded within each layer; some tensors can remain replicated. | Weights and KV cache are divided by layer, while edge stages also hold input-side or output-side modules. | Both reduce per-rank memory, but equal parallel sizes do not guarantee equal peak memory. |
| Scaling constraints | The usable TP size can be limited by attention heads, KV heads, hidden dimensions, expert routing, or supported kernels. | The model must support PP and have partitionable layers. | PP is useful when increasing TP is invalid or no longer efficient. |
| Main bottleneck | A slow collective delays every rank in the TP group. | The slowest stage and its boundary transfer limit the pipeline. | TP depends on collective efficiency; PP depends on stage balance and activation-transfer efficiency. |

### Pipeline Bubbles and Workload Effects

PP introduces idle time while work enters and leaves the pipeline, and whenever
one stage takes longer than the others. This idle time is commonly called a
pipeline bubble. Equal layer counts do not guarantee equal stage latency: the
first and last stages can also execute embeddings, the final norm, LM head,
logits processing, sampling, or a speculative-decoding drafter.

| Workload characteristic | Expected behavior |
| --- | --- |
| One request or low-concurrency Decode | There is little independent work to occupy multiple stages. Stage traversal and communication commonly increase latency relative to an efficient node-local TP baseline. Against a TP group that spans nodes, measure both configurations because cross-node collective latency can reverse the result. |
| High concurrency with continuous batching | Concurrent scheduler batches can occupy different stages, reducing the relative pipeline bubble and improving throughput. |
| Long-context Prefill | More computation per chunk can amortize PP communication, and PP can provide the memory capacity required by long contexts. |
| Variable prompt lengths | Batch and chunk durations vary, so a fixed-length benchmark can hide stage imbalance. |
| Unbalanced layer partition | The slowest stage makes the other stages wait and limits steady-state throughput. |

PP is therefore more likely to outperform cross-node TP when TP can remain
inside each node, the stages can be balanced by measured latency and memory,
and the workload provides enough concurrent requests or Prefill chunks to keep
the pipeline occupied. TP remains the preferred baseline when the model fits in
one node, low-concurrency latency is the primary objective, or the stages cannot
be balanced.

For long-context Prefill workloads, consider
[Dynamic Chunked Pipeline Parallel](dynamic_chunk_pipeline_parallel.md), which
adjusts chunk sizes using measured execution time to reduce stage idle time.

## Compatibility and Limitations

- PP cannot be combined with Prefill Context Parallelism (PCP) in the current
  release. See [Context Parallel](context_parallel.md).
- Xlite graph mode is not compatible with PP.
- Sparse KV Cache Offload does not support PP.
- `MooncakeConnectorV1` currently requires the Decode-side PP size to be `1`.
- For cross-node MoE deployments over RoCE, PP and DP cannot currently be
  enabled together because the `MoeDistributeDispatch` communication path does
  not support this combined topology.
- Feature combinations remain model-specific. Check the relevant feature guide
  before production deployment.

## Troubleshooting

| Symptom | Checks and actions |
| --- | --- |
| Startup reports insufficient resources | Confirm that visible NPUs satisfy the selected DP/PP/TP topology and that Ray sees every expected NPU. |
| Startup rejects `VLLM_PP_LAYER_PARTITION` | Check that the entry count equals the PP size and that the sum equals the target hidden-layer count. |
| One PP rank runs out of memory | Check whether it is the last stage or hosts a local drafter. Measure all ranks, then move one target layer at a time. |
| The first request hangs | Retry with smaller sequence and batch limits. Use `--enforce-eager` to determine whether graph capture is involved, and verify inter-rank communication. |
| An MP worker cannot join the head node | Verify that both nodes use the same master address, master port, node count, model, and parallel sizes; that each node rank is unique; and that the master port is reachable. The worker command must include `--headless`. |
| A Ray deployment hangs or misses ranks | Verify identical environments, model paths, communication variables, and Ray resources on every node. |
| A cross-node RoCE MoE deployment fails or hangs after enabling PP and DP | Keep either the PP size or the DP size at `1`; the current `MoeDistributeDispatch` path does not support this combined topology. |
| P/D handshake times out | Compare the P/D connector topology, `prefill.pp_size`, optional `prefill.pp_layer_partition`, and the available `kv_port` range. |

## Related Documentation

- [Supported Models](../support_matrix/supported_models.md)
- [Ray Distributed](../../tutorials/features/ray.md)
- [Dynamic Chunked Pipeline Parallel](dynamic_chunk_pipeline_parallel.md)
- [Speculative Decoding](speculative_decoding.md)
- [PD Disaggregation with Mooncake](../../tutorials/features/pd_disaggregation_mooncake_multi_node.md)
- [Performance Benchmark](../../developer_guide/performance_and_debug/performance_benchmark.md)
- [Service Profiling Guide](../../developer_guide/performance_and_debug/service_profiling_guide.md)
