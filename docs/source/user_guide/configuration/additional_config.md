# Additional Configuration

Additional configuration is a mechanism provided by vLLM to allow plugins to control inner behavior by themselves. VLLM Ascend uses this mechanism to make the project more flexible.

## How to use

With either online mode or offline mode, users can use additional configuration. Take Qwen3 as an example:

**Online mode**:

```bash
vllm serve Qwen/Qwen3-8B --additional-config='{"config_key":"config_value"}'
```

**Offline mode**:

```python
from vllm import LLM

LLM(model="Qwen/Qwen3-8B", additional_config={"config_key":"config_value"})
```

### Configuration options

The following table lists additional configuration options available in vLLM Ascend:

| Name                                | Type | Default | Description                                                                                               |
|-------------------------------------|------|---------|-----------------------------------------------------------------------------------------------------------|
| `xlite_graph_config`                | dict | `{}`    | Configuration options for xlite graph mode                                                                |
| `weight_prefetch_config`            | dict | `{}`    | Configuration options for weight prefetch                                                                 |
| `finegrained_tp_config`             | dict | `{}`    | Configuration options for module tensor parallelism                                                       |
| `ascend_compilation_config`         | dict | `{}`    | Configuration options for ascend compilation                                                              |
| `refresh`                           | bool | `false` | Whether to refresh global Ascend configuration content. This is usually used by rlhf or ut/e2e test case. |
| `dump_config_path`                  | str  | `None`  | Configuration file path for msprobe dump(eager mode).                                                     |
| `enable_async_exponential`          | bool | `False` | Whether to enable async exponential overlap. To enable async exponential, set this config to True.        |
| `enable_shared_expert_dp`           | bool | `False` | When the expert is shared in DP, it delivers better performance but consumes more memory. Currently only DeepSeek series models are supported. |
| `multistream_overlap_shared_expert` | bool | `False` | Whether to enable multistream shared expert. This option only takes effect on MoE models with shared experts. |
| `multistream_overlap_gate`          | bool | `False` | Whether to enable multistream overlap gate. This option only takes effect on MoE models with shared experts.  |
| `recompute_scheduler_enable`        | bool | `False` | Whether to enable recompute scheduler.                                                                    |
| `enable_cpu_binding`                | bool | `False` | Whether to enable CPU binding.                                                                            |
| `SLO_limits_for_dynamic_batch`      | int  | `-1`    | SLO limits for dynamic batch. This is new scheduler to support dynamic feature                            |
| `enable_npugraph_ex`                | bool | `False` | Whether to enable npugraph ex graph mode.                                                                 |
| `pa_shape_list`                     | list | `[]`    | The custom shape list of page attention ops.                                                              |
| `dynamic_eplb`                      | bool | `False` | Whether to enable dynamic EPLB.                                                                           |
| `expert_map_path`                   | str  | `None`  | When using expert load balancing for an MoE model, an expert map path needs to be passed in.              |  
| `num_iterations_eplb_update`        | int  | `400`   | Forward iterations when EPLB begins.                                                                      |
| `gate_eplb`                         | bool | `False` | Whether to enable EPLB only once.                                                                         |
| `num_wait_worker_iterations`        | int  | `30`    | The forward iterations when the EPLB worker will finish CPU tasks. In our test default value 30 can cover most cases. |
| `expert_map_record_path`            | str  | `None`  | Save the expert load calculation results to a new expert table in the specified directory.                |
| `init_redundancy_expert`            | int  | `0`     | Specify redundant experts during initialization.                                                          |

The details of each configuration option are as follows:

**xlite_graph_config**

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `enabled` | bool | `False` | Whether to enable xlite graph mode. Currently only Llama, Qwen dense series models, and Qwen3-vl are supported. |
| `full_mode` | bool | `False` | Whether to enable xlite for both the prefill and decode stages. By default, xlite is only enabled for the decode stage. |

**weight_prefetch_config**

| Name             | Type | Default                                                     | Description                        |
|------------------|------|-------------------------------------------------------------|------------------------------------|
| `enabled`        | bool | `False`                                                     | Whether to enable weight prefetch. |
| `prefetch_ratio` | dict | `{"attn": {"qkv": 1.0, "o": 1.0}, "moe": {"gate_up": 0.8}}` | Prefetch ratio of each weight.     |

**finegrained_tp_config**

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `lmhead_tensor_parallel_size`    | int  | `0` | The custom tensor parallel size of lmhead.    |
| `oproj_tensor_parallel_size`     | int  | `0` | The custom tensor parallel size of oproj.     |
| `embedding_tensor_parallel_size` | int  | `0` | The custom tensor parallel size of embedding. |
| `mlp_tensor_parallel_size`       | int  | `0` | The custom tensor parallel size of mlp.       |

**ascend_compilation_config**

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `fuse_norm_quant`  | bool | `True` | Whether to enable fuse_norm_quant pass. |
| `fuse_qknorm_rope` | bool | `False` | Whether to enable fuse_qknorm_rope pass. It's set to True by default when Triton is installed. |

### Example

An example of additional configuration is as follows:

```
{
    "weight_prefetch_config": {
        "enabled": True,
        "prefetch_ratio": {
            "attn": {
                "qkv": 1.0,
                "o": 1.0,
            },
            "moe": {
                "gate_up": 0.8
            }
        },
    },
    "finegrained_tp_config": {
        "lmhead_tensor_parallel_size": 8,
        "oproj_tensor_parallel_size": 8,
        "embedding_tensor_parallel_size": 8,
        "mlp_tensor_parallel_size": 8,
    },
    "multistream_overlap_shared_expert": True,
    "refresh": False,
}
```
