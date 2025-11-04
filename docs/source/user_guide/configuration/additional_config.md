# Additional Configuration

Additional configuration is a mechanism provided by vLLM to allow plugins to control inner behavior by their own. vLLM Ascend uses this mechanism to make the project more flexible.

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

| Name                                | Type | Default | Description                                                                                                                                   |
|-------------------------------------|------|---------|-----------------------------------------------------------------------------------------------------------------------------------------------|
| `torchair_graph_config`             | dict | `{}`    | Configuration options for torchair graph mode                                                                                                    |
| `ascend_scheduler_config`           | dict | `{}`    | Configuration options for ascend scheduler                                                                                                       |
| `weight_prefetch_config`            | dict | `{}`    | Configuration options for weight prefetch                                                                                                        |
| `refresh`                           | bool | `false` | Whether to refresh global Ascend configuration content. This is usually used by rlhf or ut/e2e test case.                                      |
| `expert_map_path`                   | str  | `None`  | When using expert load balancing for an MoE model, an expert map path needs to be passed in.                                                 |
| `kv_cache_dtype`                    | str  | `None`  | When using the KV cache quantization method, KV cache dtype needs to be set, currently only int8 is supported.                                |
| `enable_shared_expert_dp`           | bool | `False` | When the expert is shared in DP, it delivers better performance but consumes more memory. Currently only DeepSeek series models are supported. |
| `lmhead_tensor_parallel_size`       | int  | `None`  | The custom tensor parallel size of lmhead.                                                                                                    |
| `oproj_tensor_parallel_size`        | int  | `None`  | The custom tensor parallel size of oproj.                                                                                                     |
| `multistream_overlap_shared_expert` | bool | `False` | Whether to enable multistream shared expert. This option only takes effects on MoE models with shared experts.                                |
| `dynamic_eplb`                      | bool | `False` | Whether to enable dynamic EPLB.                                                                                                                |
| `num_iterations_eplb_update`        | int  | `400`   | Forward iterations when EPLB begins.                                                                                                      |
| `gate_eplb`                         | bool | `False` | Whether to enable EPLB only once.                                                                                                              |
| `num_wait_worker_iterations`        | int  | `30`    | The  forward iterations when the EPLB worker will finish CPU tasks. In our test default value 30 can cover most cases.                           |
| `expert_map_record_path`            | str  | `None`  | When dynamic EPLB is completed, save the current expert load heatmap to the specified path.                                                   |
| `init_redundancy_expert`            | int  | `0`     | Specify redundant experts during initialization.                                                                                              |

The details of each configuration option are as follows:

**torchair_graph_config**

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `enabled` | bool | `False` | Whether to enable torchair graph mode. Currently only DeepSeek series models and PanguProMoE are supported. |
| `mode` | str | `None` | When using reduce-overhead mode for torchair, it needs to be set. |
| `enable_multistream_mla`| bool | `False` | Whether to put vector operators of MLA to another stream. This option only takes effect on models using MLA (for example, DeepSeek). |
| `enable_view_optimize` | bool | `True` | Whether to enable torchair view optimization. |
| `enable_frozen_parameter` | bool | `True` | Whether to fix the memory address of weights during inference to reduce the input address refresh time during graph execution. |
| `use_cached_graph` | bool | `False` | Whether to use cached graph. |
| `graph_batch_sizes` | list[int] | `[]` | The batch size for torchair graph cache. |
| `graph_batch_sizes_init` | bool | `False` | Init graph batch size dynamically if `graph_batch_sizes` is empty. |
| `enable_kv_nz`| bool | `False` | Whether to enable KV Cache NZ layout. This option only takes effect on models using MLA (for example, DeepSeek). |
| `enable_super_kernel` | bool | `False` | Whether to enable super kernel to fuse operators in deepseek moe layers. This option only takes effects on moe models using dynamic w8a8 quantization.|

**ascend_scheduler_config**

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `enabled` | bool | `False` | Whether to enable ascend scheduler for V1 engine.|
| `enable_pd_transfer` | bool | `False` | Whether to enable P-D transfer. When it is enabled, decode is started only when prefill of all requests is done. This option only takes effect on offline inference. |
| `decode_max_num_seqs` | int | `0` | Whether to change max_num_seqs of decode phase when P-D transfer is enabled. This option only takes effect when enable_pd_transfer is True. |
| `max_long_partial_prefills` | Union[int, float] | `float('inf')` | The maximum number of prompts longer than long_prefill_token_threshold that will be prefilled concurrently. |
| `long_prefill_token_threshold` | Union[int, float] | `float('inf')` | a request is considered long if the prompt is longer than this number of tokens. |

ascend_scheduler_config also support the options from [vllm scheduler config](https://docs.vllm.ai/en/stable/api/vllm/config.html#vllm.config.SchedulerConfig). For example, you can add `enable_chunked_prefill: True` to ascend_scheduler_config as well.

**weight_prefetch_config**

| Name             | Type | Default                                                     | Description                        |
|------------------|------|-------------------------------------------------------------|------------------------------------|
| `enabled`        | bool | `False`                                                     | Whether to enable weight prefetch. |
| `prefetch_ratio` | dict | `{"attn": {"qkv": 1.0, "o": 1.0}, "moe": {"gate_up": 0.8}}` | Prefetch ratio of each weights.    |

### Example

An example of additional configuration is as follows:

```
{
    "torchair_graph_config": {
        "enabled": True,
        "use_cached_graph": True,
        "graph_batch_sizes": [1, 2, 4, 8],
        "graph_batch_sizes_init": False,
        "enable_kv_nz": False
    },
    "ascend_scheduler_config": {
        "enabled": True,
        "enable_chunked_prefill": True,
        "max_long_partial_prefills": 1,
        "long_prefill_token_threshold": 4096,
    },
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
    "multistream_overlap_shared_expert": True,
    "refresh": False,
}
```
