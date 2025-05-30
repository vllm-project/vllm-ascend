# Additional Configuration

addintional configuration is a mechanism provided by vLLM to allow plugins to control inner behavior by their own. vLLM Ascend uses this mechanism to make the project more flexible.

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

The following table lists the additional configuration options available in vLLM Ascend:
| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `torchair_graph_config` | dict | `{}` | The config options for torchair graph mode |
| `ascend_scheduler_config` | dict | `{}` | The config options for ascend scheduler  |
| `expert_tensor_parallel_size` | str | `1` | Expert tensor parallel size the model to use. |

**torchair_graph_config**
| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `enabled` | bool | `False` | Whether to enable torchair graph mode |
| `use_cached_graph` | bool | `False` | Whether to use cached graph |

**ascend_scheduler_config**
| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `enabled` | bool | `False` | Whether to enable ascend scheduler for V1 engine|


### Example

A fuul example of additional configuration is as follows:

```
{
    "torchair_graph_config": {
        "enabled": true,
        "use_cached_graph": true
    },
    "ascend_scheduler_config": {
        "enabled": true
    },
    "expert_tensor_parallel_size": 1
}
```
