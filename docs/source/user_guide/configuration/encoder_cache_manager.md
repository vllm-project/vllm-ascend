# Score Encoder Cache Manager

The Score encoder cache manager uses a CPU cache tier and a score-based policy
to manage multimodal encoder outputs. Selecting the manager enables the feature;
there is no separate `enabled` option.

## Online inference

```bash
vllm serve MODEL \
  --ec-manager-config '{
    "encoder_cache_manager_cls": "vllm_ascend.ec_manager.score_ec_manager.ScoreEncoderCacheManager",
    "manager_config": {
      "cpu_cache_slots": 100000,
      "max_clock": 15,
      "clock_decay_every": 64,
      "watermark": 0.2,
      "promote_percentile": 0.2
    }
  }'
```

## Offline inference

```python
from vllm import LLM

llm = LLM(
    model="MODEL",
    ec_manager_config={
        "encoder_cache_manager_cls": (
            "vllm_ascend.ec_manager.score_ec_manager.ScoreEncoderCacheManager"
        ),
        "manager_config": {
            "cpu_cache_slots": 100000,
            "max_clock": 15,
            "clock_decay_every": 64,
            "watermark": 0.2,
            "promote_percentile": 0.2,
        },
    },
)
```

## Configuration options

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `cpu_cache_slots` | int | `100000` | Maximum CPU cache capacity in encoder embedding slots. |
| `max_clock` | int | `15` | Maximum clock value used by the aging policy. |
| `clock_decay_every` | int | `64` | Number of requests between clock decay steps. |
| `watermark` | float | `0.2` | Target ratio of NPU cache slots to keep free after eviction. |
| `promote_percentile` | float | `0.2` | Score percentile threshold used to promote CPU entries to NPU. |

Custom managers are selected by fully qualified class name. Each manager defines
its own `manager_config` schema and consumes it through `create_manager`.
