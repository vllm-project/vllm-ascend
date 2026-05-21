# MiniMax-M2.7

## Introduction

MiniMax-M2.7 is part of the MiniMax-M2 series, MiniMax's flagship large language model family. It shares the same architecture and deployment requirements as MiniMax-M2.5.

This document confirms that **vLLM-Ascend supports MiniMax-M2.7** and provides deployment guidance.

## Supported Features

Refer to [supported features](../../user_guide/support_matrix/supported_models.md) to get the model's supported feature matrix.

Refer to [feature guide](../../user_guide/feature_guide/index.md) to get the feature's configuration.

## Deployment

MiniMax-M2.7 follows the same deployment procedures as MiniMax-M2.5. Please refer to the [MiniMax-M2.5 documentation](MiniMax-M2.5.md) for detailed instructions on:

- Environment preparation
- Docker setup
- Online inference configuration
- Multi-NPU deployment
- Performance tuning

### Quick Start

For MiniMax-M2.7 deployment, use the same configuration as MiniMax-M2.5:

```{code-block} bash
vllm serve /path/to/weight/MiniMax-M2.7 \
    --served-model-name "MiniMax-M2.7" \
    --host 0.0.0.0 \
    --port 8000 \
    --trust-remote-code \
    --quantization ascend \
    --tensor-parallel-size 4 \
    --data-parallel-size 4 \
    --max-num-seqs 48 \
    --max-model-len 40690 \
    --gpu-memory-utilization 0.85
```

### Key Points

1. **Architecture**: MiniMax-M2.7 uses the same `minimax_m2` architecture as MiniMax-M2.5
2. **Configuration**: All MiniMax-M2 series models share the same vLLM-Ascend patches and optimizations
3. **Hardware**: Recommended hardware setup is identical to MiniMax-M2.5 (Atlas 800 A3 or 2× Atlas 800I A2)

## Verify the Service

Test with an OpenAI-compatible client:

```{code-block} python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="na")

resp = client.chat.completions.create(
    model="MiniMax-M2.7",
    messages=[{"role": "user", "content": "你好，请介绍一下你自己。"}],
    max_tokens=256,
)
print(resp.choices[0].message.content)
```

## FAQ

- **Q: Is MiniMax-M2.7 supported?**

  A: Yes, vLLM-Ascend supports all MiniMax-M2 series models, including M2.5 and M2.7.

- **Q: What's the difference between MiniMax-M2.5 and M2.7?**

  A: They are different versions in the MiniMax-M2 series. They share the same architecture and deployment requirements.

- **Q: Can I use the same configuration for M2.5 and M2.7?**

  A: Yes, all MiniMax-M2 series models use the same configuration and optimizations in vLLM-Ascend.

## See Also

- [MiniMax-M2.5 Documentation](MiniMax-M2.5.md) - Detailed deployment guide
- [Supported Models](../../user_guide/support_matrix/supported_models.md) - Full model support matrix
