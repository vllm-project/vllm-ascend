The coverage of 310p e2e light is as follows:

### Generation Models

| Feature | Qwen3-8B-W8A8SC | Qwen3-8B-W8A8 | Qwen3.5-4B | Qwen3-14B-W8A8SC | Qwen3-30B-A3B | Qwen3.5-35B-A3B | Qwen3-VL-8B |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Card count | 1 | 1 | 1 | 4 | 4 | 4 | 2 |
| Model Type | Dense | Dense | Dense | Dense | MoE | MoE | VL-Dense |
| ACLGraph | ✅ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ |
| Quantization | ✅ (W8A8SC) | ✅ (W8A8) | ❌ | ✅ (W8A8SC) | ❌ | ❌ | ❌ |
| Mamba/SSM | ❌ | ❌ | ✅ | ❌ | ❌ | ✅ | ❌ |

### Pooling Models

| Feature | Qwen3-Embedding-0.6B | multilingual-e5-small | bge-m3 | bge-reranker-v2-m3 | Qwen2.5-1.5B-apeach |
| --- | --- | --- | --- | --- | --- |
| Card count | 1 | 1 | 1 | 1 | 1 |
| Task | Embedding | Embedding | Embedding | Scoring | Classification |
| ACLGraph | ❌ | ❌ | ✅ | ❌ | ❌ |
