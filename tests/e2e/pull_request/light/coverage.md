The coverage of e2e light is as follows:

| Feature | Qwen3-0.6B | Qwen3-8B-W8A8 | Qwen3-Embedding | Qwen3.5-0.8B | Qwen3-30B-A3B | Qwen3-VL-30B | DeepSeek-V3-Pruning (Multistream MoE) | DeepSeek-V3.2-W8A8-Pruning (TP/PP/EP) | DeepSeek-V3.2-W8A8-Pruning (PD/SFA/DSA) |
| -------------------- | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| Card count           | 1 | 1 | 1 | 1 | 2 | 2 | 2 | 4 | 4 |
| Dense                | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| MoE                  | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ✅ | ✅ | ✅ |
| Embedding            | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Mamba/SSM            | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Multimodal Reasoning | ❌ | ❌ | ❌ | ✅ | ❌ | ✅ | ❌ | ❌ | ❌ |
| TP                   | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ✅ | ✅ | ❌ |
| PP                   | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ✅ | ❌ |
| EP                   | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ✅ | ❌ |
| EPLB                 | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ |
| Multistream MoE      | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ |
| Full Graph           | ❌ | ✅ | ❌ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ |
| Default FULL_AND_PIECEWISE Graph | ✅ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| PD disaggregation    | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| W8A8                 | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ |
| MTP                  | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Eagle-3              | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| SFA/DSA              | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
