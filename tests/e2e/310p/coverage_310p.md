The coverage of 310p e2e light is as follows:

| Feature | Qwen3-8B-W8A8SC (TP1) | Qwen3-8B-W8A8 (TP1) | Qwen3.5-4B (TP1) | Qwen3-14B-W8A8SC (TP4) | Qwen3-30B-A3B (TP4) | Qwen3.5-35B-A3B (TP4) | Qwen3-VL-8B (TP2) |
|---|---|---|---|---|---|---|---|---|
| Card count | 1 | 1 | 1 | 4 | 4 | 4 | 2 |
| Dense | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ |
| MoE | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | ❌ |
| VL | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| ACLGraph | ✅ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ |
| W8A8 | ✅ (W8A8SC) | ✅ (W8A8) | ❌ | ✅ (W8A8SC) | ❌ | ❌ | ❌ |
| Sharded State | ✅ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ |
| Mamba/SSM | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ |
