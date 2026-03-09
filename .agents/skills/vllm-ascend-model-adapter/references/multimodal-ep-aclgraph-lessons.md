# Multimodal + EP + ACLGraph Lessons

This note captures practical patterns that repeatedly matter for VL checkpoints on Ascend.

## 1) Out-of-box feature expectation

Try best to validate key features by default:

- ACLGraph
- MTP
- multimodal (if model supports VL)
- EP (MoE models only)
- flashcomm1 (MoE models only)

If any feature fails, keep logs and explain the reason in the final report.
For non-MoE models, EP/flashcomm1 should be marked not-applicable.

## 2) Validate in this order

1. Single text request success (`/v1/models` + `/v1/chat/completions`).
2. Single text+image request success.
3. Graph evidence (`Replaying aclgraph`) when graph mode is expected.
4. Capacity baseline: `128k + bs16`.
5. Concurrency expansion if needed (`32/64` suggested).

## 3) EP + graph startup expectations

- Startup latency is much higher than eager due to:
    - compile warmup
    - graph capture rounds
    - multimodal encoder profiling
- Do not treat slow startup as failure unless logs show hard errors.

## 4) Always distinguish two max lengths

- **Theoretical max**: from model config (`max_position_embeddings`).
- **Practical max**: largest value that actually starts and serves on current hardware + TP/EP settings.

Report both values explicitly.

## 5) Multimodal testing with temporary layer reduction

- Reducing `num_hidden_layers` can speed smoke tests.
- This does **not** remove ViT structure itself.
- Still require one full-layer validation before final sign-off.

## 6) Feature-status semantics

Use four categories:

- ✅ supported and verified
- ❌ framework-level unsupported
- ⚠️ checkpoint missing (weights/config do not provide feature)
- N/A not-applicable (for example EP/flashcomm1 on non-MoE models)

Typical examples:

- flashcomm1 on non-MoE VL models is often N/A or ❌ depending on framework gate.
- MTP may be ⚠️ checkpoint missing even if framework has code paths.

## 7) Keep docs and defaults aligned with latest success path

- If EP+graph is validated and requested/expected, it should be the default runbook path.
- Eager mode should be documented as fallback/troubleshooting only.
