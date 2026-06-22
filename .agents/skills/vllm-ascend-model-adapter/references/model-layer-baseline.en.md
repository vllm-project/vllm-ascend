# Model Layer Baseline

This document defines the current capability baseline for layer-by-layer analysis of model internal layer components. The "layer" here refers to the model structure layer, not the adaptation working layer.

The goal is to answer first every time the model is adapted:

- What key layers does this model consist of;
- Whether each layer has reusability capabilities in the current vLLM / vllm-ascend;
- What are the specific differences of each layer?
- Which layers need to be changed and which layers just need to be verified for wiring.

## 1. Currently only required model types

The current skill only requires the use of layer-by-layer templates for two types of models:

1. `dense llm`
2. `moe llm`

Other types (VLM, Whisper, pooling model, etc.) will be expanded later.

## 2. Why do we need to analyze layer by layer?

Many adaptation failures are not "overall not supported", but that a certain layer component is inconsistent with existing assumptions, for example:

- embedding weights are named differently;
- rope / position id are organized in different ways;
- MLP is a gated / fused / parallel variant;
- MoE expert gate / routed expert have different structures;
-Norm has different sharding methods under TP/KV replication;
- lm_head is different from embedding tied / untied;
- The multimodal projector or encoder input format is different.

If we do not break it down layer by layer, it is easy to misjudge the loader problem as an attention or operator problem.

## 3. How to understand the current capability baseline

"Current capability baseline" does not mean that each layer has explicit reference documents, but that it must first be answered with the existing implementation of the warehouse:

- Is there any similar model in vLLM that has already implemented this layer?
- Whether the execution path corresponding to vllm-ascend is currently supported;
- The difference in this layer is a structural difference, a weight naming difference, or a backend difference.

For common layers, the default priority is as follows:

### 3.1 Embedding

Prioritize:

- token embedding naming
- tied embedding / untied lm_head
- vocab resize / padding id / special token processing

High frequency questions:

- The weight names are different;
- lm_head reuses embedding in different ways;
- Multiple embedding branches or additional modality embedding.

### 3.2 Positional / RoPE

Prioritize:

- rope scaling
- rope type / interleaving
- Variants such as mrope / dynamic rope / longrope
- How position ids are organized

High frequency questions:

- The rope parameter position is different;
- qk rope dimensions are inconsistent with existing assumptions;
- The position generation logic is inconsistent with the existing model runner assumptions.

### 3.3 Attention

This is the most complex layer, but not the only one to look at. Detailed baseline:

- `references/attention-v1-analysis.md`

### 3.4 MLP / FFN

Prioritize:

- gated MLP / SwiGLU / GeGLU / parallel MLP
- Weight division method
- Naming of intermediate dimensions

High frequency questions:

- gate/up/down proj naming and loading methods are different;
- Inconsistent with TP shard rules;
- The fused linear assumption does not hold.

### 3.5 MoE

Prioritize:

- number of experts
- gate/router structure
- Does shared expert exist?
- routed expert weight layout
- Are EP/flashcomm/MoE communication paths relevant?

Detailed baseline:

- `references/moe-fused-analysis.md`

High frequency questions:

- Non-standard expert naming;
- Gate/router weight mapping is inconsistent;
- EP-only paths behave differently than TP-only paths.

### 3.6 Norm

Prioritize:

- RMSNorm / LayerNorm / other variations
- q_norm / k_norm / post-attn norm / pre-ffn norm
- TP / replicated KV heads under shard rules

High frequency questions:

- Norm weights require local shards instead of ordinary uniform shards;
- qk norm is strongly related to head topology.

### 3.7 LM Head / Output Head

Prioritize:

- whether tied to embedding
- Whether there are additional logits scale / soft cap / output norm
- Whether the generated header is a separate module

High frequency questions:

- The sharing relationship between embedding and lm_head is not wired correctly;
- Output layer remap is incorrect.

## 4. Select template by model type

First select a template based on `CLASSIFICATION_SUMMARY`.

### 4.1 Dense LLM Template

Applicable conditions:

- non-multimodal
- non-encoder-decoder
- Non-MoE
- The main body is the standard decoder-only dense LLM

Fixed template:

```markdown
## Layer-by-Layer Compatibility Matrix

| Layer | Current capability | Model requirement | Gap | Adaptation plan |
| --- | --- | --- | --- | --- |
| embedding |  |  |  |  |
| positional/rope |  |  |  |  |
| attention |  |  |  |  |
| mlp/ffn |  |  |  |  |
| norm |  |  |  |  |
| lm_head/output |  |  |  |  |
```

### 4.2 MoE LLM Template

Applicable conditions:

- decoder-only LLM
- Contains routed experts / shared experts / MoE router

Fixed template:

```markdown
## Layer-by-Layer Compatibility Matrix

| Layer | Current capability | Model requirement | Gap | Adaptation plan |
| --- | --- | --- | --- | --- |
| embedding |  |  |  |  |
| positional/rope |  |  |  |  |
| attention |  |  |  |  |
| moe router/gate |  |  |  |  |
| moe experts |  |  |  |  |
| shared expert / residual mlp | N/A or ... |  |  |  |
| norm |  |  |  |  |
| lm_head/output |  |  |  |  |
```

## 5. Fill in the requirements

Regardless of which template you use, the following requirements are met:

- `Current capability` must refer to the existing implementation or existing path in the current warehouse. Do not write "supported" in a generic way.
- `Model requirement` must come from `config.json`, modeling code, checkpoint key, or running evidence.
- `Gap` must be a specific difference, do not write "may be incompatible".
- `Adaptation plan` It must be made clear:
- Reuse existing paths;
- Change model adapter / loader;
- Change processor;
- Change framework wiring;
- Verify that Ascend backend exists;
- or stop the upgrade.

### 5.1 Dense LLM Answer at least these points

- Whether embedding and lm_head are tied
- rope type and scaling
- attention subtype
- Is MLP a normal FFN or a gated FFN?
- norm type and shard risk

### 5.2 MoE LLM additionally must answer these points

- Implementation and weight naming of router/gate
- expert weight layout
- Whether there is a shared expert
- EP/TP path differences
- Does the MoE layer also have special communication assumptions other than attention?

## 6. Relationship with other baseline documents

- attention row priority reference `attention-v1-analysis.md`
- The quantized layer also refers to `quantization-baseline.md`
- When the framework behaves abnormally, combine it with `framework-integration-baseline.md`
- If expanded to multi-modality in the future, then pick up `processor-and-multimodal-baseline.md`

## 7. Suggestions on usage order

Recommended order:

1. Do `Layer-by-Layer Compatibility Matrix` first
2. Do it again `Model Adapter Gap Analysis`
3. Expand each special gap analysis of attention / multimodal / operator / framework / quantization as needed.

In this way, the overall structure diagram of the model can be established first, and then complex special projects can be entered.
