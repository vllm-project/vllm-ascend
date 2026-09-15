# Real A3 PD materials

The V1 cases require the placement-registration fix from PR #15878. This PR
carries the identical connector prerequisite so its branch runs independently;
that source diff disappears once #15878 reaches main. DSV4 uses the separate
`MooncakeHybridConnector`, matching its supported compressed-cache deployment.

The reduced checkpoints retain real tensor bytes, full tensor widths and the
source quantization. They establish functional PD contracts, not full-model
accuracy. Generate them once in the runner's model cache before scheduling
`select_tests.py --curated pd-e2e`; preparation time is part of the package budget.

```bash
python tests/e2e/pull_request/pd_materials/prepare.py k3 /models/Kimi-K3-w4a8 /cache/pd/k3
python tests/e2e/pull_request/pd_materials/prepare.py dsv4 /models/DeepSeek-V4-Flash-w8a8-mtp /cache/pd/dsv4
python tests/e2e/pull_request/pd_materials/prepare.py glm /models/GLM-5.2-w8a8 /cache/pd/glm
python tests/e2e/pull_request/pd_materials/prepare.py minimax /models/MiniMax-M3-w8a8 /cache/pd/minimax
```

Use the source variants whose quantization descriptions match the test
assertions. K3 retains 16 experts and the corresponding router rows. GLM retains
layers 0–6, DSV4 retains layers 0–3, and MiniMax retains text layers 0–4.
All reduced materials disable MTP and exclude its weights: removing backbone
layers does not preserve full-model draft acceptance. MTP coverage remains in
the existing full-model Nightly tests. Vision is outside these text-only contracts. The exporter rejects an existing output directory.

Archive each `pd_material_manifest.json` with the source model revision. It
records configuration/index/quantization hashes and every exported file hash.
Keep those files immutable after validation, including tokenizer and processor
files. The same output material is used by the colocated and PD services.

Set `VLLM_MODEL_REDIRECT_PATH` to a JSON mapping containing:

```json
{
  "Qwen/Qwen3-8B": "/models/Qwen3-8B",
  "pd-e2e/Kimi-K3-Text-4layer-16expert-W4A8": "/cache/pd/k3",
  "pd-e2e/DeepSeek-V4-Flash-4layer-W8A8": "/cache/pd/dsv4",
  "pd-e2e/GLM-5.2-7layer-W8A8": "/cache/pd/glm",
  "pd-e2e/MiniMax-M3-5layer-W8A8": "/cache/pd/minimax"
}
```

These `pd-e2e/` keys are local material identifiers, not public download IDs.
Missing material fails explicitly. Use one A3 allocation of four logical NPUs
and run the selected files sequentially; never run independent model files
concurrently on the same allocation.

Stage model files on local disk and verify hashes after staging. Start the
test container with `--init` so terminated worker descendants are reaped. Model server fixtures default HCCL device sockets
to automatic free-port allocation while preserving explicit runner settings.

K3 materials promote the original text configuration to the top level, select `KimiK3ForCausalLM` and remove the `language_model.` wrapper from tensor and quantization keys. The legacy `mla_use_rope=false` default is explicit. Tensor values stay unchanged; this avoids version-dependent vision-tower initialization in a text-only PD test.

Qwen, MiniMax and K3 compare all ten generated token IDs exactly. K3's colocated
reference processes the final known input token through decode, matching the
stateful connector's prefill boundary: it withholds that input token from the
prompt, forces only that known input token, then generates ten unconstrained
answer tokens. Four HTTP clients remain concurrent; K3 schedules one sequence
at a time to align the reference and decoder compute shapes. DSV4 uses the same
known-input alignment and one-sequence scheduling, retaining probability scores
for the ten unconstrained answers. Its P cache retention interval and chunk
budget are 4096, while D prefix caching is disabled, matching the existing
DSV4 P-prefix/D-no-prefix deployment. The 8193-token input still requires two
4096-token P chunks and crosses C128 page boundaries.

Reduced quantized GLM and DSV4 use the existing sequence-parallel precision
test's top-5 distribution limits: maximum absolute logprob delta below 1.0 and
mean delta below 0.15. Both greedy choices must also belong to the other
distribution's top five. After a greedy branch, each remaining position is
replayed with the reference history so every comparison has identical input;
divergent free-running histories are never compared. The same checks first
validate the colocated cold/warm control, then the concurrent PD responses and
the correlated transfer response. Missing candidates, nonfinite values and
distribution drift fail. These functional contracts do not replace full-model
dataset accuracy evaluation.
