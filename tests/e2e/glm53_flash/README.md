# GLM-5.3-Flash real-weight logprob regression

This is a numerical regression gate for a real five-layer projection of
`GLM-5.3-Flash-W8A8-0916`. It is **not** a full-model quality evaluation. It
does not use dummy weights, generate its own reference during a CI run, or
skip when weights/goldens are missing.

## Fixed coverage

- A3, two logical NPUs, TP2/EP2/DP1; MRV1 and hybrid KV cache manager.
- Original text layers 0–4: three KDA/Dense, one sparse-attention/MoE, and
  one KDA/MoE layer. Keep all experts, tensor widths, quantization payloads,
  tokenizer, global weights and non-text weights. Disable MTP and vision input.
- Eager, sequential BS1, chunked prefill with a 512-token budget, prefix
  caching and async scheduling disabled, one GiB KV cache per rank.
- Eleven fixed token-ID inputs, 32 output tokens each, two replay rounds.
  Lengths 2051/2052 straddle the first discarded complete KPool (pool size
  four, 512 selected complete pools); 4097 covers chunked sparse prefill.
- Actual `curl /v1/completions`, greedy sampling, raw top-five logprobs.
  Exact generated IDs and candidate membership; absolute tolerance `1e-4`,
  relative tolerance zero. No comparison of decoded token strings.

The test-only worker asserts the effective configuration in **both** ranks
after real model loading. Its machine-readable log record is required for
success. The usual worker and model-runner implementations are unchanged.
The deterministic environment also pins `GLOO_SOCKET_IFNAME=lo`: this is a
single-host test, and its CPU process groups must not depend on external DNS
resolution of a container hostname. NPU collective transport is unchanged.

## Provisioning and immutable provenance

An infrastructure owner must expose the reviewed checkpoint in the existing
CI model-cache mount, read-only. Set the repository Actions variable
`GLM53_FLASH_SOURCE_DIR` to its **in-container** path. Locally, export the same
variable. It is a test-only path, not a credential; it has no implicit default.
No private download credentials or personal network paths are committed.

`fixtures/source.json` pins the source files consumed by this projection:
metadata, tokenizer, the source index and all shards containing any retained
tensor. Discarded-only shards are not read. Provisioning may retain the full
checkpoint; online preparation never needs its discarded-only shards.

For an explicitly reviewed new checkpoint, author a new manifest outside CI:

```bash
python -m tests.e2e.glm53_flash.checkpoint \
  --source-dir "$GLM53_FLASH_SOURCE_DIR" \
  --manifest /path/to/new-source.json --create-source-manifest
```

Review the manifest before adopting it; do not regenerate it in response to
a checksum failure. The source index, per-layer configuration, quantization
description and safetensors payload bounds are checked. No dequantization or
requantization is performed. The projection streams original bytes with an
8 MiB copy buffer, publishes atomically, and refuses to overwrite a different
or corrupt existing projection.

```bash
python -m tests.e2e.glm53_flash.checkpoint \
  --source-dir "$GLM53_FLASH_SOURCE_DIR" \
  --manifest tests/e2e/glm53_flash/fixtures/source.json \
  --output-dir /path/to/reduced-5
```

The PR test caches the projection under pytest's cache directory, keyed by
source identity and recipe. Cold preparation verifies consumed source files;
warm reuse verifies every derived file. The golden additionally pins the
entire projection manifest, input fixture and serving configuration.

## Golden capture and review

Use a known-good main checkout, its `.github/vllm-main-verified.commit`
upstream vLLM revision, a fixed CANN/container stack and the **same A3 hardware
class as the precision runner**. Never capture from an unreviewed candidate PR.

Only when deliberately updating the tokenizer/input corpus, author a new
prompt fixture using `fixtures/build_prompts.py --source-dir ... --output ...`.
The regression test consumes committed IDs, without tokenizing or reformatting
chat prompts at runtime.

```bash
python -m tests.e2e.glm53_flash.capture \
  --model /path/to/reduced-5 \
  --artifacts /path/to/new-capture \
  --candidate /path/to/new-golden-candidate.json \
  --ascend-commit <verified-main-sha> \
  --vllm-commit <verified-upstream-sha> \
  --image <container-image-and-digest>
```

Capture performs three independent server starts and two rounds per start.
The first observed outputs form a **candidate**, and all subsequent outputs
must pass the same strict comparator. Failure leaves logs and raw responses
but does not publish a candidate. Successful capture still requires review of
source identity, parameters, hardware, package versions and numerical results
before placing it in `fixtures/golden.json`. CI never invokes capture or
updates the golden. A stack upgrade is recorded and tested, not automatically
accepted by refreshing the baseline.

## Running and diagnosing

```bash
pytest -sv tests/e2e/pull_request/two_card/test_glm53_flash_logprobs.py
pytest -q tests/ut/glm53_flash
```

For CPU helper development without installing vLLM/NPU dependencies:

```bash
python -m pytest --confcutdir=tests/ut/glm53_flash tests/ut/glm53_flash
```

The selected-tests workflow uploads `tests/outputs/glm53_flash/**`: effective
worker configuration, environment, server log, each curl request/response,
stderr, normalized outputs, timing and first-divergence reports. A server
failure, timeout, invalid JSON, missing logprobs, non-finite value, token
divergence, candidate-set change or numerical mismatch fails the test.

The precision list routes this file to the existing A3-560T partition.
It does not make the test run on every PR: existing coverage/AST selection
and full-suite selection still apply. The source cache and reviewed golden
must be provisioned **before enabling this gate upstream**. Record measured
cold/warm duration and peak memory at acceptance. The initial 600-second
per-file scheduling budget follows the selector's default and is explicitly
provisional, not a claimed measurement.
