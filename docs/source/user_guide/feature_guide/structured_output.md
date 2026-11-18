# Structured Output Guide

## Overview

### What is structured output?

LLMs can be unpredictable when you need output in specific formats. Think of asking a model to generate JSON without guidance: it might produce valid text that breaks the JSON specification. **Structured Output (also known as Guided Decoding)** enables LLMs to generate outputs that follow a desired structure while preserving the non-deterministic nature of the system.

In simple terms, structured decoding gives LLMs a "template" to follow. Users provide a schema that "influences" the model output, ensuring compliance with the desired structure.

![structured decoding](./images/structured_output_1.png)

## Usage in vllm-ascend

Currently, the usage of structured output feature in vllm-ascend is totally the same as that in vLLM.

Find more examples and explanations about these usages in [vLLM official document](https://docs.vllm.ai/en/stable/features/structured_outputs/).

## Kimi reasoning with speculative decoding

When a Kimi draft closes reasoning and later opens another think block,
structured-output constraints start at the first completed close marker.
Rejecting the later draft suffix must not undo the JSON constraint on a shorter
accepted prefix. This follows the existing single-boundary policy: it does not
implement multiple unconstrained thinking blocks after JSON generation starts.
Other reasoning parsers retain their upstream behavior.

A temporary compatibility patch backports
[vLLM #68112](https://github.com/vllm-project/vllm/pull/68112) while supported
vLLM versions still need it. It changes boundary selection, not the rejection
sampler or residual sampling kernels. No additional serving flag is required.

The [regression report](https://github.com/linnea-lin-00638949/vllm/blob/bd7becdc9d/review_artifacts/pr68112/generation-regression.md)
includes a fixed-draft before/after comparison and a normal HTTP replay payload.
The reported 100 HTTP replays and ten multi-turn requests used a downstream
adapter with direct decode; they do not validate the full prefill/decode KV
transfer path or this branch's platform-patch startup. Branch-specific CPU
compatibility results and remaining accelerator coverage are listed in the PR.
