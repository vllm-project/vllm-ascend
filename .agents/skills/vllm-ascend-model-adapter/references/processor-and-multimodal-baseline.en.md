# Processor And Multimodal Baseline

This document defines the current baseline for processor behavior, multimodal input wiring, and vision-language entry paths.

## What this layer focuses on

It focuses on:

- whether tokenizer and processor code is compatible with the current `transformers` API
- how multimodal input is normalized into a vLLM-consumable form
- whether vision encoders, image tokens, and embeddings enter the correct path
- whether a failure belongs to the processor layer, model wiring, or the attention/backend layer

## Current baseline

- VLMs always need processor analysis; adapter-only analysis is not enough.
- `skip_tensor_conversion` mismatch is a common processor-layer failure.
- Text-only isolation is useful for diagnosis, but not a final success criterion.
- Multimodal models should be validated with both text-only and text-plus-image requests.
