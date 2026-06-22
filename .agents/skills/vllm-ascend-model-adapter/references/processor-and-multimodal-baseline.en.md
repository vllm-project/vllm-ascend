# Processor And Multimodal Baseline

This document defines the current baseline for processor behavior, multimodal input wiring, and vision-language entry paths.

## What this layer focuses on

It focuses on:

- whether tokenizer and processor code is compatible with the current `transformers` API
- how multimodal input is normalized into vLLM-consumable form
- whether vision encoders, image tokens, and embeddings enter the correct path
- whether a failure belongs to the processor layer, the model-wiring layer, or the attention/backend layer

## Current baseline

The current skill assumes:

- VLMs always need processor analysis; adapter-only analysis is not enough
- `skip_tensor_conversion` mismatches are a common processor-layer failure
- text-only isolation is useful for diagnosis, but not a final success criterion
- multimodal models should be validated with both text-only and text+image requests
- multimodal attention/backend issues should be separated from processor issues when possible

## Typical evidence

- `config.json`
- `processor_config.json`
- `preprocessor_config.json`
- `auto_map`
- `processing_*.py`
- `tokenizer_config.json`
- text-only startup result
- the first text+image error stack
