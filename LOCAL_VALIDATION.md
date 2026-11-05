# Qwen GDN projection overlap validation

This checkout uses Ascend baseline `934236001` and verified upstream vLLM
`24420b7a341af37a9bc3186e6fc6062512e0aa58`. The overlap path and eligibility
checks match the dev implementation tested on 910C. The reported performance
improvement applies to the dev checkout; rerun A/B on this checkout and its
paired upstream before drawing conclusions for this version.

Run the host tests from this Ascend checkout:

```bash
python3 tests/ut/ops/test_gdn_projection_overlap.py
```

On 910C with the matching CANN and torch_npu build, run:

```bash
python -m pytest -v tests/e2e/pull_request/one_card/test_gdn_projection_overlap.py
python -m pytest -v tests/ut/ops/test_gdn.py tests/ut/ops/test_gdn_projection_overlap.py
python -m pytest -v tests/ut/test_ascend_config.py -k gdn_projection_overlap
```

The projection test checks both streams were selected during capture and
compares the captured outputs bit for bit with the serial outputs while
alternating two graph sizes. It does not exercise the model's packed weights
or recurrent state.

For whole-model A/B, use the same 910C node and BF16 Qwen3.5 model revision.
Use Model Runner V2 and `--compilation-config
'{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY"}'` in both arms. Change only
`gdn_projection_overlap_max_tokens` between `0` and a trial threshold such as
`64`. Check TP1 and the intended production TP, with MTP off and with the
intended MTP settings if supported. Exercise graph buckets both within and
above the threshold. Include Qwen3-Next if deploying the interleaved layout.

- Compare generated token IDs with temperature 0 on fixed prompts, including
  repeated graph replay and transitions between capture buckets. Investigate
  any differences with per-layer projection outputs and logits.
- Warm up both configurations, alternate A/B runs and measure at least three
  times at each target concurrency. Compare TPOT p50/p95, TTFT, output tokens/s,
  and peak NPU memory. Compare enabled and disabled runs with the same
  compilation settings so the difference isolates projection overlap.
- Use a device trace to confirm that BA and QKVZ actually overlap and both
  finish before their outputs are consumed. Host submission to two streams
  alone does not demonstrate parallel device execution or a performance gain.

Hardware tests have not been run locally on this checkout. Use the real PR
number for the release fragment and recheck overlapping community work
before submission.
