# Mooncake Push with layerwise Store offload

Use the mainline MooncakePushConnector together with Mooncake Store offload.
Install the vLLM version verified by the current vLLM-Ascend mainline.

- P: use `prefill_kv_transfer.json` with `--kv-transfer-config`.
- D: use `decode_kv_transfer.json` with `--kv-transfer-config`.
- Use `load_balance_proxy_layerwise_server_example.py` (D-first), as described in
  the disaggregated prefill guide. Keep PP, PCP and DCP at 1 for initial testing.
  P request preemption is not supported by this version of Push.

The Store waits for both its own save and the predecessor's Push WRITEs before
reusing a slot. A Store prefix hit is not advertised to Push before the layer
has restored and written its cache. Validate GLM-5.3-Flash TP8/EP8 to DP8/TP1 and
native MTP on Ascend before relying on this configuration.
