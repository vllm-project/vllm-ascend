# Sleep Mode evidence (PR #20481)

Measured on 4x Ascend 910B2C, 2026-11-19, model Qwen3-0.6B.
Config: lightweight + inplace + sleep_mode_extra_cleanup.

| File | Description |
|------|-------------|
| tp1_lightweight_inplace.log | Full TP=1 run (2 sleep/wake + generate) |
| tp2_lightweight_inplace.log | Full TP=2 run (fallback_destroy; wake HCCL timeout) |
| key_excerpts.txt | Key lines (report / freed / suspend / destroy / errors) |

See also ../design_sleep_mode_lifecycle.md section 9.
