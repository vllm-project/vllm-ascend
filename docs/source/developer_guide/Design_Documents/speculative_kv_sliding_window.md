# DSpark draft KV sliding window

Set `additional_config.draft_window_size` to a positive integer to configure
the Qwen3 DSpark draft attention layers with a physical sliding-window cache.
Without an explicit value the checkpoint's draft configuration is preserved.

For an uncompressed MLA/indexer target and regular DSpark sliding-window
attention, the native layout keeps target and draft cache groups separate.
Each layer retains its native page geometry and the draft uses the upstream
sliding-window cache manager, so old draft blocks can be recycled.

This native-layout specialization applies when the hybrid cache manager is
enabled and PCP/DCP are both one. Other model/cache layouts continue through
the existing planners. Target-model cache retention is unchanged.

The producer and consumer must agree on draft cache geometry for PD transfer.
The connector uses each group's actual native block stride.
