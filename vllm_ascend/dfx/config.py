# SPDX-License-Identifier: Apache-2.0
from pydantic import Field, model_validator

from vllm_ascend.config_utils import config


@config
class DfxConfig:
    """Bounded MRV1/MRV2 DFX with separately enabled device probes; off by default."""

    # Master startup switch: overrides all probe options; no recorder when false.
    enabled: bool = False
    output_dir: str | None = None
    run_id: str = ""
    max_records: int = Field(default=512, ge=2, le=65536)
    max_buffer_bytes: int = Field(default=32 * 1024 * 1024, ge=4096, le=1024 * 1024 * 1024)
    max_record_bytes: int = Field(default=1024 * 1024, ge=4096, le=64 * 1024 * 1024)
    max_dumps: int = Field(default=4, ge=1, le=100)
    dump_on_violation: bool = True
    detect_outputs: bool = True
    detect_host_kv: bool = False
    track_block_ownership: bool = False
    audit_transfer_registration: bool = False
    max_tracked_blocks: int = Field(default=65536, ge=1, le=1048576)
    # 0 disables device probes; N > 0 samples every N local executions.
    device_capture_interval: int = Field(default=0, ge=0, le=1000000)
    device_capture_layers: list[str] = Field(default_factory=list)
    device_capture_blocks: int = Field(default=2, ge=1, le=128)
    device_capture_bytes: int = Field(default=256 * 1024, ge=4096, le=64 * 1024 * 1024)
    device_pending_limit: int = Field(default=2, ge=1, le=16)
    reference_trace: str | None = None
    reference_atol: float = Field(default=1e-3, ge=0, allow_inf_nan=False)
    reference_rtol: float = Field(default=1e-3, ge=0, allow_inf_nan=False)
    max_checked_tokens: int = Field(default=4096, ge=1, le=65536)
    max_tracked_requests: int = Field(default=4096, ge=1, le=65536)
    token_history_size: int = Field(default=128, ge=8, le=4096)
    repeat_min_count: int = Field(default=16, ge=2, le=1024)
    repeat_max_period: int = Field(default=4, ge=1, le=32)
    spec_min_proposals: int = Field(default=128, ge=1, le=65536)
    spec_acceptance_floor: float = Field(default=0.1, ge=0, le=1, allow_inf_nan=False)
    token_patterns: list[list[int]] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate(self):
        if self.enabled and (self.output_dir is None or not self.output_dir.strip()):
            raise ValueError("dfx_config.output_dir is required when enabled")
        if self.max_record_bytes > self.max_buffer_bytes:
            raise ValueError("dfx_config.max_record_bytes must not exceed max_buffer_bytes")
        if len(self.run_id) > 128:
            raise ValueError("dfx_config.run_id must contain at most 128 characters")
        if len(self.device_capture_layers) > 256 or len(set(self.device_capture_layers)) != len(
            self.device_capture_layers
        ):
            raise ValueError("device_capture_layers must contain at most 256 unique layer names")
        if self.device_capture_interval and self.device_capture_bytes * 2 + 16384 > self.max_record_bytes:
            raise ValueError("device capture requires max_record_bytes >= 2 * device_capture_bytes + 16384")
        if self.reference_trace is not None and (not self.reference_trace.strip() or not self.device_capture_interval):
            raise ValueError("reference_trace requires a nonempty path and device_capture_interval > 0")
        if self.repeat_min_count * self.repeat_max_period > self.token_history_size:
            raise ValueError("repeat_min_count * repeat_max_period must fit token_history_size")
        if len(self.token_patterns) > 32 or any(
            not pattern or len(pattern) > self.token_history_size or any(token < 0 for token in pattern)
            for pattern in self.token_patterns
        ):
            raise ValueError("token_patterns supports at most 32 nonempty bounded sequences of nonnegative token IDs")
        return self
