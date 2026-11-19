# Engram SSD automatic preparation implementation plan

Goal: users supply only additional_config.engram_ssd.path; prepare before engine workers and reuse completed data on subsequent starts.

Architecture: a CPU-only artifact module streams local safetensors directly to 288-byte rows. A platform configuration hook resolves the managed artifact, loader and conservative memory/IO defaults before Ascend validation. Existing explicit manifest configurations remain unchanged.

Constraints: preserve checkpoint and existing profiling data; no upstream vLLM changes; no service launches or 221 GB conversion for unit tests; retain current feature branch; do not commit/push on the user's behalf.

Decisions: default whole-service host budget 32 decimal GB, split over TP owners; payload initially 5/8 of that budget, validated with the existing estimator. Artifact identity includes format, checkpoint header and source file metadata (including ctime) and bounded first/middle/last content samples, avoiding a full checkpoint scan on every startup. Incomplete or detectably modified managed output is rebuilt under a filesystem lock, never reused. Automatic file preparation runs only for the new path interface.

## Task 1: artifact preparation and reuse
- [x] Add tests for exact row content including E8M0 extremes, second-start unchanged inode/mtime, checkpoint change, malformed source, truncated output, interrupted writer and concurrent preparers.
- [x] Observe missing-module failure, implement Engram SSD artifact module, pass tests.
- [x] Use per-root flock, bounded reads, free-space check, source-change detection, fsync and atomic publication; log creation/reuse.

## Task 2: configuration initialization
- [x] Add tests for path-only defaults, explicit legacy preservation, conflicts, unsupported topology, low memory budget and repeated resolution.
- [x] Implement configuration resolver and call it from platform.check_and_update_config before Engram validation; add managed config fields.
- [x] Keep default IO settings conservative and respect allowed CPU affinity; avoid changing the inference hot path.

## Task 3: documentation and verification
- [x] Document first-start generation, persistent reuse, internal files, optional whole-service DDR budget, legacy compatibility and identity limitations.
- [x] Add release fragment, run focused and existing SSD regression tests and lint.
- [x] Fresh review, fix substantive issues, and report evidence plus remaining hardware validation limits.

Progress: initial repository clean except pre-existing docs/superpowers and native/.gitkeep. Work is in the existing feat/engram-ssd-offload-v1 checkout, as requested by the ongoing task. No isolated checkout needed for this same-branch evolution.

Verification: 225 passed, 1 skipped across the new artifact/config tests and existing SSD/platform regressions. Ruff check and git diff --check pass. Independent review completed; filesystem coarse timestamps prompted bounded content sampling. Full checkpoint conversion and model-serving performance are not exercised. Release fragment uses pr:null until a PR number exists. Host memory precheck is not cgroup-aware.

Real configuration integration: EngineArgs.create_engine_config with the real model config/quantization JSON and tiny synthetic safetensors passed twice (TP4/DP2), including existing SSDManifest.load validation. Table inode and mtime stayed unchanged. Generation/reuse logs are validated through the same vLLM logger used by ascend_config. No worker or model service was constructed.
