# Engram SSD startup activation

SSD DP sharing activates automatically during engine startup. No epoch file,
API health watcher, or manual worker RPC is required.

## Lifecycle

1. Worker construction resets its process-local protocol epoch to zero.
   Profiling and graph capture retain the startup-local behavior.
2. After the engine constructor and profiling-chunk warmup return, all DP
   engine parents rendezvous on their existing Gloo CPU group. This is a
   different group from the workers' SSD P0–P4 protocol group.
3. Each parent submits `activate_engram_ssd_protocol(1)` through its executor
   and checks one matching acknowledgement per local worker.
4. Parents reduce the success status. Any rejection aborts startup; READY is
   sent only after all engines report successful activation.
5. Busy and idle worker steps use the existing P0–P4 protocol. Epoch reads are
   memory-only. A restarted worker starts at zero regardless of old files.

`additional_config.engram_ssd.protocol_activation_timeout_s` defaults to 1800
seconds and must be positive. It is one total budget, measured from the end
of each parent's warmup, for the rendezvous, local RPC and acknowledgement
collective. The existing process-group timeout can impose a tighter bound.
Failures are fatal to that engine startup: recreate the engine and workers;
do not retry on a timed-out process group.

SSD disabled and DP=1 skip activation collectives. SSD DP sharing currently
requires the repository's existing non-elastic topology constraints. This
change does not add elastic membership or independent worker recovery.

## Migration from phase-3 diagnostic launches

Remove `VLLM_ASCEND_ENGRAM_SSD_EPOCH_FILE` and the helper that writes the file
following `/health`. The legacy environment variable is accepted only to
log a startup warning; its path is never opened. Preserve the container's
CANN environment (including its Python module paths).

A stale file cannot activate a new server. If an activation RPC fails or a
rank is missing, startup fails rather than exposing a partly activated
service. Startup logs include each worker's epoch and each engine's complete
activation acknowledgement.

## Performance scope

The previous file-based mechanism re-read the epoch on every lookup. An NFS
metadata stall stopped the submitting CPU and delayed peer collectives,
which was then charged to a later D2H event synchronization. Removing this
file dependency removes that cause of long waits. The hash D2H event wait,
cache lease protection and gather stream ordering remain necessary for the
current host-managed cache; this change does not promise zero synchronization.

## Compatibility hook and removal

The existing consolidated `patch_engine_core.py` wraps `EngineCore.__init__`
after the profiling wrapper, before `EngineCoreProc` creates its input thread
or announces READY. Import of the consolidated spawn target installs the
same hook in engine children. There is no upstream post-warmup/pre-READY
extension point in the pinned vLLM revision. Replace this wrapper with that
extension point when available; retain the control-plane tests.
