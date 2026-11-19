# Engram SSD protocol activation

Approved scope: replace the diagnostic epoch-file mechanism with an ordered,
process-local activation lifecycle. Existing SSD work is uncommitted and must
be preserved. No upstream vLLM change or D2H synchronization removal.

Evidence: phase-3/runs/20261113-sync-trace-110657. NFS epoch reads introduced
hundreds of milliseconds of CPU stalls and delayed peer TP collectives.
Moving just the file to tmpfs reduced a 32 x 48-token round from 51.43 s to
2.674 s. The formal solution must remove file reads from inference entirely.

Contract:
1. A fresh worker starts at epoch 0 for model loading, profiling and capture.
2. After EngineCore initialization (including profiling chunk warmup), all DP
   engine parents rendezvous on their existing CPU group, separate from the
   worker protocol group. No engine sends READY until activation succeeds.
3. Each parent uses the ordered executor RPC to activate every local worker.
   Validate every acknowledgement; globally agree success before returning.
4. Activation and both startup collectives have a finite, configurable budget.
   Partial failures abort startup; no best-effort fallback or retry on a
   poisoned group. A failed engine is recreated with fresh workers.
5. Epoch reads are memory-only, monotonic within one worker lifetime. Legacy
   epoch-file env is ignored with a startup migration warning. No file I/O.
6. Feature-off and DP=1 require no activation collective. Existing idle-DP
   participation, cache leases and NPU stream dependencies stay unchanged.
7. Hook is an Ascend compatibility patch in the consolidated engine entry.
   Remove it when upstream exposes a post-warmup/pre-READY startup hook.

Validation: CPU epoch invariants, real multiprocess Gloo startup ordering,
partial failure and missing-rank bounds; existing protocol tests; real TP4/DP2
FULL_DECODE_ONLY startup without an epoch watcher, idle and concurrent traffic,
restart, and latency comparison with the recorded tmpfs control.
