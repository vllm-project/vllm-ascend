# Engram SSD activation implementation plan

Spec: ../specs/2026-11-13-engram-ssd-activation.md

Global constraints: use the existing feat/engram-ssd-offload checkout; preserve
user changes and keep a before snapshot. Python is ../.venv/bin/python. No
commit of pre-existing changes. Reuse existing engine CPU group and RPC.

## Task 1: Memory-only worker epoch
- Write failing tests for fresh lifetime, monotonic/invalid activation and
  absence of filesystem access even with the legacy env set.
- Replace file-backed epoch with memory state, reset at worker construction,
  return the acknowledged epoch from worker RPC; remove runtime file binding.
- Run targeted tests. Expected: RED before implementation, GREEN afterward.

## Task 2: Ordered engine activation
- Write real two-process Gloo tests for delayed warmup, all-rank ack, partial
  RPC error, malformed ack and missing-rank timeout.
- Add activation coordinator with one deadline, ready rendezvous, local RPC,
  global status agreement; wire after EngineCore init before READY using the
  consolidated Ascend patch. Add positive timeout configuration and patch docs.
- Run targeted and existing patch/protocol tests. Expected: GREEN; failed or
  missing peers never yield a successful startup.

## Task 3: NPU regression and review
- Start TP4/DP2 without epoch file/watcher, FULL_DECODE_ONLY, real packed SSD.
- Exercise idle DP and 32 concurrent 48-token requests; record latency and
  complete outputs. Restart and repeat. Expected: all workers activated before
  READY, successful requests, no old hundreds-of-ms epoch-file stalls.
- Review delta versus before snapshot, run relevant checks, write deployment
  instructions and results, stop only owned test processes, download report.
