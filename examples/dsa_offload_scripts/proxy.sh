#!/bin/bash
# =============================================================================
# SFA PD-disaggregated proxy / metaserver launcher.
#
# This connector only needs the proxy for the metaserver rendezvous (D posts
# remote_block_ids / remote_host / remote_port; proxy relays them to P). The
# per-layer base-addr + te_rpc_port exchange happens over a ZMQ side channel
# directly between P and D, NOT through the proxy.
#
# The existing layerwise proxy is protocol-agnostic (it transparently forwards
# kv_transfer_params), so we reuse it unchanged:
#   load_balance_proxy_layerwise_server_example.py
#
# GOTCHA: the layerwise proxy forbids --host 0.0.0.0 (it builds the metaserver
# URL from {host}:{port} that D must be able to reach). Use a concrete IP:
# single-box -> 127.0.0.1; multi-host -> the proxy's reachable IP.
#
# START ORDER: 1) D (run_sfa_pd_decode.sh)  2) P (run_sfa_pd_prefill.sh)
#              3) this proxy  4) send requests to the proxy's /v1/completions
# =============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---------------------------- CONFIG (edit me) -------------------------------
PROXY_HOST="141.61.81.142"                  # MUST be a concrete IP (not 0.0.0.0); reachable from D
PROXY_PORT=7122                         # clients send requests here
P_HOST="141.61.81.142"; P_PORT=7120         # matches run_sfa_pd_prefill.sh SERVE_*
D_HOST="141.61.81.153"; D_PORT=7121         # matches run_sfa_pd_decode.sh SERVE_*
# ----------------------------------------------------------------------------

exec python "$HERE/load_balance_proxy_layerwise_server_example.py" \
  --host "$PROXY_HOST" \
  --port "$PROXY_PORT" \
  --prefiller-hosts "$P_HOST" \
  --prefiller-ports "$P_PORT" \
  --decoder-hosts "$D_HOST" \
  --decoder-ports "$D_PORT"
