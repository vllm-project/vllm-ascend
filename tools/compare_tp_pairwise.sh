#!/usr/bin/env bash
set -euo pipefail

# Pairwise compare among three tensors:
#   A: tp=1 local
#   B: tp=2 aggregated pre_all_reduce (sum across ranks)
#   C: tp=2 all_reduce (rank0)
#
# Outputs 3 comparison reports:
#   1) A vs C : local vs all_reduce
#   2) A vs B : local vs sum(pre_all_reduce)
#   3) B vs C : sum(pre_all_reduce) vs all_reduce
#
# Example:
#   bash tools/compare_tp_pairwise.sh \
#     --tp1-dir /data/tp1_dump \
#     --tp2-dir /data/tp2_dump \
#     --out-dir /data/compare_out \
#     --mode pt \
#     --layer-ids 0,1,2,3,4 \
#     --start-step 0 \
#     --max-steps 20

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPARE_PY="${SCRIPT_DIR}/compare_tp_dump.py"

TP1_DIR=""
TP2_DIR=""
OUT_DIR=""
MODE="pt"
LAYER_IDS="0,1,2,3,4"
START_STEP="0"
MAX_STEPS="20"
PROJ="o_proj,down_proj"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tp1-dir)
      TP1_DIR="$2"; shift 2 ;;
    --tp2-dir)
      TP2_DIR="$2"; shift 2 ;;
    --out-dir)
      OUT_DIR="$2"; shift 2 ;;
    --mode)
      MODE="$2"; shift 2 ;;
    --layer-ids)
      LAYER_IDS="$2"; shift 2 ;;
    --start-step)
      START_STEP="$2"; shift 2 ;;
    --max-steps)
      MAX_STEPS="$2"; shift 2 ;;
    --proj)
      PROJ="$2"; shift 2 ;;
    -h|--help)
      sed -n '1,80p' "$0"
      exit 0 ;;
    *)
      echo "[error] unknown arg: $1" >&2
      exit 1 ;;
  esac
done

if [[ -z "${TP1_DIR}" || -z "${TP2_DIR}" || -z "${OUT_DIR}" ]]; then
  echo "[error] --tp1-dir / --tp2-dir / --out-dir are required" >&2
  exit 1
fi

mkdir -p "${OUT_DIR}"

COMMON_ARGS=(
  --mode "${MODE}"
  --layer-ids "${LAYER_IDS}"
  --start-step "${START_STEP}"
  --max-steps "${MAX_STEPS}"
  --proj "${PROJ}"
  --ignore-comm-mode-in-key
)

echo "[run] A vs C: tp1 local vs tp2 all_reduce"
python "${COMPARE_PY}" \
  --base "${TP1_DIR}" \
  --cand "${TP2_DIR}" \
  --base-rank 0 \
  --cand-rank 0 \
  --base-comm-mode local \
  --cand-comm-mode all_reduce \
  --summary-csv "${OUT_DIR}/ac_local_vs_allreduce_summary.csv" \
  --details-csv "${OUT_DIR}/ac_local_vs_allreduce_details.csv" \
  "${COMMON_ARGS[@]}"

echo "[run] A vs B: tp1 local vs tp2 sum(pre_all_reduce)"
python "${COMPARE_PY}" \
  --base "${TP1_DIR}" \
  --cand "${TP2_DIR}" \
  --base-rank 0 \
  --cand-rank -1 \
  --aggregate-cand-ranks sum \
  --base-comm-mode local \
  --cand-comm-mode pre_all_reduce \
  --summary-csv "${OUT_DIR}/ab_local_vs_pre_sum_summary.csv" \
  --details-csv "${OUT_DIR}/ab_local_vs_pre_sum_details.csv" \
  "${COMMON_ARGS[@]}"

echo "[run] B vs C: tp2 sum(pre_all_reduce) vs tp2 all_reduce"
python "${COMPARE_PY}" \
  --base "${TP2_DIR}" \
  --cand "${TP2_DIR}" \
  --base-rank -1 \
  --cand-rank 0 \
  --aggregate-base-ranks sum \
  --base-comm-mode pre_all_reduce \
  --cand-comm-mode all_reduce \
  --summary-csv "${OUT_DIR}/bc_pre_sum_vs_allreduce_summary.csv" \
  --details-csv "${OUT_DIR}/bc_pre_sum_vs_allreduce_details.csv" \
  "${COMMON_ARGS[@]}"

echo "[done] outputs at: ${OUT_DIR}"
