#!/bin/bash
# GBSA quantType=5 (full FP8) ATK entry — run on A5/950 with FP8 support.
# Source CANN + custom OPP env before executing.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# Clean previous artifacts (optional)
rm -rf ./result/op_generic_block_sparse_attention_quant/json/all_op_generic_block_sparse_attention_quant.json
rm -rf ./atk_output/*

# Generate cases
atk case -f op_generic_block_sparse_attention_quant.yaml \
    -p generator_genericblocksparseattention_quant.py

# Execute dual-benchmark accuracy
atk task -c ./result/op_generic_block_sparse_attention_quant/json/all_op_generic_block_sparse_attention_quant.json \
    -n nodes.yaml \
    -p aclnn_genericblocksparseattention_quant.py \
    -to 7200

# Re-run a single case (example):
# atk task -c ./result/op_generic_block_sparse_attention_quant/json/all_op_generic_block_sparse_attention_quant.json \
#     -n nodes.yaml -p aclnn_genericblocksparseattention_quant.py -wl [0] -to 7200
