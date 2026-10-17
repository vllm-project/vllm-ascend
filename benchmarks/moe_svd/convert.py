"""Offline ModelSlim to packed INT4 low-rank MoE checkpoint conversion."""

from vllm_ascend.quantization.moe_svd_checkpoint import main

if __name__ == "__main__":
    main()
