# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import torch_npu
from vllm.logger import init_logger

_ACL_FORMAT_ND = 2
_SCRATCH_BLOCK_SIZE = 64
logger = init_logger(__name__)


class CANNBotBf16MlaProlog:
    """Run the BF16 projection and absorption path with the CANNBot V4 operator."""

    def __init__(self, layer_name, wqa, wqb, wkva, wk, gamma_q, gamma_k, eps):
        from mla_prolog_v4 import mla_prolog_v2

        self.op = mla_prolog_v2
        # The SDK's DLPack argument binder expects base tensors.
        weights = (wqa.data, wqb.data, wkva.data)
        self.weights = tuple(
            w if torch_npu.get_npu_format(w) == _ACL_FORMAT_ND else torch_npu.npu_format_cast(w, _ACL_FORMAT_ND)
            for w in weights
        )
        self.wk = wk.contiguous()
        self.gamma_q = gamma_q.data.float()
        self.gamma_k = gamma_k.data.float()
        self.eps = eps
        self.kv_width = wkva.shape[0]
        logger.info("Hy4 CANNBot MLA Prolog V4 active for %s", layer_name)

    def __call__(self, hidden):
        tokens = hidden.shape[0]
        blocks = (tokens + _SCRATCH_BLOCK_SIZE - 1) // _SCRATCH_BLOCK_SIZE
        scratch = hidden.new_empty((blocks, _SCRATCH_BLOCK_SIZE, 1, self.kv_width))
        slots = torch.arange(tokens, dtype=torch.int64, device=hidden.device)
        # Store BF16 scratch rows. Hy4 applies its own MRoPE and C8 quantizer.
        query, _, query_norm = self.op(
            hidden.contiguous(),
            *self.weights,
            self.wk,
            self.gamma_q,
            self.gamma_k,
            scratch,
            slots,
            norm_eps=self.eps,
            rope_mode=self.op.RopeMode.NO_ROPE,
            return_qr=True,
        )
        return query_norm, query, scratch.view(-1, self.kv_width)[:tokens]
