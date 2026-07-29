"""Attach DSA sparse-cache request state from vllm-ascend.

The scheduler tracks a small DSA cache-layout state machine per request.  The
state is Ascend-specific, so keep vLLM's ``Request`` class source unchanged and
initialize the extra fields from a platform-stage patch before requests are
created.
"""

from __future__ import annotations

from functools import wraps

from vllm.v1.request import Request

from vllm_ascend.dsa_sparse.dsa_types import ReqStage

if not getattr(Request, "_dsa_sparse_request_init_patched", False):
    _original_init = Request.__init__

    @wraps(_original_init)
    def _dsa_sparse_request_init(self: Request, *args, **kwargs) -> None:
        _original_init(self, *args, **kwargs)
        self.dsa_req_stage = ReqStage.PREFILL
        self.dsa_next_req_stage = ReqStage.PREFILL
        self.dsa_resident_valid_seq_len = -1
        self.dsa_sparse_budget_tokens = 0
        # 根据 prompt token 数在 scheduler admission 时选择一次，后续
        # decode 不升档、不降档。0 只表示尚未完成 DSA admission 初始化。
        self.dsa_target_resident_budget_tokens = 0

    Request.__init__ = _dsa_sparse_request_init
    Request._dsa_sparse_request_init_patched = True
