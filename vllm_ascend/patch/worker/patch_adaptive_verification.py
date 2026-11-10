from vllm.v1.worker.gpu.spec_decode import adaptive_verification

# vLLM 0.33.0 still uses a torch.compile allocator. Keep it eager on Ascend.
adaptive_verification._assign_draft_token_budget_compiled = adaptive_verification._assign_draft_token_budget
