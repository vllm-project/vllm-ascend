"""Legacy placeholder for removed v2 attention utils patch.

The previous patch targeted private helpers that no longer exist in
`vllm_ascend.worker.v2.attn_utils`. The patch is not imported by the active
patch registry, so keeping this file as a no-op avoids stale mypy failures
without changing runtime behavior.
"""
