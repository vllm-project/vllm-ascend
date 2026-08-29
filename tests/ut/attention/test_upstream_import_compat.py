"""Regression tests for upstream vLLM import/signature compatibility.

Upstream vLLM main moved the PCP helper and extended the logits-processor
signature; the Ascend modules must import and dispatch regardless of which
vLLM tree (vendored or open-source) is installed.
"""

import inspect

from tests.ut.base import TestBase


class TestUpstreamImportCompat(TestBase):
    def test_attention_modules_import_and_bind_pcp_helper(self):
        """attention_v1 / mla_v1 / sfa_cp import with the PCP helper bound.

        Before the fix these modules raised ModuleNotFoundError when the
        vendored ``vllm.model_executor.layers.attention.pcp`` module was
        absent (open-source vLLM keeps it at ``vllm.v1.attention.ops.pcp``).
        """
        import vllm_ascend.attention.attention_v1 as attention_v1
        import vllm_ascend.attention.context_parallel.sfa_cp as sfa_cp
        import vllm_ascend.attention.mla_v1 as mla_v1

        for module in (attention_v1, mla_v1, sfa_cp):
            self.assertTrue(callable(module._gather_prefill_cache_inputs))

    def test_get_logits_accepts_skip_gather(self):
        """AscendLogitsProcessor._get_logits takes upstream's skip_gather.

        Upstream passes the argument positionally; a signature mismatch made
        every TP logits call raise TypeError.
        """
        from vllm_ascend.ops.vocab_parallel_embedding import AscendLogitsProcessor

        for method in (
            AscendLogitsProcessor._get_logits,
            AscendLogitsProcessor._get_logits_normal,
            AscendLogitsProcessor._get_logits_lmheadtp,
        ):
            params = inspect.signature(method).parameters
            self.assertIn("skip_gather", params, msg=method.__name__)
