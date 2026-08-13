# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import torch

try:
    from vllm.v1.spec_decode.suffix_proposer_gpu import SuffixProposerGPU
except ModuleNotFoundError as exc:
    if exc.name != "vllm.v1.spec_decode.suffix_proposer_gpu":
        raise

    class SuffixProposerGPU:  # type: ignore[no-redef]
        """Compatibility placeholder for vLLM versions without suffix_gpu."""

        def __init__(self, *args, **kwargs) -> None:
            raise ImportError("suffix_gpu requires a vLLM version containing vllm-project/vllm#52097")


class AscendSuffixProposerNPU(SuffixProposerGPU):
    """Initial Ascend adapter for the upstream device-state suffix proposer.

    The SuffixGPU package currently enables fused Triton kernels only for CUDA.
    On NPU it therefore uses the package's device-agnostic PyTorch fallback.
    This provides the functional path while NPU kernels and ACL Graph support
    are developed separately.
    """

    def __init__(self, vllm_config, device: torch.device, runner=None):
        super().__init__(vllm_config, device=device, runner=runner)

    def capture_draft_graph(self, token_ids_gpu: torch.Tensor) -> None:
        # Upstream capture uses torch.cuda.CUDAGraph directly. Keep the first
        # functional Ascend implementation eager until ACL Graph is supported.
        self._graph_failed = True

    def _warmup(self, token_ids_gpu: torch.Tensor) -> None:
        # The upstream warmup only JIT-compiles CUDA Triton kernels. SuffixGPU
        # selects its pure PyTorch fallback for NPU tensors, so no warmup is
        # required for the functional path.
        self._warmed_up = True

    @torch.inference_mode()
    def dummy_run(
        self,
        num_tokens=1,
        with_prefill=None,
        in_graph_capturing=None,
        num_reqs=None,
        num_tokens_across_dp=None,
        aclgraph_runtime_mode=None,
        batch_descriptor=None,
        dummy_compute_logits=lambda hidden_states: None,
        is_profile=False,
    ) -> None:
        pass
