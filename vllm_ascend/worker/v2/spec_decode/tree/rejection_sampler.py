import torch
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.metrics.logits import get_num_nans
from vllm.v1.worker.gpu.sample.output import SamplerOutput
from vllm.v1.worker.gpu.spec_decode.rejection_sampler import RejectionSampler

from vllm_ascend.ops.triton.spec_decode.tree.greedy_reject import verify_tree_greedy
from vllm_ascend.worker.v2.spec_decode.tree.acceptance import TreeAcceptance
from vllm_ascend.worker.v2.spec_decode.tree.layout import TreeLayout


class TreeRejectionSampler(RejectionSampler):
    """Greedy tree verification with shared accepted-path result buffers."""

    def __init__(self, sampler, spec_config, device: torch.device, *, async_scheduling: bool):
        super().__init__(sampler, spec_config, device)
        self.device = device
        self.async_scheduling = async_scheduling
        self.acceptance: TreeAcceptance | None = None
        self._result_buf: TreeAcceptance | None = None
        self._sampled_buf: torch.Tensor | None = None
        self._buf_cap_reqs = 0

    def _ensure_bufs(self, num_reqs: int) -> None:
        if self._result_buf is not None and num_reqs <= self._buf_cap_reqs:
            return
        self._buf_cap_reqs = max(num_reqs, self._buf_cap_reqs)
        num_reqs = self._buf_cap_reqs
        device = self.device
        self._result_buf = TreeAcceptance(
            path_node_ids=torch.empty((num_reqs, self.num_speculative_steps), dtype=torch.long, device=device),
            num_sampled=torch.empty(num_reqs, dtype=torch.int32, device=device),
            last_accepted_node_id=torch.empty(num_reqs, dtype=torch.long, device=device),
            active=torch.empty(num_reqs, dtype=torch.bool, device=device),
        )
        self._sampled_buf = torch.empty((num_reqs, self.num_speculative_steps + 1), dtype=torch.long, device=device)

    def __call__(
        self,
        logits: torch.Tensor,
        input_batch: InputBatch,
        draft_logits: torch.Tensor | None = None,
        invalid_drafts: torch.Tensor | None = None,
    ) -> SamplerOutput:
        tree = TreeLayout(
            tokens=input_batch.tree_tokens,
            depths=input_batch.tree_depths,
            parents=input_batch.tree_parents,
            num_nodes=input_batch.tree_num_nodes,
            visibility=input_batch.tree_visibility,
            first_child=input_batch.tree_first_child,
            next_sibling=input_batch.tree_next_sibling,
        )
        num_nans = get_num_nans(logits) if self.sampler.compute_nans else None
        num_reqs = input_batch.num_reqs
        self._ensure_bufs(num_reqs)
        buf = self._result_buf
        acceptance = TreeAcceptance(
            path_node_ids=buf.path_node_ids[:num_reqs],
            num_sampled=buf.num_sampled[:num_reqs],
            last_accepted_node_id=buf.last_accepted_node_id[:num_reqs],
            active=buf.active[:num_reqs],
        )
        cu = input_batch.cu_num_logits[: num_reqs + 1]
        num_logits = cu[1:] - cu[:-1]
        query_cu = input_batch.query_start_loc[: num_reqs + 1]
        is_chunked = (
            input_batch.seq_lens[:num_reqs]
            < self.sampler.req_states.prefill_len.gpu[input_batch.idx_mapping[:num_reqs]]
        )
        acceptance.active.copy_(~is_chunked & (num_logits > 0) & (query_cu[1:] > query_cu[:-1]))
        num_verifiable_draft_nodes = tree.verifiable_draft_node_counts(query_cu, cu)
        sampled = self._sampled_buf[:num_reqs]
        verify_tree_greedy(
            tree.tokens,
            tree.first_child,
            tree.next_sibling,
            logits.argmax(dim=-1),
            cu,
            num_verifiable_draft_nodes,
            sampled,
            acceptance.path_node_ids,
            acceptance.num_sampled,
            acceptance.last_accepted_node_id,
            acceptance.active,
        )
        num_rejected = torch.where(acceptance.active, num_logits - acceptance.num_sampled, 0).to(torch.int32)
        self.acceptance = acceptance
        input_batch.tree_acceptance = acceptance
        # AsyncOutput copies on a separate stream and can outlive this step.
        # The next verification may overwrite the scratch tensors.
        num_sampled = acceptance.num_sampled
        if self.async_scheduling:
            sampled = sampled.clone()
            num_sampled = num_sampled.clone()
        return SamplerOutput(
            sampled_token_ids=sampled,
            logprobs_tensors=None,
            num_nans=num_nans,
            num_sampled=num_sampled,
            num_rejected=num_rejected,
        )
