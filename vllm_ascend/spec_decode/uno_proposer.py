#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""UNO speculative decoding: the gated-LoRA draft forward.

UNO ("one model") has no draft model.  Every decode cycle runs the *target*
transformer twice, and this file owns the first of the two passes.

Let ``C`` be a request's committed KV frontier -- the position of the last
emitted token, whose KV has not been written yet -- and ``F`` the forward
width.

**Draft forward (here).** ``F`` query rows per request holding
``[seed, noise_1 ... noise_{F-1}]`` at positions ``C .. C+F-1``, attending
causally over the prefix of length ``C``.  Row 0 runs on the base weights and
rows ``1 .. F-1`` run on the UNO LoRA; that per-row split *is* the method (the
adapter is trained as a gated LoRA and only ever applies to noised rows).  The
noise tokens are drawn uniformly from ``[0, vocab_size)``, matching the
checkpoint's ``random_uniform`` noise mode.  Row ``j`` predicts position
``C+j+1``, so the ``F`` sampled tokens are a proposed continuation of positions
``C+1 .. C+F``; ``candidate[0]`` comes from the base row and is the "clean"
token, ``candidate[1:]`` are the diffusion proposals.

**Verify forward (vLLM's, unchanged).** ``F+1`` rows holding
``[seed, candidate_0 ... candidate_{F-1}]`` at ``C .. C+F``, base weights only,
followed by the stock rejection sampler.  Registering the candidates as vLLM's
``num_speculative_tokens = F`` draft tokens makes the acceptance arithmetic come
out exactly right: ``candidate_0`` is drawn from the same distribution the
verify row 0 evaluates, so it is accepted and every step emits at least two
tokens, which is SGLang's ``accept_len = n + 2`` written in vLLM's
``n_accepted + 1`` convention.

That acceptance is mathematically certain but not bit-certain: the two rows are
the same token at the same position over the same prefix on the same weights,
yet they sit in query blocks of different widths (F versus F+1), and the fused
attention kernel tiles by shape. A near-tie in the argmax can therefore flip.
The cost of a flip is one cycle's speedup, never a wrong token -- the rejection
sampler still emits the target's own token -- so the e2e test asserts a
position-0 acceptance rate near 1.0 rather than exactly 1.0.

Compared with SGLang's layout this recomputes the seed row inside the verify
forward (``F+1`` rows instead of ``F``).  That row cannot be skipped here:
vLLM's ``num_computed_tokens`` bookkeeping requires the verify window to start
at the last emitted token, and the KV that the draft forward wrote at ``C``
belongs to a token that acceptance may have replaced.

The draft rows also write LoRA-weight KV into the shared cache at
``C+1 .. C+F-1``.  That is safe only because the next verify forward overwrites
exactly those slots with base-weight KV before anything reads them; see
``_build_draft_slot_mapping``.
"""

import copy
import os
from typing import TYPE_CHECKING, Any

import torch
from vllm import envs
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.forward_context import BatchDescriptor, get_forward_context
from vllm.logger import logger
from vllm.lora.request import LoRARequest
from vllm.transformers_utils.repo_utils import hf_api
from vllm.utils.torch_utils import PIN_MEMORY
from vllm.v1.sample.sampler import _SAMPLING_EPS
from vllm.v1.spec_decode.llm_base_proposer import empty_exponential_noise_like
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
from vllm.v1.utils import CpuGpuBuffer

from vllm_ascend.ascend_forward_context import set_ascend_forward_context
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.utils import AscendCommonAttentionMetadata
from vllm_ascend.compilation.acl_graph import ACLGraphWrapper, update_full_graph_params
from vllm_ascend.sample.rejection_sampler import apply_sampling_constraints
from vllm_ascend.spec_decode.llm_base_proposer import AscendSpecDecodeBaseProposer

if TYPE_CHECKING:
    from vllm.v1.sample.metadata import SamplingMetadata

# The adapter is loaded internally and is never request-selectable, so it owns
# the single LoRA slot. ``lora_int_id`` must be > 0: ``convert_mapping`` treats
# every non-positive id as "base weights only".
UNO_LORA_NAME = "__uno_draft__"
UNO_LORA_INT_ID = 1

# F == 1 would leave no noise rows for the adapter to act on.
MIN_UNO_FORWARD_WIDTH = 2
# ``npu_fused_infer_attention_score`` in TND layout accepts at most this many
# query rows per request; the attention metadata builder asserts the same bound
# as ``decode_threshold = 1 + num_speculative_tokens``.
MAX_FIA_QUERY_ROWS_PER_REQUEST = 16
MAX_UNO_FORWARD_WIDTH = MAX_FIA_QUERY_ROWS_PER_REQUEST - 1

# Row budget for the draft sampler's scratch buffer. The `[rows, vocab_size]`
# float32 noise is the only transient UNO adds on top of `q` itself, and it is
# sliced so that it stays a fixed cost instead of scaling with `max_num_seqs`.
DRAFT_SAMPLE_CHUNK_BYTES = 128 * 1024 * 1024

# How many `[rows, vocab_size]` float32-sized buffers are live at once inside
# `apply_sampling_constraints` when top_k/top_p are set, counted from
# `vllm_ascend/sample/sampler.py::_apply_top_k_top_p_pytorch`: the logits
# themselves, `probs`, `probs_sort`, the int64 sort indices bound to `_` (two
# planes' worth), `cumprob`, and the two bool masks (half a plane together).
# Used only to report the reservation at startup.
DRAFT_SAMPLER_CONSTRAINED_PLANES = 7

# Must match ``patch_uno_speculative_config.UNO_METHOD``; duplicated rather than
# imported because that module is a platform patch and importing it from here
# would invert the patch ordering. ``test_uno_config`` asserts the two agree.
UNO_METHOD = "uno"


def uno_owns_lora_slot(speculative_config) -> bool:
    """True when the LoRA subsystem exists only to serve UNO's gated draft.

    The engine then holds exactly one internal adapter that no request can
    select, which is what lets the runner capture graphs with LoRA switched off
    entirely.
    """
    return speculative_config is not None and speculative_config.method == UNO_METHOD


def resolve_uno_lora_path(lora_path: str, revision: str | None = None) -> str:
    """Resolve a bundled HF adapter before the ordinary LoRA loader sees it.

    vLLM accepts repository IDs but not ``namespace/repo/subdirectory``. Only
    the latter needs a download here; local paths and standalone adapter repos
    retain the ordinary loader's behaviour.
    """
    expanded_path = os.path.expanduser(lora_path)
    if os.path.isabs(expanded_path) or os.path.exists(expanded_path) or lora_path.startswith(("./", "../")):
        return os.path.abspath(expanded_path)

    parts = lora_path.split("/")
    if len(parts) <= 2:
        return lora_path
    if any(part in ("", ".", "..") for part in parts):
        raise ValueError(f"Invalid UNO adapter repository/subdirectory: {lora_path!r}")
    if envs.VLLM_USE_MODELSCOPE:
        raise ValueError(
            "UNO repository subdirectories use Hugging Face; with ModelScope, pass a local adapter directory."
        )

    repo_id = "/".join(parts[:2])
    subdirectory = "/".join(parts[2:])
    snapshot_path = hf_api().snapshot_download(
        repo_id=repo_id,
        revision=revision,
        allow_patterns=[f"{subdirectory}/*"],
    )
    return os.path.join(snapshot_path, *parts[2:])


class AscendUnoProposer:
    """Runs UNO's gated-LoRA draft forward on the target model.

    This deliberately does not subclass :class:`AscendSpecDecodeBaseProposer`.
    That class derives the drafter's attention layers as
    ``all_attn_layers - target_attn_layers`` and then indexes ``[0]``; for UNO
    the difference is empty, so ``load_model`` would raise an ``IndexError``
    far from its cause. UNO reuses the target's layers, KV cache and attention
    backend outright.
    """

    # Reused verbatim: the implementation only touches ``self.backup_next_token_ids``
    # and the input batch, and keeping one copy means UNO tracks any fix made
    # for EAGLE/DFlash.
    prepare_next_token_ids_padded = AscendSpecDecodeBaseProposer.prepare_next_token_ids_padded

    def __init__(self, vllm_config: VllmConfig, device: torch.device, runner=None):
        self.vllm_config = vllm_config
        self.speculative_config = vllm_config.speculative_config
        self.device = device
        self.runner = runner

        assert self.speculative_config is not None
        # F: the draft forward width, i.e. the number of proposals handed to the
        # verify forward. Row 0 of the draft block is the seed, so F-1 rows are
        # noise. The FIA TND layout caps a request's query rows at 16 and the
        # verify window is F+1 wide, which is asserted in the attention metadata
        # builder; validate here so the message names UNO.
        self.forward_width = int(self.speculative_config.num_speculative_tokens)
        if self.forward_width < MIN_UNO_FORWARD_WIDTH:
            # F == 1 leaves no noise rows, so the adapter is never applied and
            # the whole cycle degenerates to two AR forwards emitting two
            # tokens -- correct, but strictly slower than not speculating. Fail
            # rather than ship a silent 1.0x.
            raise ValueError(
                f"UNO requires num_speculative_tokens >= {MIN_UNO_FORWARD_WIDTH} "
                "(the draft block is one seed row plus F-1 noise rows); "
                f"got {self.forward_width}."
            )
        if self.forward_width > MAX_UNO_FORWARD_WIDTH:
            raise ValueError(
                "UNO's verify window is num_speculative_tokens + 1 query rows, and the "
                f"NPU fused-infer-attention TND layout supports at most "
                f"{MAX_FIA_QUERY_ROWS_PER_REQUEST}; got num_speculative_tokens="
                f"{self.forward_width}."
            )

        self.uno_lora_path = resolve_uno_lora_path(
            self.speculative_config.model, revision=self.speculative_config.revision
        )
        self.lora_request = LoRARequest(
            lora_name=UNO_LORA_NAME,
            lora_int_id=UNO_LORA_INT_ID,
            lora_path=self.uno_lora_path,
        )

        self.vocab_size = vllm_config.model_config.get_vocab_size()
        self.max_model_len = vllm_config.model_config.max_model_len
        self.max_num_reqs = vllm_config.scheduler_config.max_num_seqs
        self.max_num_tokens = self.max_num_reqs * self.forward_width

        # Private buffers. The runner's own input_ids/positions/slot_mapping are
        # read again after the target forward, so the draft forward must not
        # alias them.
        self.input_ids = torch.zeros(self.max_num_tokens, dtype=torch.int32, device=device)
        self.positions = torch.zeros(self.max_num_tokens, dtype=torch.int64, device=device)
        self.slot_mapping = torch.zeros(self.max_num_tokens, dtype=torch.int32, device=device)
        self.seq_lens = torch.zeros(self.max_num_reqs, dtype=torch.int32, device=device)
        self.backup_next_token_ids = CpuGpuBuffer(
            self.max_num_reqs,
            dtype=torch.int32,
            pin_memory=PIN_MEMORY,
            device=device,
            with_numpy=True,
        )

        # Per-token LoRA routing vectors, keyed by batch size. Both are pure
        # functions of `num_reqs` (UNO's routing never depends on which requests
        # are in the batch), so they are built once per distinct size.
        self._gated_mapping_cache: dict[int, tuple[int, ...]] = {}
        self._base_mapping_cache: dict[int, tuple[int, ...]] = {}

        self.block_offsets = torch.arange(self.forward_width, dtype=torch.int64, device=device)
        self.query_start_loc = (
            torch.arange(self.max_num_reqs + 1, dtype=torch.int32, device=device) * self.forward_width
        )
        self.query_start_loc_cpu = torch.arange(self.max_num_reqs + 1, dtype=torch.int32) * self.forward_width
        # Inclusive per-request cumulative row count, the shape
        # ``apply_sampling_constraints`` expects for the draft block.
        self.cu_draft_rows = (
            torch.arange(1, self.max_num_reqs + 1, dtype=torch.int32, device=device) * self.forward_width
        )

        # Draft probabilities for the stochastic acceptance path, published to
        # the runner through ``take_last_draft_probs``. Same gate the upstream
        # base proposer uses.
        self._last_draft_probs: torch.Tensor | None = None
        self._enable_draft_probs = (
            self.speculative_config.rejection_sample_method == "standard"
            and self.speculative_config.draft_sample_method == "probabilistic"
        )
        if not self._enable_draft_probs:
            # `draft_sample_method` is forced to "probabilistic" in
            # `_uno_post_init`, so only the rejection method can be the reason.
            # A non-standard rejection sampler decides acceptance without
            # reading q at all, which drops UNO's structural guarantee that the
            # clean token is always accepted.
            logger.warning(
                "UNO is running with rejection_sample_method=%s, so no draft "
                "probabilities are published. UNO's clean token is only accepted "
                "with certainty when the rejection sampler scores it against the "
                "distribution it was drawn from; with a synthetic acceptance rule "
                "a temperature>0 request may emit fewer than 2 tokens per cycle, "
                "i.e. be slower than not speculating.",
                self.speculative_config.rejection_sample_method,
            )

        self._lora_loaded = False
        self._block_size: int | None = None
        self._draft_graph: ACLGraphWrapper | None = None
        self._draft_graph_batch_sizes: set[int] = set()

    def graph_capture_sizes(self, verify_capture_sizes: list[int]) -> list[int]:
        """Separate F-row draft buckets from the runner's F+1-row verifier."""
        if self.vllm_config.compilation_config.cudagraph_mode != CUDAGraphMode.FULL_DECODE_ONLY:
            return []
        if self.speculative_config.enforce_eager:
            return []
        if envs.VLLM_BATCH_INVARIANT:
            raise ValueError(
                "UNO FULL_DECODE_ONLY requires VLLM_BATCH_INVARIANT=0; use PIECEWISE for batch invariance."
            )
        return sorted(
            {
                size // (self.forward_width + 1) * self.forward_width
                for size in verify_capture_sizes
                if size % (self.forward_width + 1) == 0 and 0 < size // (self.forward_width + 1) <= self.max_num_reqs
            }
        )

    @torch.inference_mode()
    def capture_model(self) -> None:
        """Capture after target warmup has finished removing dummy adapters.

        The runner supplies the graph-capture stream and enables capture. Only
        the transformer is recorded; draft noise and sampling stay outside.
        Exact batch sizes avoid dummy requests writing into the shared KV.
        """
        sizes = self.graph_capture_sizes(self.runner.compilation_config.cudagraph_capture_sizes)
        if not sizes:
            return
        self.load_lora_adapter()
        self._draft_graph = ACLGraphWrapper(
            self.get_model(),
            self.vllm_config,
            runtime_mode=CUDAGraphMode.FULL,
            enable_enpu=self.runner.enable_enpu,
        )
        frontier = torch.zeros(self.max_num_reqs, dtype=torch.int32, device=self.device)
        for num_tokens in reversed(sizes):
            num_reqs = num_tokens // self.forward_width
            positions = self._build_draft_positions(frontier[:num_reqs], num_reqs)
            self.input_ids[:num_tokens].zero_()
            # Capture never commits draft KV into a request's cache.
            self.slot_mapping[:num_tokens].fill_(-1)
            cpu_seq_lens = torch.full((num_reqs,), self.forward_width, dtype=torch.int32)
            common = AscendCommonAttentionMetadata(
                query_start_loc=self.query_start_loc[: num_reqs + 1],
                query_start_loc_cpu=self.query_start_loc_cpu[: num_reqs + 1],
                seq_lens=self.seq_lens[:num_reqs],
                seq_lens_cpu=cpu_seq_lens,
                _seq_lens_cpu=cpu_seq_lens,
                num_computed_tokens_cpu=torch.zeros(num_reqs, dtype=torch.int32),
                num_reqs=num_reqs,
                num_actual_tokens=num_tokens,
                max_query_len=self.forward_width,
                max_seq_len=self.forward_width,
                block_table_tensor=self.runner.input_batch.block_table[0].get_device_tensor()[:num_reqs],
                slot_mapping=self.slot_mapping[:num_tokens],
            )
            metadata = self._build_draft_attn_metadata(
                common, frontier[:num_reqs], positions, self.slot_mapping[:num_tokens], num_reqs
            )
            self._set_gated_lora_routing(num_reqs)
            try:
                for _ in range(self.vllm_config.compilation_config.cudagraph_num_of_warmups):
                    self._forward(self.input_ids[:num_tokens], positions, metadata, num_reqs)
                torch.npu.synchronize()
                self._forward(self.input_ids[:num_tokens], positions, metadata, num_reqs, capture=True)
                torch.npu.synchronize()
                self._draft_graph_batch_sizes.add(num_reqs)
            finally:
                self._clear_lora_routing(num_tokens)
        logger.info(
            "UNO captured FULL_DECODE_ONLY draft graphs for request counts %s", sorted(self._draft_graph_batch_sizes)
        )

    def _forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        metadata: dict[str, Any],
        num_reqs: int,
        num_tokens_across_dp: torch.Tensor | None = None,
        *,
        capture: bool = False,
        compute_logits: bool = False,
    ) -> torch.Tensor:
        num_tokens = num_reqs * self.forward_width
        use_graph = capture or num_reqs in self._draft_graph_batch_sizes
        graph_mode = CUDAGraphMode.FULL if use_graph else CUDAGraphMode.NONE
        descriptor = (
            BatchDescriptor(num_tokens, num_reqs, uniform=True, has_lora=True, num_active_loras=1)
            if use_graph
            else None
        )
        with set_ascend_forward_context(
            metadata,
            self.vllm_config,
            num_tokens=num_tokens,
            num_tokens_across_dp=num_tokens_across_dp,
            num_actual_tokens=num_tokens,
            aclgraph_runtime_mode=graph_mode,
            batch_descriptor=descriptor,
            # The shared target's compiled callable was traced with LoRA off.
            # Draft capture must record the original gated transformer instead.
            skip_compiled=True,
            model_instance=self.get_model(),
            is_draft_model=True,
            draft_attn_metadatas=[metadata],
        ):
            context = get_forward_context()
            context.moe_layer_index = 0
            runnable = self._draft_graph if use_graph else self.get_model()
            if use_graph and not capture and self.runner.enable_enpu:
                torch.npu.current_stream().synchronize()
                self._update_graph_params(context, num_tokens, metadata)
            hidden_states = runnable(
                input_ids=input_ids, positions=positions, intermediate_tensors=None, inputs_embeds=None
            )
            if use_graph and not capture and not self.runner.enable_enpu:
                self._update_graph_params(context, num_tokens, metadata)
            if isinstance(hidden_states, tuple):
                hidden_states = hidden_states[0]
            if compute_logits:
                # Keep gated routing and its sampler mask through lm_head.
                return self.get_model().compute_logits(hidden_states[:num_tokens])
            return hidden_states

    def _update_graph_params(self, context, num_tokens: int, metadata: dict[str, Any]) -> None:
        update_full_graph_params(
            self.runner.attn_backend,
            self.runner.update_stream,
            context,
            num_tokens,
            self.vllm_config,
            self.speculative_config,
            draft_attn_metadatas=[metadata],
        )

    # ------------------------------------------------------------------
    # Wiring
    # ------------------------------------------------------------------
    def load_model(self, target_model: torch.nn.Module) -> None:
        """UNO has no draft weights; only the gated adapter has to be resident.

        The adapter is *not* loaded here. ``load_model`` runs before the runner
        wraps the target in LoRA layers, so the manager does not exist yet and a
        model reference captured now would be the pre-LoRA module. The runner
        calls ``load_lora_adapter`` once that wrapping is in place, and again
        after warmup.
        """
        del target_model

    def get_model(self) -> torch.nn.Module:
        # Always resolve through the runner: the LoRA wrapper is installed after
        # ``load_model`` runs.
        return self.runner.get_model()

    def take_last_draft_probs(self) -> torch.Tensor | None:
        probs, self._last_draft_probs = self._last_draft_probs, None
        return probs

    def dummy_run(self, num_tokens: int, num_reqs: int = 0, **kwargs) -> None:
        """Profile-run stand-in.

        UNO's draft forward has the same per-layer cost as the verify forward
        that the runner already profiles, and it allocates no private KV, so
        there is nothing extra to reserve. Keep the DP metadata exchange so idle
        ranks stay in step with ranks that run a real draft.
        """
        del num_reqs, kwargs
        if self.runner is not None:
            self.runner._sync_metadata_across_dp(num_tokens, is_draft_model=True)

    # ------------------------------------------------------------------
    # LoRA routing
    # ------------------------------------------------------------------
    def load_lora_adapter(self) -> None:
        """(Re-)register the gated draft adapter and prove it is resident.

        The runner calls this twice, and both calls are load-bearing:

        * right after the target model is LoRA-wrapped, so that a bad adapter
          path fails at startup rather than inside a decode step;
        * again after warmup, because *every* dummy run ends in
          ``remove_all_adapters()`` (``maybe_setup_dummy_loras``' exit) and
          ``_capture_cudagraphs`` does it once more. Without the second call the
          first ``propose`` would re-read hundreds of megabytes from disk -- or
          from the hub -- in the middle of a decode step, and a transient
          failure there kills the engine core instead of the server start.

        The warmup dummy adapters are created with ``lora_int_id`` 1..max_loras,
        which collides with UNO's own id, so a stale one is evicted rather than
        trusted: ``add_adapter`` is a no-op when the id is already present, and
        silently drafting against a random rank-8 warmup adapter would only show
        up as a poor acceptance rate.
        """
        self.runner._ensure_lora_enabled()
        manager = self.runner.lora_manager
        manager.remove_adapter(UNO_LORA_INT_ID)
        manager.add_adapter(self.lora_request)
        self._verify_adapter_resident()
        self._lora_loaded = True
        logger.info("UNO speculative decoding: loaded the draft LoRA from %s.", self.uno_lora_path)

    def _verify_adapter_resident(self) -> None:
        """Fail loudly if the draft adapter did not actually load.

        A draft forward that silently runs on base weights still produces
        correct output -- only the acceptance rate collapses -- so this is
        checked directly rather than inferred from generated text. It runs once,
        after the first routing install.
        """
        loaded = self.runner.lora_manager.list_adapters()
        if UNO_LORA_INT_ID not in loaded:
            raise RuntimeError(
                "UNO's draft LoRA was not loaded from "
                f"{self.uno_lora_path!r}; the draft forward would run on base "
                "weights and produce no speedup. Loaded adapter ids: "
                f"{sorted(loaded)}."
            )

    def _set_gated_lora_routing(self, num_reqs: int) -> None:
        """Route row 0 of every request to base weights and rows 1..F-1 to UNO.

        ``InputBatch.make_lora_inputs`` cannot express this: it repeats one id
        per request across that request's tokens. The punica layer's mapping is
        genuinely per token, so the routing is built here by hand.
        """
        if not self._lora_loaded:
            # Startup wiring is supposed to have done this; loading here would
            # stall the step, so say why instead.
            raise RuntimeError(
                "UNO's draft LoRA was never loaded. `AscendUnoProposer.load_lora_adapter()` "
                "must run once after the target model is LoRA-wrapped."
            )
        token_lora_mapping = self._gated_mapping_cache.get(num_reqs)
        if token_lora_mapping is None:
            row_ids = (0,) + (UNO_LORA_INT_ID,) * (self.forward_width - 1)
            # Rebuilding this every step is an O(num_reqs * F) host allocation
            # on the decode critical path, and it only ever takes `max_num_seqs`
            # distinct values.
            token_lora_mapping = row_ids * num_reqs
            self._gated_mapping_cache[num_reqs] = token_lora_mapping
        # ``prompt_mapping`` has one entry per *sampled* token, and UNO samples
        # every draft row, so it is the same per-row vector. Sizing it per
        # request instead would leave the sampler index vector shorter than the
        # rows it is narrowed to.
        prompt_lora_mapping = token_lora_mapping
        # Naming the request every step is what keeps the adapter resident:
        # ``set_active_adapters`` loads anything it has not seen, and with
        # ``max_loras=1`` nothing else can evict it.
        self.runner._set_active_loras(
            prompt_lora_mapping,
            token_lora_mapping,
            {self.lora_request},
        )

    def _clear_lora_routing(self, num_tokens: int) -> None:
        """Return the model to base-only weights before the next verify forward.

        ``LoRAModelManager.set_adapter_mapping`` short-circuits on an *equal*
        mapping, which is harmless here: this alternates with the gated mapping,
        so the two are never equal and the install always takes.

        The zero mapping means "base weights", which the punica wrapper answers
        with an early return everywhere, so this costs no device work -- but it
        does have to happen, because the next step's ``set_active_loras`` would
        otherwise be the one to clear it, leaving the draft routing installed
        across anything that runs a forward in between.
        """
        base_mapping = self._base_mapping_cache.get(num_tokens)
        if base_mapping is None:
            base_mapping = (0,) * num_tokens
            self._base_mapping_cache[num_tokens] = base_mapping
        self.runner._set_active_loras(base_mapping, base_mapping, set())

    # ------------------------------------------------------------------
    # Draft forward
    # ------------------------------------------------------------------
    def _require_single_kv_cache_group(self) -> None:
        """UNO reuses the target's KV cache, so it needs exactly one group.

        Checked before anything reads ``kernel_block_sizes`` or the block table:
        on a hybrid-cache model those are per group, and picking group 0 would
        build a slot mapping for the wrong cache rather than fail.
        """
        num_groups = len(self.runner.attn_groups)
        if num_groups != 1:
            raise NotImplementedError(
                "UNO reuses the target's KV cache and expects a single KV cache "
                f"group, found {num_groups}. Hybrid-cache models are not supported yet."
            )

    def _get_block_size(self) -> int:
        if self._block_size is None:
            kernel_block_sizes = self.runner.kernel_block_sizes
            size = kernel_block_sizes[0] if isinstance(kernel_block_sizes, list) else kernel_block_sizes
            self._block_size = int(size[0] if isinstance(size, (list, tuple)) else size)
        return self._block_size

    def _build_draft_input_ids(self, seed_token_ids: torch.Tensor, num_reqs: int) -> torch.Tensor:
        """``[seed, noise x (F-1)]`` per request, flattened.

        The noise is uniform over the full vocabulary, which is what the
        checkpoint was trained with. Every value is a valid token id, so nothing
        downstream can mistake it for the ``-1`` rejection sentinel.
        """
        rows = self.input_ids[: num_reqs * self.forward_width].view(num_reqs, self.forward_width)
        # ``prepare_next_token_ids_padded`` returns one entry per row of the
        # sampled-token tensor, which can be padded past the live requests.
        rows[:, 0] = seed_token_ids[:num_reqs].to(torch.int32)
        if self.forward_width > 1:
            rows[:, 1:] = torch.randint(
                0,
                self.vocab_size,
                (num_reqs, self.forward_width - 1),
                dtype=torch.int32,
                device=self.device,
            )
        return rows.view(-1)

    def _build_draft_positions(self, frontier: torch.Tensor, num_reqs: int) -> torch.Tensor:
        """Positions ``C' .. C'+F-1``, clamped to the last addressable position.

        A request within ``F`` tokens of ``max_model_len`` drafts past the end
        of its own block table, and the block-table gather would then read out
        of range rather than fail cleanly. Clamping (the same guard vLLM's EAGLE
        slot-mapping kernel applies) keeps both the RoPE lookup and the gather
        in bounds.

        The clamp cannot corrupt committed KV: it targets ``max_model_len - 1``,
        and the frontier ``C'`` is never past that, so the slot it lands on is
        at or after the frontier -- scratch that the next verify forward
        rewrites before anything reads it.
        """
        rows = self.positions[: num_reqs * self.forward_width].view(num_reqs, self.forward_width)
        torch.add(frontier.to(torch.int64).unsqueeze(1), self.block_offsets.unsqueeze(0), out=rows)
        rows.clamp_(max=self.max_model_len - 1)
        return rows.view(-1)

    def _build_draft_slot_mapping(
        self,
        positions: torch.Tensor,
        block_table: torch.Tensor,
        num_reqs: int,
    ) -> torch.Tensor:
        """Physical KV slots for the draft rows.

        These are the *same* slots the following verify forward writes at
        ``C+1 .. C+F-1``, which is what makes the LoRA-weight KV the noise rows
        leave behind harmless.

        The reservation is exact, with no slack. At schedule time the request
        had ``num_computed = C`` and ``num_new = F + 1`` (seed plus F
        candidates), and UNO's ``num_lookahead_tokens = F`` extends the
        allocation to position ``C + 2F``. This draft forward runs at the
        post-acceptance frontier ``C' = C + num_emitted <= C + F + 1`` and
        writes rows ``C' .. C'+F-1``, whose maximum is exactly ``C + 2F``.
        That fit depends on the draft block *starting* at the frontier -- a
        variant that followed SGLang more literally, skipping the seed row so
        the block began one position later, would need one more reserved slot.
        Writing past the reservation would not fail: the block table would
        return a stale or zero entry and the KV would land in someone else's
        block.
        """
        block_size = self._get_block_size()
        rows = positions.view(num_reqs, self.forward_width)
        block_numbers = rows // block_size
        # Gather first, cast after: the block table is [num_reqs, max_blocks]
        # and widening the whole table every step would dwarf the F entries
        # actually read. A draft block of at most 15 rows can straddle one
        # 128-token boundary, so the per-row gather is required (a single base
        # slot plus an offset would be wrong across that boundary).
        block_ids = torch.gather(block_table[:num_reqs], 1, block_numbers).to(torch.int64)
        slots = block_ids * block_size + (rows % block_size)
        out = self.slot_mapping[: num_reqs * self.forward_width]
        out.copy_(slots.view(-1).to(torch.int32))
        return out

    def _build_draft_attn_metadata(
        self,
        common_attn_metadata: AscendCommonAttentionMetadata,
        frontier: torch.Tensor,
        positions: torch.Tensor,
        slot_mapping: torch.Tensor,
        num_reqs: int,
    ) -> dict[str, Any]:
        num_tokens = num_reqs * self.forward_width
        draft_metadata = copy.copy(common_attn_metadata)
        draft_metadata.query_start_loc = self.query_start_loc[: num_reqs + 1]
        draft_metadata.query_start_loc_cpu = self.query_start_loc_cpu[: num_reqs + 1]
        draft_metadata.seq_lens = self._build_draft_seq_lens(frontier, num_reqs)
        # The CPU mirrors stay at the verify step's values, and UNO sets
        # ``parallel_drafting`` so the Ascend builder reads the device tensor
        # above instead. Note this does not avoid a device sync -- the builder
        # calls ``seq_lens.tolist()`` either way -- it avoids having to know the
        # frontier on the host in order to write a correct mirror.
        draft_metadata.num_actual_tokens = num_tokens
        draft_metadata.num_input_tokens = num_tokens
        draft_metadata.max_query_len = self.forward_width
        draft_metadata.slot_mapping = slot_mapping
        draft_metadata.positions = positions
        draft_metadata.positions_cpu = None
        draft_metadata.decode_token_per_req = self.forward_width
        # Causal within the block: noise row j must not see rows j+1.. .  This
        # is the regime the adapter was trained in -- SDAR's block-diffusion mask
        # keeps ``q_idx >= kv_idx`` inside the noised block whenever
        # ``use_regular_causal`` is set, and that defaults to True.
        draft_metadata.causal = True
        draft_metadata.attn_state = AscendAttentionState.ChunkedPrefill
        draft_metadata.graph_pad_size = -1

        per_layer_metadata: dict[str, Any] = {}
        for attn_group in self.runner.attn_groups[0]:
            builder = attn_group.get_metadata_builder()
            attn_metadata = builder.build(0, draft_metadata)
            for layer_name in attn_group.layer_names:
                per_layer_metadata[layer_name] = attn_metadata
        return per_layer_metadata

    def _build_draft_seq_lens(self, frontier: torch.Tensor, num_reqs: int) -> torch.Tensor:
        """KV extent seen by the draft rows: ``C' + F``, capped at the model length.

        The block table has ``cdiv(max_model_len, block_size)`` columns and the
        builder hands FIA the whole thing, bounded only by this length, so an
        unclamped ``C' + F`` on a request within F tokens of the length limit
        walks off the end of its own row and into the next request's blocks.
        ``_build_draft_positions`` already clamps the rows themselves, which
        caps the real KV extent at ``max_model_len``.

        FIA runs bottom-right-aligned causal, so where the clamp bites (only the
        final F steps of a request that runs to ``max_model_len``) the draft
        rows see a slightly short context instead of an out-of-range one. That
        costs acceptance on those steps, not correctness: the verify forward is
        untouched and the rejection sampler still decides what is emitted.
        """
        seq_lens = self.seq_lens[:num_reqs]
        torch.add(frontier.to(torch.int32), self.forward_width, out=seq_lens)
        seq_lens.clamp_(max=self.max_model_len)
        return seq_lens

    def _sample_draft_tokens(
        self,
        logits: torch.Tensor,
        sampling_metadata: "SamplingMetadata | None",
        num_reqs: int,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Sample one token per draft row, under the same constraints as ``p``.

        Greedy batches take ``argmax`` and publish no ``q``; the rejection
        sampler then treats the proposal as a point mass, which is exactly right
        because the verify row it is checked against is an ``argmax`` too.

        For a sampling batch the draft rows must be *sampled*, and sampled from
        the distribution the verify forward will actually score them against.
        The rejection sampler builds ``p`` by running the target logits through
        ``apply_sampling_constraints`` -- temperature, then top-k, then top-p --
        so ``q`` is built by calling that same function on the draft logits.
        Using temperature alone (which is what upstream's generic draft samplers
        do) would let the clean token land outside ``p``'s nucleus, where
        ``p(x) = 0`` rejects it outright; UNO's whole speedup rests on that token
        being accepted, and the released checkpoint's own defaults are
        ``top_k=20, top_p=0.95``.

        Rejection sampling stays exact for any ``q``, so a mismatch would cost
        acceptance rather than correctness -- but here it would cost nearly all
        of it.
        """
        if sampling_metadata is None or not self._enable_draft_probs:
            return logits.argmax(dim=-1), None
        if sampling_metadata.all_greedy:
            return logits.argmax(dim=-1), None

        # ``apply_sampling_constraints`` expects an inclusive cumulative count of
        # rows per request; UNO's block is uniformly F wide.
        cu_num_draft_tokens = self.cu_draft_rows[:num_reqs]
        processed = apply_sampling_constraints(logits, cu_num_draft_tokens, sampling_metadata, None)
        if isinstance(processed, tuple):
            processed = processed[0]

        probs = processed.softmax(dim=-1, dtype=torch.float32)
        draft_token_ids = _sample_chunked(probs)
        if not sampling_metadata.all_random:
            # A greedy request inside a mixed batch is verified by argmax, so its
            # draft rows must be argmax too.
            factor = logits.shape[0] // sampling_metadata.temperature.shape[0]
            is_greedy = (sampling_metadata.temperature < _SAMPLING_EPS).repeat_interleave(factor, dim=0)
            draft_token_ids = torch.where(is_greedy, probs.argmax(dim=-1), draft_token_ids)
        return draft_token_ids, probs

    @torch.inference_mode()
    def propose(
        self,
        num_speculative_tokens: int,
        sampled_token_ids: torch.Tensor,
        common_attn_metadata: AscendCommonAttentionMetadata,
        sampling_metadata: "SamplingMetadata | None",
        spec_decode_metadata: SpecDecodeMetadata | None,
    ) -> torch.Tensor:
        """Return ``[num_reqs, F]`` proposals for the next verify forward."""
        if num_speculative_tokens != self.forward_width:
            # UNO's forward width is baked into the LoRA row routing and the
            # verify window; a per-step K would silently change the meaning of
            # the gated rows.
            raise ValueError(
                "UNO does not support dynamic speculative lengths: "
                f"got num_speculative_tokens={num_speculative_tokens}, "
                f"expected {self.forward_width}."
            )

        num_reqs = common_attn_metadata.num_reqs
        num_tokens = num_reqs * self.forward_width
        if num_reqs == 0:
            return torch.empty((0, self.forward_width), dtype=torch.int64, device=self.device)
        self._require_single_kv_cache_group()

        seed_token_ids, valid_sampled_tokens_count = self.prepare_next_token_ids_padded(
            sampled_token_ids,
            self.runner.requests,
            self.runner.input_batch,
            self.runner.discard_request_indices.gpu,
            self.runner.num_discarded_requests,
        )
        self.runner._copy_valid_sampled_token_count(seed_token_ids, valid_sampled_tokens_count)

        # The committed frontier after acceptance. ``common_attn_metadata.seq_lens``
        # is the verify window's end (C + F + 1 on a steady-state decode step),
        # and the scheduler charges back whatever the rejection sampler did not
        # emit, so the difference is the new frontier C'. On the first decode
        # step after a prefill there are no drafts and the frontier is the
        # window end itself.
        frontier = compute_uno_frontier(
            common_attn_metadata.seq_lens[:num_reqs],
            spec_decode_metadata,
            valid_sampled_tokens_count,
            num_reqs,
        )

        input_ids = self._build_draft_input_ids(seed_token_ids, num_reqs)
        positions = self._build_draft_positions(frontier, num_reqs)
        slot_mapping = self._build_draft_slot_mapping(
            positions,
            common_attn_metadata.block_table_tensor,
            num_reqs,
        )
        per_layer_metadata = self._build_draft_attn_metadata(
            common_attn_metadata,
            frontier,
            positions,
            slot_mapping,
            num_reqs,
        )

        # Keep the collective so idle data-parallel ranks stay in step, but the
        # draft buffers are sized for the real batch only, so a padded width has
        # nowhere to go. UNO rejects DP at config time; fail loudly if that ever
        # stops holding rather than reading past the buffers.
        num_input_tokens, num_tokens_across_dp, _ = self.runner._sync_metadata_across_dp(
            num_tokens,
            is_draft_model=True,
        )
        if num_input_tokens != num_tokens:
            raise NotImplementedError(
                "UNO's draft forward cannot run a data-parallel padded batch "
                f"({num_input_tokens} padded vs {num_tokens} real tokens)."
            )

        self._set_gated_lora_routing(num_reqs)
        try:
            logits = self._forward(
                input_ids,
                positions,
                per_layer_metadata,
                num_reqs,
                num_tokens_across_dp,
                compute_logits=True,
            )
        finally:
            # Always hand the model back to the verify forward on base weights,
            # even if the draft forward raised.
            self._clear_lora_routing(num_tokens)

        draft_token_ids, draft_probs = self._sample_draft_tokens(logits, sampling_metadata, num_reqs)
        self._last_draft_probs = None if draft_probs is None else draft_probs.view(num_reqs, self.forward_width, -1)
        return draft_token_ids.view(num_reqs, self.forward_width)


def _sample_chunked(probs: torch.Tensor) -> torch.Tensor:
    """Exponential-noise sampling that does not triple the ``[rows, vocab]`` peak.

    ``probs`` is UNO's ``q``: the rejection sampler reads it after this returns,
    so it must survive unmodified. The upstream helper mutates its first
    argument, so the obvious call is
    ``sample_with_exponential_noise(probs.clone(), noise)`` -- three live
    ``[num_reqs * F, vocab_size]`` float32 tensors at once, which at the
    documented ``--max-num-seqs 256`` with ``F=8`` on a 152k vocabulary is
    3.6 GB of transient the memory profile never saw.

    Reciprocating the noise in place (the branch the helper itself takes when
    the dtypes differ) removes the copy, and slicing the rows bounds the noise
    buffer, leaving ``probs`` plus one small scratch.
    """
    num_rows, vocab_size = probs.shape
    chunk_rows = max(1, DRAFT_SAMPLE_CHUNK_BYTES // (vocab_size * probs.element_size()))
    if chunk_rows >= num_rows:
        return _argmax_over_noise(probs)

    out = torch.empty(num_rows, dtype=torch.int64, device=probs.device)
    for start in range(0, num_rows, chunk_rows):
        end = min(start + chunk_rows, num_rows)
        out[start:end] = _argmax_over_noise(probs[start:end])
    return out


def _argmax_over_noise(probs: torch.Tensor) -> torch.Tensor:
    """``argmax(probs / Exp(1))`` -- the exponential race -- leaving ``probs`` alone.

    The reciprocate-then-multiply form is the one
    ``sample_with_exponential_noise`` itself uses when its two arguments have
    different dtypes; it is spelled out here because the equal-dtype branch it
    would otherwise take divides *into* ``probs``.
    """
    scores = empty_exponential_noise_like(probs, False)
    scores.exponential_()
    scores.reciprocal_()
    scores.mul_(probs)
    return scores.argmax(dim=-1).view(-1)


def compute_uno_frontier(
    verify_window_end: torch.Tensor,
    spec_decode_metadata: SpecDecodeMetadata | None,
    valid_sampled_tokens_count: torch.Tensor,
    num_reqs: int,
) -> torch.Tensor:
    """The committed KV frontier C' after this step's acceptance.

    ``verify_window_end`` is ``num_computed_tokens + num_scheduled_tokens``,
    i.e. ``C + k + 1`` for a request that was scheduled with ``k`` draft
    tokens. The scheduler charges back everything the rejection sampler did not
    emit (``num_rejected = k + 1 - num_emitted``), so

        C' = (C + k + 1) - (k + 1 - num_emitted) = C + num_emitted

    which is the position of the newly emitted last token -- the seed for the
    next draft forward, and the one position in the window whose KV is stale.
    A request with no drafts (the first decode step after a prefill) has
    ``num_rejected = 0`` and its frontier is the window end itself.
    """
    if spec_decode_metadata is None:
        return verify_window_end

    # ``cu_num_draft_tokens`` is an *inclusive* cumulative sum with no leading
    # zero, so the per-request count is its first difference.
    cu_num_draft_tokens = spec_decode_metadata.cu_num_draft_tokens
    num_draft_tokens = torch.cat(
        (
            cu_num_draft_tokens[0:1],
            cu_num_draft_tokens[1:] - cu_num_draft_tokens[:-1],
        )
    )[:num_reqs]
    num_rejected = torch.where(
        num_draft_tokens > 0,
        num_draft_tokens + 1 - valid_sampled_tokens_count[:num_reqs],
        torch.zeros_like(num_draft_tokens),
    )
    return verify_window_end - num_rejected.to(verify_window_end.dtype)


def log_uno_configuration(speculative_config, model_config, max_num_seqs: int | None = None) -> None:
    """One-line summary plus the two warnings worth surfacing at startup."""
    forward_width = int(speculative_config.num_speculative_tokens)
    logger.info(
        "UNO speculative decoding enabled: forward width F=%d (verify window %d rows), draft LoRA=%s",
        forward_width,
        forward_width + 1,
        speculative_config.model,
    )
    # The draft forward's logits and `q` are allocated after the memory profile
    # run, which never executes a draft forward, so they are not covered by
    # `--gpu-memory-utilization`. Report the size rather than let it surface as
    # an out-of-memory at the first large batch.
    if max_num_seqs is not None:
        rows = max_num_seqs * forward_width
        vocab_size = model_config.get_vocab_size()
        plane = rows * vocab_size * 4  # one [rows, vocab_size] float32 buffer
        gib = float(1 << 30)
        logger.info(
            "UNO samples %d x %d draft logits outside the memory profile, which "
            "never runs a draft forward. One such float32 buffer is %.2f GiB; the "
            "peak is about %.1f GiB for a sampled batch with top_k/top_p (%d such "
            "buffers live at once inside the constraint step), %.1f GiB without "
            "them, and %.1f GiB for a purely greedy workload. Lower --max-num-seqs, "
            "or --gpu-memory-utilization, if the KV cache leaves no room for it.",
            rows,
            vocab_size,
            plane / gib,
            (DRAFT_SAMPLER_CONSTRAINED_PLANES * plane + DRAFT_SAMPLE_CHUNK_BYTES) / gib,
            (2 * plane + DRAFT_SAMPLE_CHUNK_BYTES) / gib,
            plane / gib,
            DRAFT_SAMPLER_CONSTRAINED_PLANES,
        )
    # The adapter is trained on blocks of `block_size` noised tokens that attend
    # only within their own block. A wider draft block puts noise rows in
    # positions the checkpoint never saw, which costs acceptance rather than
    # correctness -- so warn, do not fail.
    block_size = getattr(model_config.hf_text_config, "block_size", None)
    if isinstance(block_size, int) and forward_width > block_size:
        logger.warning(
            "UNO forward width F=%d exceeds the checkpoint's diffusion block "
            "size (%d). The draft rows past the block boundary are outside the "
            "trained regime; expect a lower acceptance rate.",
            forward_width,
            block_size,
        )


__all__ = [
    "UNO_LORA_INT_ID",
    "UNO_LORA_NAME",
    "UNO_METHOD",
    "AscendUnoProposer",
    "compute_uno_frontier",
    "log_uno_configuration",
    "uno_owns_lora_slot",
]
