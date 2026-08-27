"""GLM-5.2 MegaKernel integration for vLLM-Ascend.

When ``ENABLE_MEGAKERNEL=1`` and the model is ``glm_moe_dsa``
(``GlmMoeDsaForCausalLM``), the forward is routed through the MegaKernel
(``blockrt.models.glm_5_2.Glm52MegaKernel``) instead of the vLLM/Ascend
op-by-op path.

This mirrors the sglang_npu ``qwen3_moe.py`` integration:

* weights are loaded once into the MegaKernel model
  (``blockrt.models.glm_5_2.weight_loader.load_glm5_2_weights``; HF param
  names match vLLM checkpoints verbatim);
* the MegaKernel reuses the paged KV / DSA indexer caches already allocated by
  vLLM-Ascend (fp8 packed SFA C8 layout), so prefill writes and decode reads
  the same physical cache; vLLM drives scheduling and provides metadata
  (slot mapping, block tables, seq lens);
* persistent per-batch-size device buffers keep tensor addresses stable so
  the MegaKernel DAG graph-cache key does not change between decode steps
  (same trick as the sglang adapter's ``kv_seq_len_cache``/``q_seq_len_cache``).

Usage:

    ENABLE_MEGAKERNEL=1 vllm serve /path/to/GLM-5.2-10L-W4A8C8-A5 \
        --tensor-parallel-size 1 ...

Notes:
* TP must be 1 in this milestone (``Glm52MegaKernel(tp_size=1)``).
* ``MEGA_KERNEL_HOME`` / megakernel python path must be on ``PYTHONPATH``
  (``blockrt`` import) and the megakernel CANN registration must be done
  (``megakernel/install.sh``).
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from vllm.forward_context import get_forward_context
from vllm.config import VllmConfig
from vllm.model_executor.models.deepseek_v2 import GlmMoeDsaForCausalLM
from vllm.sequence import IntermediateTensors

_ENABLED = os.environ.get("ENABLE_MEGAKERNEL", "0") in ("1", "true", "True")

# Packed-KV layout constants are resolved from vLLM config at state init
# (cache_config.block_size / scheduler_config.max_num_batched_tokens /
# model_config.max_model_len) — see _MegaKernelGLM52State.__init__.


def _mega_enabled() -> bool:
    return _ENABLED


def _model_args_from_hf_config(config: Any):
    """Build a blockrt ModelArgsGLM5 from the vLLM HF config."""
    from blockrt.models.glm_5_2.model_args import ModelArgsGLM5

    args = ModelArgsGLM5()
    for key in (
        "num_hidden_layers",
        "hidden_size",
        "num_attention_heads",
        "num_key_value_heads",
        "q_lora_rank",
        "kv_lora_rank",
        "qk_head_dim",
        "qk_nope_head_dim",
        "qk_rope_head_dim",
        "v_head_dim",
        "intermediate_size",
        "moe_intermediate_size",
        "n_routed_experts",
        "n_shared_experts",
        "num_experts_per_tok",
        "first_k_dense_replace",
        "index_topk",
        "index_n_heads",
        "index_head_dim",
        "max_position_embeddings",
        "vocab_size",
        "rms_norm_eps",
    ):
        val = getattr(config, key, None)
        if val is not None:
            setattr(args, key, int(val) if isinstance(val, (int, float)) else val)
    args.quantized = True
    # The reduced MXFP checkpoint carries no MTP head weights; build the
    # blockrt model without the MTP module (MTP decode is unsupported here).
    args.num_nextn_predict_layers = 0
    return args


def _extract_dsa_metadata(attn_metadata: Any) -> AscendDSAMetadata:
    """Normalize v1 ForwardContext attn_metadata to one AscendDSAMetadata."""
    if isinstance(attn_metadata, dict):
        # v1: {layer_name: AttentionMetadata}; all layers share the batch shape
        for meta in attn_metadata.values():
            if meta is not None:
                print(
                    "[MegaKernel] attn metadata type: "
                    f"{type(meta).__name__}",
                    flush=True,
                )
                return meta
        raise RuntimeError("MegaKernel: no attention metadata in forward context")
    if isinstance(attn_metadata, (list, tuple)):
        return _extract_dsa_metadata(attn_metadata[0])
    return attn_metadata


def _has_usable_metadata(attn_meta: Any) -> bool:
    """True if attn_meta exposes the fields the mega kernel bridge needs."""
    if attn_meta is None:
        return False
    if getattr(attn_meta, "num_actual_tokens", 0) <= 0:
        return False
    req = getattr(attn_meta, "req_metadata", None) or attn_meta
    return (
        getattr(req, "block_table", None) is not None
        and getattr(req, "seq_lens", None) is not None
        and getattr(req, "slot_mapping", None) is not None
    )


def _cache_tensors(cache: Any) -> tuple[torch.Tensor, ...]:
    """Normalize a vLLM bound KV cache into a tuple of tensors."""
    if cache is None:
        return ()
    if isinstance(cache, (tuple, list)):
        return tuple(t for t in cache if t is not None)
    return (cache,)


class _MegaKernelGLM52State:
    """Lazily-initialized MegaKernel runtime for one GLM-5.2 model."""

    def __init__(self, vllm_config: VllmConfig, vllm_model: Any):
        self.vllm_config = vllm_config
        hf_config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config

        # Prefer vLLM/vLLM-Ascend-owned configuration over MegaKernel-specific
        # magic numbers. ``packed_kv_dim`` is the same value AscendSFA uses for
        # its packed C8 KV layout.
        self.block_size = int(getattr(cache_config, "block_size", 128))
        self.rope_head_dim = int(getattr(hf_config, "qk_rope_head_dim", 64))
        self.index_head_dim = int(getattr(hf_config, "index_head_dim", 128))
        self.packed_kv_dim = self._resolve_packed_kv_dim(hf_config)
        self.max_tokens_default = self._resolve_max_tokens_default(
            vllm_config
        )

        # The MegaKernel can selectively load layers 0..num_hidden_layers-1
        # from the FULL checkpoint even when vLLM serves a reduced one.
        self.model_dir = os.environ.get(
            "MEGA_WEIGHTS_DIR", vllm_config.model_config.model
        )
        self.model_args = _model_args_from_hf_config(
            vllm_config.model_config.hf_config
        )

        num_blocks = self._num_blocks()
        self.num_blocks = num_blocks
        self.k_cache, self.index_k_buffer, self.index_k_scale_buffer = (
            self._borrow_vllm_kv_cache(vllm_model)
        )
        print(
            "[MegaKernel] reusing vLLM KV cache: "
            f"layers={len(self.k_cache)} packed_blocks={self.k_cache[0].shape[0]} "
            f"packed_dim={self.k_cache[0].shape[-1]}",
            flush=True,
        )
        self.index_cos_sin_cache = torch.empty(
            (self.max_tokens_default, self.index_head_dim),
            dtype=torch.bfloat16,
            device="npu",
        )
        self.mla_cos_cache = torch.empty(
            (self.max_tokens_default, self.rope_head_dim),
            dtype=torch.bfloat16,
            device="npu",
        )
        self.mla_sin_cache = torch.empty(
            (self.max_tokens_default, self.rope_head_dim),
            dtype=torch.bfloat16,
            device="npu",
        )
        # Per-batch-size persistent metadata buffers (stable addresses).
        self.meta_buffers: Dict[int, Dict[str, torch.Tensor]] = {}

        # Tensor parallelism: derive rank/world from the vLLM parallel config
        # so the blockrt weights/model match the serving TP degree.
        pc = vllm_config.parallel_config
        self.tp_size = int(getattr(pc, "tensor_parallel_size", 1) or 1)
        if self.tp_size > 1:
            from vllm.distributed import get_tensor_model_parallel_rank
            self.tp_rank = int(get_tensor_model_parallel_rank())
            from blockrt.dist.utils import init_shemm
            _ms = int(os.environ.get("MEGA_SHMEM_SIZE", str(1 << 33)))
            print(f"[MegaKernel] init_shemm rank={self.tp_rank} world={self.tp_size} mem_size={_ms}", flush=True)
            init_shemm(
                rank=self.tp_rank,
                world_size=self.tp_size,
                mem_size=_ms,
                ip_port=os.environ.get("MEGA_SHMEM_IP_PORT", "tcp://127.0.0.1:8666"),
            )
        else:
            self.tp_rank = 0

        # Weights + MegaKernel model.
        from blockrt.models.glm_5_2.glm5_2 import Glm52MegaKernel
        from blockrt.models.glm_5_2.weight_loader import load_glm5_2_weights

        weight_dict = load_glm5_2_weights(
            self.model_dir,
            model_args=self.model_args,
            device="npu",
            tp_size=self.tp_size,
            tp_rank=self.tp_rank,
        )
        self.mega = Glm52MegaKernel(
            self.model_args, weight_dict,
            tp_size=self.tp_size, tp_rank=self.tp_rank,
        )

    @staticmethod
    def _resolve_packed_kv_dim(self, hf_config: Any) -> int:
        """Resolve the packed MLA KV head dimension from vLLM-Ascend.

        Uses the vLLM cache block size (``cache_config.block_size``, stored on
        ``self.block_size`` at init) as the packed-KV tile width, matching the
        layout AscendSFA uses for the C8 packed cache.
        """
        try:
            from vllm_ascend.attention.utils import (
                get_sfa_qsfa_packed_head_dim,
            )

            return int(
                get_sfa_qsfa_packed_head_dim(
                    int(getattr(hf_config, "kv_lora_rank", 512)),
                    int(getattr(hf_config, "qk_rope_head_dim", 64)),
                    self.block_size,
                )
            )
        except Exception as exc:  # noqa: BLE001
            print(
                "[MegaKernel] falling back to model-config packed cache "
                f"dimension: {exc}",
                flush=True,
            )
        kv_lora_rank = int(getattr(hf_config, "kv_lora_rank", 512))
        rope_head_dim = int(getattr(hf_config, "qk_rope_head_dim", 64))
        scale_dim = (
            kv_lora_rank // self.block_size
        ) * 4
        return kv_lora_rank + rope_head_dim * 2 + scale_dim

    @staticmethod
    def _resolve_max_tokens_default(vllm_config: VllmConfig) -> int:
        env_value = os.environ.get("MEGA_MAX_TOKENS")
        if env_value is not None:
            return max(int(env_value), 1)
        scheduler = getattr(vllm_config, "scheduler_config", None)
        max_num_batched_tokens = getattr(
            scheduler, "max_num_batched_tokens", None
        )
        if max_num_batched_tokens is not None:
            return max(int(max_num_batched_tokens), 1)
        model_cfg = getattr(vllm_config, "model_config", None)
        max_model_len = getattr(model_cfg, "max_model_len", None)
        if max_model_len is not None:
            return max(int(max_model_len), 1)
        return 8192

    def _borrow_vllm_kv_cache(
        self, vllm_model: Any
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        """Read the KV cache views already bound by vLLM.

        AscendConfig now forces ``enable_sparse_sfa_c8`` /
        ``enable_sparse_li_c8`` in MegaKernel mode, so every MLA layer's bound
        main cache has the same ``[num_blocks, block_size, 1, 656]`` packed
        FP8 layout as blockrt's ``rms_norm_rope_scatter`` and
        ``kv_quant_sparse_flash_attention`` kernels. Indexer layers likewise
        expose ``(k, scale)`` caches that can be flattened into blockrt's
        ``[num_blocks * block_size, head_dim]`` / ``[..., 1]`` views.
        """
        layers = getattr(getattr(vllm_model, "model", None), "layers", None)
        if layers is None or len(layers) < self.model_args.num_hidden_layers:
            raise RuntimeError(
                "MegaKernel: cannot locate vLLM decoder layers for shared KV cache"
            )

        main_caches: list[torch.Tensor] = []
        indexer_caches: list[torch.Tensor] = []
        indexer_scale_caches: list[torch.Tensor] = []

        for layer_id in range(self.model_args.num_hidden_layers):
            layer = layers[layer_id]
            self_attn = getattr(layer, "self_attn", None)
            if self_attn is None:
                raise RuntimeError(
                    f"MegaKernel: layer {layer_id} has no self_attn"
                )

            # vLLM 0.27.1 DeepSeek/GLM MLA wrapper keeps the bound cache on
            # ``self_attn.mla_attn``. Fall back to the older ``attn`` name.
            mla_attn = getattr(self_attn, "mla_attn", None) or getattr(
                self_attn, "attn", None
            )
            if mla_attn is None:
                raise RuntimeError(
                    f"MegaKernel: layer {layer_id} has no MLA attention wrapper"
                )

            main = _cache_tensors(getattr(mla_attn, "kv_cache", None))
            if len(main) != 1:
                raise RuntimeError(
                    "MegaKernel shared KV cache requires packed SFA C8 main cache; "
                    f"layer {layer_id} got {len(main)} cache tensor(s). "
                    "Check that AscendConfig.enable_sparse_sfa_c8 is enabled."
                )
            packed = main[0]
            if packed.ndim != 4 or packed.shape[-1] != self.packed_kv_dim:
                raise RuntimeError(
                    "MegaKernel shared KV cache has unexpected shape "
                    f"{tuple(packed.shape)} for layer {layer_id}; expected "
                    f"[num_blocks, {self.block_size}, 1, "
                    f"{self.packed_kv_dim}]."
                )
            main_caches.append(packed)

            indexer = getattr(self_attn, "indexer", None)
            indexer_cache_module = getattr(indexer, "k_cache", None)
            indexer_cache = _cache_tensors(
                getattr(indexer_cache_module, "kv_cache", None)
            )

            if indexer is None or len(indexer_cache) == 0:
                # Shared DSA layers reuse the previous layer's top-k indices
                # and never touch an indexer cache. Keep empty placeholders so
                # ForwardBatchInfo still has one list entry per layer.
                indexer_caches.append(
                    torch.empty(
                        (0,), dtype=torch.float8_e4m3fn, device=packed.device
                    )
                )
                indexer_scale_caches.append(
                    torch.empty((0,), dtype=torch.float32, device=packed.device)
                )
                continue

            indexer_k = indexer_cache[0]
            indexer_caches.append(
                indexer_k.reshape(-1, self.index_head_dim)
            )
            if len(indexer_cache) == 2:
                indexer_scale_caches.append(indexer_cache[1].reshape(-1, 1))
            else:
                indexer_scale_caches.append(
                    torch.empty((0,), dtype=torch.float32, device=packed.device)
                )

        return main_caches, indexer_caches, indexer_scale_caches

    def _num_blocks(self) -> int:
        cfg = self.vllm_config.cache_config
        nb = getattr(cfg, "num_gpu_blocks", None)
        if nb is None:
            nb = getattr(cfg, "num_blocks", None)
        if nb is None:
            nb = int(os.environ.get("MEGA_KV_BLOCKS", "2048"))
        return int(nb)

    def get_meta_buffers(self, batch_size: int) -> Dict[str, torch.Tensor]:
        if batch_size not in self.meta_buffers:
            max_tokens = max(self.max_tokens_default, batch_size * 4)
            self.meta_buffers[batch_size] = {
                "input_ids": torch.empty(
                    (max_tokens,), dtype=torch.int32, device="npu"
                ),
                "positions": torch.empty(
                    (max_tokens,), dtype=torch.int64, device="npu"
                ),
                "slot_mapping": torch.empty(
                    (max_tokens,), dtype=torch.int32, device="npu"
                ),
                "block_tables": torch.empty(
                    (batch_size, self.num_blocks),
                    dtype=torch.int32,
                    device="npu",
                ).fill_(-1),
                "q_seq_len": torch.empty(
                    (batch_size,), dtype=torch.int32, device="npu"
                ),
                "kv_seq_len": torch.empty(
                    (batch_size,), dtype=torch.int32, device="npu"
                ),
            }
        return self.meta_buffers[batch_size]

    def build_forward_batch_info(
        self,
        attn_meta: AscendDSAMetadata,
        meta_bufs: Dict[str, torch.Tensor],
        num_tokens: int,
    ):
        from blockrt.models.glm_5_2.model_args import ForwardBatchInfo

        # Fields for ForwardBatchInfo come from the persistent meta_bufs and
        # the caches of this state; per-request fields (block_table, seq_lens,
        # slot_mapping) are read from the metadata object itself in
        # _forward_mega (AscendSFAMetadata has no req_metadata sub-object).
        req = getattr(attn_meta, "req_metadata", None) or attn_meta
        del req
        # Per-token caches are persistent full-width buffers; blockrt consumes
        # them as [num_tokens, ...] views (view shares storage, so the DAG
        # captured data pointers stay stable across decode steps).
        return ForwardBatchInfo(
            sin_cos_cache=self.mla_cos_cache[:num_tokens],
            k_cache=self.k_cache,
            v_cache=self.k_cache,
            num_blocks=self.num_blocks,
            block_tables=meta_bufs["block_tables"],
            slot_mapping=meta_bufs["slot_mapping"][:num_tokens].reshape(
                num_tokens, 1
            ),
            kv_seq_len=meta_bufs["kv_seq_len"],
            q_seq_len=meta_bufs["q_seq_len"],
            num_tokens=num_tokens,
            mask=meta_bufs["mask"] if "mask" in meta_bufs else None,
            mask_type=1,
            index_cos_sin_cache=self.index_cos_sin_cache[:num_tokens],
            index_k_buffer=self.index_k_buffer,
            index_k_scale_buffer=self.index_k_scale_buffer,
            mla_cos_cache=self.mla_cos_cache[:num_tokens],
            mla_sin_cache=self.mla_sin_cache[:num_tokens],
        )


class AscendGlm52MegaForCausalLM(GlmMoeDsaForCausalLM):
    """GLM-5.2 model that routes through MegaKernel when enabled."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self.vllm_config = vllm_config
        self._mega_state: Optional[_MegaKernelGLM52State] = None

    def _ensure_mega(self, vllm_config: VllmConfig) -> _MegaKernelGLM52State:
        if self._mega_state is None:
            # vLLM runs the model forward under torch.inference_mode(), which
            # is thread-local. Tensors created there are "inference tensors"
            # and cannot be mutated in worker threads (the blockrt weight
            # loader parallelizes expert fusion with a thread pool). Run the
            # whole initialization in a fresh thread so every tensor is a
            # regular tensor.
            import threading

            print("[MegaKernel] initializing GLM-5.2 mega kernel", flush=True)
            result: Dict[str, Any] = {}
            errors: list[BaseException] = []

            def _init() -> None:
                try:
                    if torch.npu.is_available():
                        torch.npu.set_device(torch.npu.current_device())
                    result["state"] = _MegaKernelGLM52State(vllm_config, self)
                except BaseException as exc:  # noqa: BLE001
                    import traceback

                    traceback.print_exc()
                    errors.append(exc)

            t = threading.Thread(target=_init)
            t.start()
            t.join()
            if errors:
                raise errors[0]
            self._mega_state = result["state"]
            # Drain the NPU stream: the weight loader enqueues a large number
            # of expert npu_format_cast kernels asynchronously.  The first
            # decode step syncs the stream inside compute_graph_cache_key
            # (int(q_seq_len)); if those kernels are still in flight, the
            # sync hits the AICPU timeout.  Force completion here instead.
            if torch.npu.is_available():
                torch.npu.synchronize()
            print("[MegaKernel] GLM-5.2 mega kernel ready", flush=True)
        return self._mega_state

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if _mega_enabled() and input_ids is not None:
            attn_meta = None
            try:
                fc = get_forward_context()
                if fc is not None:
                    attn_meta = _extract_dsa_metadata(fc.attn_metadata)
            except Exception:
                attn_meta = None
            if (
                attn_meta is not None
                and self._is_decode_step(attn_meta)
                and _has_usable_metadata(attn_meta)
            ):
                print(
                    "[MegaKernel] decode step -> megakernel forward",
                    flush=True,
                )
                try:
                    return self._forward_mega(input_ids, positions, attn_meta)
                except Exception:
                    import traceback

                    traceback.print_exc()
                    raise
        return super().forward(
            input_ids, positions, intermediate_tensors, inputs_embeds, **kwargs
        )

    def compute_logits(
        self, hidden_states: torch.Tensor
    ) -> torch.Tensor | None:
        # In mega mode the model forward already returned logits; keep them
        # as-is (skip the vLLM LM-head pass).
        if _mega_enabled() and self._mega_state is not None:
            return hidden_states
        return super().compute_logits(hidden_states)

    def load_weights(self, weights):
        # The reduced checkpoint keeps the MTP head under layer index
        # `num_hidden_layers` (renamed from the full model's MTP layer), which
        # the base (non-spec) vLLM model does not define (e.g. `rot`). The
        # MegaKernel path does not use the MTP head either, so drop it here.
        if _mega_enabled():
            weights = (
                (n, t)
                for n, t in weights
                if not n.startswith("rot.")
                and not n.startswith(f"model.layers.{self.config.num_hidden_layers}.")
            )
        return super().load_weights(weights)

    @staticmethod
    def _is_decode_step(attn_meta: Any) -> bool:
        """True for a pure decode batch (AscendAttentionState.DecodeOnly).

        Prefill and mixed chunked-prefill batches keep using the native
        vLLM-Ascend path; only decode steps are routed through MegaKernel.
        """
        from vllm_ascend.attention.attention_v1 import AscendAttentionState

        state = getattr(attn_meta, "attn_state", None)
        if state is not None:
            return state == AscendAttentionState.DecodeOnly
        # Fallback: no prefill tokens but some decode tokens.
        return (
            getattr(attn_meta, "num_prefills", 0) == 0
            and getattr(attn_meta, "num_decode_tokens", 0) > 0
        )

    def _fill_rope_caches(
        self, state: _MegaKernelGLM52State, positions: torch.Tensor, num_tokens: int
    ) -> None:
        """Populate the persistent MLA/indexer RoPE caches for this step.

        vLLM stores per-position cos/sin as [B, rotary_dim//2] pair
        coefficients (interleaved rope, is_neox_style=False); the MegaKernel
        kernels consume [B, 64] with the pair coefficients repeat-interleaved
        (c0,c0,c1,c1,...), and the DSA indexer needs [B, 128] =
        cat(cos64, sin64) under the same convention.
        """
        if num_tokens <= 0:
            return

        def _cos_sin(rope, positions_cpu):
            # Reproduce vLLM cos_sin_cache on the host: cache = cat(cos, sin)
            # with rotary_dim//2 interleaved pair coefficients per position.
            # Computed on CPU (and copied in) to avoid NPU index_select, whose
            # Index kernel fails to launch on this stack (driver halMemAlloc).
            inv_freq = 1.0 / (
                rope.base
                ** (torch.arange(0, rope.rotary_dim, 2, dtype=torch.float32) / rope.rotary_dim)
            )
            freqs = torch.einsum("i,j->ij", positions_cpu.float(), inv_freq)
            return freqs.cos(), freqs.sin()

        pos_cpu = positions[:num_tokens].cpu()
        attn0 = self.model.layers[0].self_attn

        cos32, sin32 = _cos_sin(attn0.rotary_emb, pos_cpu)
        state.mla_cos_cache[:num_tokens].copy_(
            cos32.repeat_interleave(2, dim=-1).to(torch.bfloat16).npu()
        )
        state.mla_sin_cache[:num_tokens].copy_(
            sin32.repeat_interleave(2, dim=-1).to(torch.bfloat16).npu()
        )

        irope = getattr(attn0, "indexer_rope_emb", None)
        if irope is not None:
            icos32, isin32 = _cos_sin(irope, pos_cpu)
            idx_cache = torch.cat(
                (
                    icos32.repeat_interleave(2, dim=-1),
                    isin32.repeat_interleave(2, dim=-1),
                ),
                dim=-1,
            )  # [B, 128]
            state.index_cos_sin_cache[:num_tokens].copy_(
                idx_cache.to(torch.bfloat16).npu()
            )

    def _forward_mega(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        attn_meta: AscendDSAMetadata,
    ) -> torch.Tensor:
        # The metadata may carry per-request fields directly or under a
        # `req_metadata` sub-object (AscendDSAMetadata vs. MLA/SFA metadata).
        req = getattr(attn_meta, "req_metadata", None) or attn_meta
        state = self._ensure_mega(self.vllm_config)  # type: ignore[attr-defined]

        num_tokens = int(
            getattr(attn_meta, "num_actual_tokens", None)
            or getattr(attn_meta, "num_tokens", None)
            or input_ids.shape[0]
        )
        seq_lens = getattr(req, "seq_lens", None)
        block_table = getattr(req, "block_table", None)
        slot_mapping = getattr(req, "slot_mapping", None)
        query_start_loc = getattr(req, "query_start_loc", None)
        if seq_lens is None or block_table is None:
            raise RuntimeError(
                "MegaKernel: attention metadata lacks required fields "
                f"({type(attn_meta).__name__}): "
                f"seq_lens={seq_lens is not None}, "
                f"block_table={block_table is not None}, "
                f"slot_mapping={slot_mapping is not None}, "
                f"query_start_loc={query_start_loc is not None}"
            )
        batch_size = int(seq_lens.shape[0])
        bufs = state.get_meta_buffers(batch_size)

        # Fill the persistent MLA/indexer RoPE caches for this step.
        self._fill_rope_caches(state, positions, num_tokens)

        # ── Copy per-step inputs into persistent buffers (stable addresses) ──
        ids = input_ids.to(torch.int32).contiguous()
        pos = positions.to(torch.int64).contiguous()
        bufs["input_ids"][:num_tokens].copy_(ids)
        bufs["positions"][:num_tokens].copy_(pos)

        slot = slot_mapping.to(torch.int32).reshape(-1).contiguous()
        bufs["slot_mapping"][:num_tokens].copy_(slot[:num_tokens])

        block_table = block_table.to(torch.int32).contiguous()
        max_blocks = min(block_table.shape[1], self._mega_state.num_blocks)
        bufs["block_tables"][:, :max_blocks].copy_(
            block_table[:, :max_blocks]
        )

        seq_lens = seq_lens.to(torch.int32).contiguous()
        bufs["kv_seq_len"][:batch_size].copy_(seq_lens[:batch_size])
        # q_seq_len per sequence derived from query_start_loc (works for both
        # prefill and decode batches).
        qsl_t = query_start_loc
        if qsl_t is None:
            qsl_t = getattr(attn_meta, "cum_query_lens", None)
        if qsl_t is None:
            # Fall back to per-token q_seq_len = 1 (decode-only assumption).
            bufs["q_seq_len"][:batch_size].fill_(1)
        else:
            qsl = qsl_t.to(torch.int64).contiguous()
            if qsl.shape[0] >= batch_size + 1:
                q_seq = qsl[1 : batch_size + 1] - qsl[:batch_size]
            else:
                q_seq = torch.ones(
                    (batch_size,), dtype=torch.int64, device=qsl.device
                )
            bufs["q_seq_len"][:batch_size].copy_(q_seq.to(torch.int32))

        fbi = state.build_forward_batch_info(attn_meta, bufs, num_tokens)
        logits = state.mega.forward(
            fbi,
            bufs["input_ids"][:num_tokens],
            bufs["positions"][:num_tokens],
            None,
            None,
        )
        # vLLM sampling expects float32 logits [num_tokens, vocab_size].
        return logits.to(torch.float32)


def register_megakernel_model() -> None:
    """Point the GLM-5.2 registry entry at the MegaKernel wrapper."""
    from vllm import ModelRegistry

    ModelRegistry.register_model(
        "GlmMoeDsaForCausalLM",
        "vllm_ascend.models.glm_5_2_mega:AscendGlm52MegaForCausalLM",
    )
