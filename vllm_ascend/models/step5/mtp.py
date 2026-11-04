# SPDX-License-Identifier: Apache-2.0
"""Ascend Step5 MTP draft model.

The upstream ``Step5MTP`` builds CUDA decoder blocks in its predictor
layers. This module keeps the upstream wire format and interfaces
(``forward``/``compute_logits``/``get_top_tokens``/``load_weights`` are all
inherited untouched) and only re-creates the ``__init__`` paths so the draft
transformer block runs through ``AscendStep5DecoderLayer`` -- the platform
Ascend attention backends plus the NPU kernels already proven on the target
model. The MTP layers are dense sliding layers, so they land on the platform
sliding-window backend, not DSA.

Weight loading, module names and parameter names mirror the upstream
implementation exactly, so the checkpoint's ``model.layers.92.*`` MTP
weights load unchanged.
"""

from __future__ import annotations

from copy import deepcopy

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.config.utils import replace
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding,
)
from vllm.model_executor.models.utils import WeightsMapper, maybe_prefix
from vllm.models.step5.nvidia.mtp import (
    Step5MTPSharedHead,
    Step5MTP,
    Step5MultiTokenPredictor,
    Step5MultiTokenPredictorLayer,
    _build_mtp_norm,
    _get_mtp_config,
    _get_mtp_vllm_config,
    _get_step5_mtp_moe_blocks,
    _is_step5_mtp_dense,
)
from vllm.models.step5.nvidia.model import _set_step5_moe_protocol_metadata

from vllm_ascend.models.step5.model import AscendStep5DecoderLayer


class AscendStep5MultiTokenPredictorLayer(Step5MultiTokenPredictorLayer):
    """Upstream predictor layer with the draft block swapped to Ascend."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str,
    ) -> None:
        # Bypass the parent __init__: it hardcodes the CUDA
        # Step5DecoderLayer. Everything else mirrors it verbatim so
        # module/parameter names stay checkpoint-compatible.
        nn.Module.__init__(self)
        # The DSA modules hold python-side per-layer state; stay eager.
        self.do_not_compile = True
        mtp_vllm_config = _get_mtp_vllm_config(vllm_config)
        config = mtp_vllm_config.model_config.hf_text_config
        self.fp32_residual_connection = config.fp32_residual_connection
        quant_config = vllm_config.quant_config
        self.enorm = _build_mtp_norm(config)
        self.hnorm = _build_mtp_norm(config)
        self.eh_proj = nn.Linear(
            config.hidden_size * 2, config.hidden_size, bias=False
        )
        self.shared_head = Step5MTPSharedHead(
            config=config, quant_config=quant_config, prefix=f"{prefix}.shared_head"
        )
        self.mtp_block = AscendStep5DecoderLayer(
            mtp_vllm_config,
            prefix=f"{prefix}.mtp_block",
        )


class AscendStep5MultiTokenPredictor(Step5MultiTokenPredictor):
    """Upstream predictor whose layers use the Ascend draft blocks."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        nn.Module.__init__(self)
        config = _get_mtp_config(vllm_config)
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
        )
        self.mtp_start_layer_idx = config.num_hidden_layers
        self.num_mtp_layers = config.num_nextn_predict_layers
        if self.num_mtp_layers <= 0:
            raise ValueError(
                "Step5 MTP requires num_nextn_predict_layers > 0, got "
                f"{self.num_mtp_layers}."
            )
        # to map the exact layer index from weights
        self.layers = torch.nn.ModuleDict(
            {
                str(idx): AscendStep5MultiTokenPredictorLayer(
                    vllm_config=vllm_config,
                    prefix=f"{prefix}.layers.{idx}",
                )
                for idx in range(
                    self.mtp_start_layer_idx,
                    self.mtp_start_layer_idx + self.num_mtp_layers,
                )
            }
        )

        self.logits_processor = LogitsProcessor(config.vocab_size)


class AscendStep5MTP(Step5MTP):
    """Step5MTP for Ascend: same wire format, Ascend draft decoder layer."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        # Bypass Step5MTP.__init__ (it builds the CUDA predictor); the body
        # below replicates it with the Ascend predictor.
        nn.Module.__init__(self)
        # The target VL model maps quantization exclusions to
        # language_model.*. Draft modules live at model.* and must not
        # mutate the target config.
        if vllm_config.quant_config is not None:
            quant_config = deepcopy(vllm_config.quant_config)
            quant_config.apply_vllm_mapper(
                WeightsMapper(orig_to_new_prefix={"language_model.": ""})
            )
            vllm_config = replace(vllm_config, quant_config=quant_config)
        self.config = _get_mtp_config(vllm_config)
        self.vllm_config = vllm_config
        self.model = AscendStep5MultiTokenPredictor(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        # Expose the first draft head to the generic sharing policy. Each MTP
        # layer retains its own checkpoint head for compute_logits.
        self.lm_head = self.model.layers[
            str(self.model.mtp_start_layer_idx)
        ].shared_head.head
        self.moe_layers: list = []
        if not _is_step5_mtp_dense(self.model):
            raise NotImplementedError("Step5 MTP supports dense draft layers only")
        # Expose the actual draft topology so dense MTP forwards do not
        # inherit the target model's DP collectives.
        self.is_dense_mtp = True
        _set_step5_moe_protocol_metadata(self, None)
