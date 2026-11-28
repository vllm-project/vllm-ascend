# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Qwen3.8 PLE/MRoPE IPC unpacked coverage, scheduled and timed independently."""

from dataclasses import replace

import pytest
import torch
from vllm import envs

from tests.e2e.common.rlhf.weight_transfer.case import (
    WeightUpdateModelCase,
)
from tests.e2e.common.rlhf.weight_transfer.npu_ipc import (
    run_npu_ipc_weight_transfer_transaction,
)
from tests.e2e.common.rlhf.weight_transfer.weight_source import FixedRandomWeightSource


class _Qwen38WeightSource(FixedRandomWeightSource):
    def _fixed_buffer_values(self, meta_model: torch.nn.Module) -> dict[str, torch.Tensor]:
        """Materialize only PLE hash constants; learned parameters stay on meta."""
        # Each trainer process loads the optional PLE implementation when it builds
        # this case's source. Reuse its hash initialization instead of duplicating it.
        from transformers.models.qwen4_exp.modeling_qwen4_exp import (
            Qwen4ExpTextNGramEmbedding,
            _build_layer_multipliers,
        )

        values: dict[str, torch.Tensor] = {}
        with torch.device("cpu"):
            for module_name, module in meta_model.named_modules():
                if not isinstance(module, Qwen4ExpTextNGramEmbedding):
                    continue
                buffers = {
                    "layer_multipliers": _build_layer_multipliers(
                        module.unigram_vocab_size, module.ngram_size, module.ple_layer_index, module.seed
                    ),
                    "ngram_heads_vocab_sizes": torch.tensor(module.head_vocab_sizes, dtype=torch.int64),
                    "ngram_heads_offsets": torch.tensor(module.head_offsets, dtype=torch.int64),
                }
                for leaf, tensor in buffers.items():
                    values[f"{module_name}.{leaf}"] = tensor
        return values


QWEN38_CASE = WeightUpdateModelCase(
    id="qwen3.8-flash-next-4layers-small-ple-mrope-head128",
    model="Qwen/Qwen3.8-Flash-Next",
    hf_overrides={
        "architectures": ["Qwen4ExpForConditionalGeneration"],
        "language_model_only": False,
        "text_config": {
            "num_hidden_layers": 4,
            # Overrides run after config normalization; use the runtime QSA name.
            "layer_types": ["linear_attention"] * 3 + ["indexed_attention"],
            "ple_layer_ids": [2],
            "ngram_vocab_size_base": 4096,
            "head_dim": 128,
            "partial_rotary_factor": 0.5,
            "rope_parameters": {
                "rope_type": "default",
                "rope_theta": 10000000,
                "mrope_interleaved": True,
                "mrope_section": [11, 11, 10],
                "partial_rotary_factor": 0.5,
            },
            "mtp_num_hidden_layers": 0,
            "mtp": {
                "hybrid": True,
                "layer_types": [],
                "mtp_use_hidden_state_from_layer": None,
                "num_hidden_layers": 0,
                "rope_theta": 10000000,
            },
        },
    },
    # Enumerate the text model on meta while retaining the conditional-generation
    # architecture/MRoPE in both inference servers.
    meta_config_attribute="text_config",
    checkpoint_model_prefix="model.language_model.",
    checkpoint_shard_size_bytes=2 * 2**30,
    weight_source_factory=_Qwen38WeightSource,
    extra_server_args=(
        "--max-num-seqs",
        "4",
        "--limit-mm-per-prompt",
        '{"image":0,"video":0}',
        "--engram-config",
        '{"cpu_offload":false}',
        "--kv-cache-memory-bytes",
        str(4 * 2**30),
    ),
)


QWEN38_CASES = (QWEN38_CASE,)


@pytest.mark.parametrize(
    "case",
    [pytest.param(case, id=f"unpacked-{case.id}", marks=pytest.mark.e2e_model(case.model)) for case in QWEN38_CASES],
)
def test_npu_ipc_weight_transfer_transaction(case: WeightUpdateModelCase):
    # The conditional-generation reference also loads its processor from the
    # generated checkpoint. Stage only assets, so the shared source can copy
    # them next to its synthetic weights without downloading pretrained tensors.
    asset_patterns = [
        "config.json",
        "*token*.json",
        "*preprocessor*.json",
        "vocab.json",
        "merges.txt",
        "*.jinja",
        "generation_config.json",
    ]
    if envs.VLLM_USE_MODELSCOPE:
        from modelscope import snapshot_download as ms_snapshot_download  # type: ignore[import-untyped]

        model_dir = ms_snapshot_download(case.model, allow_file_pattern=asset_patterns)
    else:
        from huggingface_hub import snapshot_download

        model_dir = snapshot_download(case.model, allow_patterns=asset_patterns)
    run_npu_ipc_weight_transfer_transaction(replace(case, model=model_dir), packed=False)
