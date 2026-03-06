#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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
from collections.abc import Mapping
from functools import lru_cache

from transformers import PreTrainedTokenizerBase
from vllm.model_executor.models.qwen3_vl import Qwen3VLMultiModalProcessor
from vllm.multimodal.processing import MultiModalPromptUpdates, PlaceholderFeaturesInfo


@lru_cache
def _get_merged_lt_splits(tokenizer: PreTrainedTokenizerBase) -> dict[int, list[int]]:
    """Pre-compute split mappings for vocab tokens that end with `<`."""
    vocab = tokenizer.get_vocab()
    lt_id = vocab["<"]
    splits: dict[int, list[int]] = {}
    for token_id in vocab.values():
        if token_id == lt_id:
            continue
        decoded = tokenizer.decode([token_id])
        if decoded.endswith("<"):
            prefix_ids = tokenizer.encode(decoded[:-1], add_special_tokens=False)
            splits[token_id] = prefix_ids + [lt_id]
    return splits


def _find_mm_placeholders(
    self,
    new_token_ids: list[int],
    mm_prompt_updates: MultiModalPromptUpdates,
) -> Mapping[str, list[PlaceholderFeaturesInfo]]:
    # Qwen3-VL may merge the `<` in timestamp text (for example `.<` or `:<`)
    # with the preceding punctuation token. Split those merged tokens before
    # placeholder matching so multimodal ranges remain aligned with features.
    tokenizer = self.info.get_tokenizer()
    merged_lt_splits = _get_merged_lt_splits(tokenizer)

    if not merged_lt_splits or merged_lt_splits.keys().isdisjoint(new_token_ids):
        return super(Qwen3VLMultiModalProcessor, self)._find_mm_placeholders(new_token_ids, mm_prompt_updates)

    replaced_token_ids = list[int]()
    replaced_orig_indices = list[int]()
    for orig_idx, token_id in enumerate(new_token_ids):
        replaced_tokens = merged_lt_splits.get(token_id, [token_id])
        replaced_token_ids.extend(replaced_tokens)
        replaced_orig_indices.extend(orig_idx for _ in range(len(replaced_tokens)))

    placeholder_infos = super(Qwen3VLMultiModalProcessor, self)._find_mm_placeholders(
        replaced_token_ids, mm_prompt_updates
    )

    return {
        modality: [
            PlaceholderFeaturesInfo(
                modality=placeholder.modality,
                item_idx=placeholder.item_idx,
                start_idx=replaced_orig_indices[placeholder.start_idx],
                tokens=placeholder.tokens,
                is_embed=placeholder.is_embed,
            )
            for placeholder in placeholders
        ]
        for modality, placeholders in placeholder_infos.items()
    }


Qwen3VLMultiModalProcessor._find_mm_placeholders = _find_mm_placeholders
