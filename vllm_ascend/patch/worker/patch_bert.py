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
#

import torch
from vllm.sequence import IntermediateTensors

from vllm.model_executor.models.bert import BertEmbedding, BertForSequenceClassification, \
    BertForTokenClassification

# aclgraph does not support shift operator for now
# TODO: revert me when aclgraph supports shift operator
TOKEN_TYPE_SHIFT = 30
TOKEN_TYPE_MULTIPLIER = 1 << 30
TOKEN_MASK = TOKEN_TYPE_MULTIPLIER - 1


def _encode_token_type_ids(input_ids: torch.Tensor,
                           token_type_ids: torch.Tensor) -> None:
    # input_ids can be padded to the right
    input_ids[:token_type_ids.shape[0]].bitwise_or_(token_type_ids *
                                                    TOKEN_TYPE_MULTIPLIER)


def _decode_token_type_ids(input_ids: torch.Tensor) -> torch.Tensor:

    token_type_ids = input_ids // TOKEN_TYPE_MULTIPLIER

    input_ids.bitwise_and_(TOKEN_MASK)

    return token_type_ids


def bert_embedding_forward(
    self,
    input_ids: torch.Tensor,
    position_ids: torch.Tensor,
    inputs_embeds: torch.Tensor | None = None,
) -> torch.Tensor:
    token_type_ids = _decode_token_type_ids(input_ids)

    if inputs_embeds is None:
        inputs_embeds = self.word_embeddings(input_ids)

    position_embeddings = self.position_embeddings(position_ids)

    token_type_embeddings = self.token_type_embeddings(token_type_ids)

    embeddings = inputs_embeds + token_type_embeddings + position_embeddings
    embeddings = self.LayerNorm(embeddings)
    return embeddings


def bert_for_sequence_classification_forward(
    self,
    input_ids: torch.Tensor | None,
    positions: torch.Tensor,
    intermediate_tensors: IntermediateTensors | None = None,
    inputs_embeds: torch.Tensor | None = None,
    token_type_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    if token_type_ids is not None:
        assert self.bert.config.vocab_size < (1 << TOKEN_TYPE_SHIFT)
        assert input_ids is not None
        _encode_token_type_ids(input_ids, token_type_ids)

    return self.bert(
        input_ids=input_ids,
        positions=positions,
        inputs_embeds=inputs_embeds,
        intermediate_tensors=intermediate_tensors,
    )


def bert_for_token_classification_forward(
    self,
    input_ids: torch.Tensor | None,
    positions: torch.Tensor,
    intermediate_tensors: IntermediateTensors | None = None,
    inputs_embeds: torch.Tensor | None = None,
    token_type_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    if token_type_ids is not None:
        assert self.bert.config.vocab_size < (1 << TOKEN_TYPE_SHIFT)
        assert input_ids is not None
        _encode_token_type_ids(input_ids, token_type_ids)

    hidden_states = self.bert(
        input_ids=input_ids,
        positions=positions,
        inputs_embeds=inputs_embeds,
        intermediate_tensors=intermediate_tensors,
    )

    hidden_states = hidden_states.to(self.head_dtype)
    return self.classifier(hidden_states)


BertEmbedding.forward = bert_embedding_forward
BertForSequenceClassification.forward = bert_for_sequence_classification_forward
BertForTokenClassification.forward = bert_for_token_classification_forward
