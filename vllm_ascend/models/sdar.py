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
"""Config support for the SDAR checkpoints used as UNO speculative-decoding targets.

A UNO bundle (for example ``s-sahoo/uno-qwen3-8B``) ships the frozen verifier at
the repository root and the gated draft LoRA under ``adapter/``.  The verifier's
``config.json`` declares ``model_type: "sdar"`` and
``architectures: ["SDARForCausalLM"]`` plus an ``auto_map`` pointing at
``modeling_sdar.py``.

That remote file is generated from transformers' Qwen3 implementation and is
Qwen3 verbatim for inference: same per-head q/k RMSNorm, same GQA, same RoPE,
same SwiGLU MLP, and a weight map that is 1:1 with vLLM's Qwen3
(``model.layers.N.self_attn.{q,k,v,o}_proj`` and friends).  Its only
architectural extra is a block-diffusion attention mask, and that is reachable
only from ``self.training``; ``use_regular_causal`` additionally defaults to
``True``, so even the training-time mask is token-causal outside the noised
block.  UNO decoding is therefore plain causal attention over Qwen3.

Registering a config class for ``"sdar"`` lets the bundle load with
``trust_remote_code=False``.  That is a correctness requirement, not a
convenience: the shipped ``SDARForCausalLM.__init__`` reads ``config.noise``,
a key the released ``config.json`` does not contain, so the remote-code path
raises ``AttributeError`` before the weights are touched.
"""

from transformers import Qwen3Config
from vllm.logger import logger

_SDAR_ARCHITECTURE = "SDARForCausalLM"
_QWEN3_ARCHITECTURE = "Qwen3ForCausalLM"


class SDARConfig(Qwen3Config):
    """A Qwen3 config wearing the ``sdar`` model type.

    Subclassing matters: registering a bare ``Qwen3Config`` under ``"sdar"``
    would make vLLM's ``_register_config_class`` set
    ``Qwen3Config.model_type = "sdar"`` process-wide and silently break every
    genuine Qwen3 checkpoint served by the same engine.
    """

    model_type = "sdar"

    def __init__(self, **kwargs):
        # vLLM keeps the *original* architecture string in
        # ``ModelConfig.architecture``, and several feature gates match on it
        # by name.  Rewriting it here (rather than only aliasing it in the
        # model registry) keeps those gates working.  Unrecognised keys such as
        # ``block_size``, ``mask_token_id`` and ``fuse_cross_entropy`` are
        # carried inertly by ``PretrainedConfig``.
        architectures = kwargs.get("architectures")
        if architectures and all(arch == _SDAR_ARCHITECTURE for arch in architectures):
            kwargs["architectures"] = [_QWEN3_ARCHITECTURE]
            # Deliberately a warning, not an info line: this substitution also
            # applies to an SDAR checkpoint served on its own, where it takes
            # precedence over `--trust-remote-code`. The result is a correct
            # autoregressive model, but it is not the bundled block-diffusion
            # sampler, and the difference should not be silent.
            logger.warning(
                "Loading an SDAR checkpoint (%s) with vLLM's %s implementation. "
                "The two are weight- and architecture-identical for causal "
                "inference, which is what UNO speculative decoding needs, but "
                "SDAR's block-diffusion generation is NOT applied and the "
                "bundled remote code is not executed.",
                _SDAR_ARCHITECTURE,
                _QWEN3_ARCHITECTURE,
            )
        super().__init__(**kwargs)
