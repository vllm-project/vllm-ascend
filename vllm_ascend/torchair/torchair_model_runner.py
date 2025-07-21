#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
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
# This file is a part of the vllm-ascend project.
# Adapted from vllm-project/vllm/vllm/worker/gpu_model_runner.py
#

import types

import torch
import torch._dynamo.cache_size
import torch.nn as nn
import torch_npu
import vllm.envs as envs_vllm
from vllm.config import VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.logger import logger

import vllm_ascend.envs as envs_ascend
from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.platform import NPUPlatform
from vllm_ascend.torchair.utils import (check_torchair_cache_exist,
                                        converting_weight_acl_format_310p,
                                        write_kv_cache_bytes_to_file)
from vllm_ascend.utils import (ACL_FORMAT_FRACTAL_ND, ACL_FORMAT_FRACTAL_NZ,
                               is_310p)
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner

if is_310p():
    torch_npu.npu.set_compile_mode(jit_compile=False)
    ACL_FORMAT = ACL_FORMAT_FRACTAL_NZ
else:
    ACL_FORMAT = ACL_FORMAT_FRACTAL_ND


class NPUTorchairModelRunner(NPUModelRunner):

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        """Init torchair related parameters"""
        super().__init__(vllm_config, device)

        ascend_config = get_ascend_config()
        self.new_kv_cache_bytes = -1
        self.torchair_compiled_model = None  # type: ignore
        self.torchair_compiled_models = {}  # type: ignore
        self.use_cached_npu_graph = ascend_config.torchair_graph_config.use_cached_graph
        self.torchair_graph_batch_sizes = ascend_config.torchair_graph_config.graph_batch_sizes
        if ascend_config.torchair_graph_config.graph_batch_sizes_init:
            self._init_torchair_graph_batch_sizes()
        if len(self.torchair_graph_batch_sizes) == 0:
            # TODO(zzzzwwjj): check torchair_graph_batch_sizes init code
            self.torchair_graph_batch_sizes = [self.max_num_reqs]
        torch._dynamo.cache_size.config.cache_size_limit += len(
            self.torchair_graph_batch_sizes)
        torch._dynamo.config.capture_dynamic_output_shape_ops = True
        torch._logging.set_logs(
            recompiles=envs_ascend.VLLM_ASCEND_TRACE_RECOMPILES)

    def _generate_extra_builder_kwargs(self, total_num_scheduled_tokens,
                                       with_prefill):
        """Generate graph_pad_size for torchair case."""
        extra_builder_kwargs = {}
        padded_batch_size = -1

        if self.dp_size > 1:
            max_num_tokens, with_prefill = self._get_forward_metadata_across_dp(
                total_num_scheduled_tokens, with_prefill)
            extra_builder_kwargs['max_num_tokens_across_dp'] = max_num_tokens
            extra_builder_kwargs['with_prefill_across_dp'] = with_prefill
        if not with_prefill:
            if self.dp_size > 1:
                padded_batch_size = self._select_torchair_padded_batch_size(
                    max_num_tokens)
            else:
                padded_batch_size = self._select_torchair_padded_batch_size(
                    total_num_scheduled_tokens)
            graph_pad_size = padded_batch_size - total_num_scheduled_tokens
            extra_builder_kwargs['graph_pad_size'] = graph_pad_size
        return extra_builder_kwargs, padded_batch_size

    def _generate_hidden_states(self, input_ids, positions,
                                intermediate_tensors, inputs_embeds,
                                attn_metadata, with_prefill, padded_batch_size,
                                **model_kwargs):
        """Generate hidden_states with compiled model."""
        model_kwargs["kv_caches"] = self.kv_caches
        model_kwargs["attn_metadata"] = attn_metadata
        if not with_prefill:
            if is_310p():
                converting_weight_acl_format_310p(self.model,
                                                  ACL_FORMAT_FRACTAL_NZ)
            input_ids = self.input_ids[:padded_batch_size]
            positions = self.positions[:padded_batch_size]
            compiled_model = self._get_torchair_lazy_compiled_model(
                padded_batch_size)
            hidden_states = compiled_model(
                input_ids=input_ids,
                positions=positions,
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=inputs_embeds,
                **model_kwargs,
            )
        else:
            hidden_states = super()._generate_hidden_states(
                input_ids, positions, intermediate_tensors, inputs_embeds,
                attn_metadata, with_prefill, padded_batch_size, **model_kwargs)
        return hidden_states

    def _generate_dummy_run_hidden_states(self, input_ids, positions,
                                          intermediate_tensors, inputs_embeds,
                                          num_tokens, with_prefill,
                                          is_compile):
        """Override _generate_dummy_run_hidden_states to use torchair graph."""
        if not with_prefill:
            if is_310p():
                converting_weight_acl_format_310p(self.model,
                                                  ACL_FORMAT_FRACTAL_NZ)
            attn_metadata = self.attn_metadata_builder.build_dummy(
                num_reqs=num_tokens, num_actual_tokens=1)
            # Only mark static while compiling
            if is_compile:
                torch._dynamo.mark_static(input_ids)
                torch._dynamo.mark_static(positions)
                torch._dynamo.mark_static(attn_metadata.decode.block_table)
                torch._dynamo.mark_static(attn_metadata.decode.input_positions)
                torch._dynamo.mark_static(attn_metadata.slot_mapping)
                for kv in self.kv_caches:
                    assert isinstance(kv, tuple), "kv_cache must be a tuple"
                    torch._dynamo.mark_static(kv[0])
                    torch._dynamo.mark_static(kv[1])
            compiled_model = self._get_torchair_lazy_compiled_model(num_tokens)
            hidden_states = compiled_model(
                input_ids=input_ids,
                positions=positions,
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=None,
                kv_caches=self.kv_caches,
                attn_metadata=attn_metadata,
            )
        else:
            if is_310p():
                converting_weight_acl_format_310p(self.model,
                                                  ACL_FORMAT_FRACTAL_ND)
            hidden_states = super()._generate_dummy_run_hidden_states(
                input_ids, positions, intermediate_tensors, inputs_embeds,
                num_tokens, with_prefill, is_compile)
        return hidden_states

    def _update_fullattention_spec(self, kv_caches, kv_cache_shape,
                                   layer_name):
        """Override _update_fullattention_spec to update kv_cache in torchair way."""
        if len(kv_cache_shape) == 3:
            # for non MLA attention backend that use torchair, we consider to pass kv_cache layout
            # of BSH ([num_blocks, block_size, kv_head_dim * head_size]) to attention.
            kv_caches[layer_name] = (torch.zeros(kv_cache_shape,
                                                 dtype=self.kv_cache_dtype,
                                                 device=self.device),
                                     torch.zeros(kv_cache_shape,
                                                 dtype=self.kv_cache_dtype,
                                                 device=self.device))
            # atb reshape_and_cache does not support torchair.
            kv_caches[layer_name] = (
                torch_npu.npu_format_cast(kv_caches[layer_name][0],
                                          ACL_FORMAT_FRACTAL_ND),
                torch_npu.npu_format_cast(kv_caches[layer_name][1],
                                          ACL_FORMAT_FRACTAL_ND),
            )
        else:
            # for MLA attention backend that use torchair.
            layer_kv_cache_nope = torch.zeros(
                kv_cache_shape[:-1] +
                (self.model_config.hf_text_config.kv_lora_rank, ),
                dtype=self.dtype,
                pin_memory=True,
                device=self.device)
            layer_kv_cache_pe = torch.zeros(
                kv_cache_shape[:-1] +
                (self.model_config.hf_text_config.qk_rope_head_dim, ),
                dtype=self.dtype,
                pin_memory=True,
                device=self.device)
            kv_caches[layer_name] = (layer_kv_cache_nope, layer_kv_cache_pe)
            kv_caches[layer_name] = (
                torch_npu.npu_format_cast(kv_caches[layer_name][0],
                                          ACL_FORMAT),
                torch_npu.npu_format_cast(kv_caches[layer_name][1],
                                          ACL_FORMAT),
            )

    def _capture_model(self) -> None:
        """Override _capture_model to generate torchair graph."""
        # TODO(NeverRaR): Calling graph_capture(device=self.device) in
        # torchair graph capture can cause some issues, so now we just
        # temporarily split the codepath for the two different graph patterns.
        torchair_graph_batch_sizes = self.torchair_graph_batch_sizes
        graph_num = len(torchair_graph_batch_sizes)

        if self.use_cached_npu_graph and not check_torchair_cache_exist():
            # If caching is enabled but does not exist, we will compile the model twice. The first
            # time is used to generate the cache, and the second time is used to load the cache to
            # skip the overhead caused by Dynamo guard mechanism.
            logger.info(
                "Use cached npu graph but cache doesn't exist! Now we compile graph to genetate torchair cache, this usually takes %.1f~%.1f mins.",
                0.5 * graph_num, 1.5 * graph_num)
            self._compile_torchair_graph(torchair_graph_batch_sizes)
            NPUPlatform.synchronize()
            torch._dynamo.reset()
            self.torchair_compiled_models.clear()
        if self.use_cached_npu_graph:
            logger.info(
                "Loading torchair graph cache, this usually takes %.1f~%.1f mins.",
                0.3 * graph_num, 0.5 * graph_num)
            self._compile_torchair_graph(torchair_graph_batch_sizes)
        else:
            logger.info(
                "Capturing torchair graph, this usually takes %.1f~%.1f mins.",
                0.5 * graph_num, 1.5 * graph_num)
            self._compile_torchair_graph(torchair_graph_batch_sizes)

        if self.new_kv_cache_bytes > 0:
            write_kv_cache_bytes_to_file(torch.distributed.get_rank(),
                                         self.new_kv_cache_bytes)

    def _get_torchair_lazy_compiled_model(self, batch_size: int):
        if batch_size < 0 or batch_size > self.max_num_reqs:
            raise ValueError(
                f"Bad graph batch size:{batch_size}! max_num_reqs:{self.max_num_reqs}"
            )

        compiled_model = self.torchair_compiled_models.get(
            batch_size
        ) if self.use_cached_npu_graph else self.torchair_compiled_model

        if compiled_model:
            return compiled_model

        import torchair  # type: ignore
        from torchair import patch_for_hcom  # type: ignore

        patch_for_hcom()

        if is_310p():
            # on 300I Duo platform, we need to patch broadcast. however, this patch will be
            # overwritten by patch_for_hcom in torchair. so we need to re-patch it here.
            from vllm_ascend.patch.platform.patch_common.patch_distributed import \
                communication_adaptation_310p
            communication_adaptation_310p()

        config = torchair.CompilerConfig()
        config.experimental_config.frozen_parameter = True
        # enabling tiling_schedule_optimize on 300I Duo has some bugs, so we have to
        # disable it on 300I Duo platform now.
        config.experimental_config.tiling_schedule_optimize = not is_310p()
        config.experimental_config.enable_view_optimize = \
        get_ascend_config().torchair_graph_config.enable_view_optimize
        torch.npu.set_compile_mode(jit_compile=False)
        if not self.use_cached_npu_graph:
            npu_backend = torchair.get_npu_backend(compiler_config=config)
            self.torchair_compiled_model = torch.compile(
                self.model,
                dynamic=True,
                fullgraph=envs_vllm.VLLM_TEST_DYNAMO_FULLGRAPH_CAPTURE,
                backend=npu_backend)
            return self.torchair_compiled_model
        else:
            # Generate a new forward proxy code object to prevent the invalidation of
            # compilation cache caused by dynamo retracing
            forward_proxy_name = f"{self.model.__class__.__name__}_forward_with_batch_size_{batch_size}"
            forward_fn = self.model.forward
            code = forward_fn.__code__
            # Mark code object with a new proxy name
            modified_code = code.replace(co_name=forward_proxy_name, )

            modified_func = types.FunctionType(modified_code,
                                               forward_fn.__globals__,
                                               name=forward_proxy_name,
                                               argdefs=forward_fn.__defaults__)

            self.model.__dict__[forward_proxy_name] = modified_func.__get__(
                self.model, nn.Module)
            self.torchair_compiled_models[
                batch_size] = torchair.inference.cache_compile(
                    self.model.__dict__[forward_proxy_name],
                    dynamic=True,
                    fullgraph=envs_vllm.VLLM_TEST_DYNAMO_FULLGRAPH_CAPTURE,
                    config=config,
                    ge_cache=False)
            return self.torchair_compiled_models[batch_size]

    def _compile_torchair_graph(self, torchair_graph_batch_sizes) -> None:
        # Trigger torchair graph capture for specific shapes.
        # Capture the large shapes first so that the smaller shapes
        # can reuse the memory pool allocated for the large shapes.
        for idx, num_tokens in enumerate(reversed(torchair_graph_batch_sizes)):
            for _ in range(self.vllm_config.compilation_config.
                           cudagraph_num_of_warmups):
                self._dummy_run(num_tokens,
                                is_compile=True,
                                with_prefill=False)
            self._dummy_run(num_tokens, is_compile=True, with_prefill=False)
            logger.info("Batchsize %d is compiled successfully: %d/%d.",
                        num_tokens, idx + 1, len(torchair_graph_batch_sizes))

    def _init_torchair_graph_batch_sizes(self):
        start_graph_batch_size = 4
        tp_size = get_tensor_model_parallel_world_size()

        # NOTE: When use all2all | mc2, We need to slice the `num_tokens` dimension into `tp_size` blocks
        start_graph_batch_size = max(start_graph_batch_size, tp_size)

        while (start_graph_batch_size <= self.max_num_reqs):
            self.torchair_graph_batch_sizes.append(start_graph_batch_size)
            start_graph_batch_size *= 2

    def _select_torchair_padded_batch_size(self, batch_size: int):
        selected_batch_size = self.max_num_reqs
        for padded_batch_size in self.torchair_graph_batch_sizes:
            if batch_size <= padded_batch_size < selected_batch_size:
                selected_batch_size = padded_batch_size
        return selected_batch_size
