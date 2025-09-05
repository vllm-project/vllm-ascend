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
# isort: skip_file

import math
import types
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch_npu
import vllm.envs as envs_vllm
from vllm.config import VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.distributed.parallel_state import get_dp_group
from vllm.forward_context import get_forward_context
from vllm.logger import logger

import vllm_ascend.envs as envs_ascend
from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.platform import NPUPlatform
from vllm_ascend.torchair.utils import (
    TORCHAIR_CACHE_DIR, TorchairCommonAttentionMetadata,
    check_torchair_cache_exist, converting_weight_acl_format,
    register_torchair_model, torchair_ops_patch,
    torchair_quant_method_register, write_kv_cache_bytes_to_file)
from vllm_ascend.utils import (ACL_FORMAT_FRACTAL_ND, ACL_FORMAT_FRACTAL_NZ,
                               is_310p, get_ascend_soc_version,
                               AscendSocVersion)
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner


class NPUTorchairModelRunner(NPUModelRunner):

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        super().__init__(vllm_config, device)
        ascend_config = get_ascend_config()
        self.new_kv_cache_bytes = -1
        self.torchair_compiled_model = None  # type: ignore
        self.torchair_compiled_models = {}  # type: ignore
        self.use_cached_npu_graph = ascend_config.torchair_graph_config.use_cached_graph
        self.use_cached_kv_cache_bytes = ascend_config.torchair_graph_config.use_cached_kv_cache_bytes
        self.torchair_graph_batch_sizes = ascend_config.torchair_graph_config.graph_batch_sizes
        if ascend_config.torchair_graph_config.graph_batch_sizes_init:
            self.init_torchair_graph_batch_sizes()

        self.update_torchair_graph_batch_sizes()

        torch._dynamo.cache_size.config.cache_size_limit += len(
            self.torchair_graph_batch_sizes)
        torch._dynamo.config.capture_dynamic_output_shape_ops = True
        torch._logging.set_logs(
            recompiles=envs_ascend.VLLM_ASCEND_TRACE_RECOMPILES)

        self._check_batch_sizes_consistency()
        register_torchair_model()
        torchair_ops_patch()
        torchair_quant_method_register()

    def _sync_metadata_across_dp(
            self, num_tokens: int, with_prefill: bool, enable_dbo: bool
    ) -> tuple[int, Optional[torch.Tensor], bool, bool]:
        """Override from NPUModelRunner to pad num_tokens"""
        if self.dp_size == 1:
            if not with_prefill:
                maybe_padded_num_tokens = self.select_torchair_padded_batch_size(
                    num_tokens)
                return maybe_padded_num_tokens, None, with_prefill, enable_dbo
            return num_tokens, None, with_prefill, enable_dbo

        num_tokens_across_dp = torch.zeros(self.dp_size + 2,
                                           dtype=torch.int32,
                                           device="npu")
        num_tokens_across_dp[self.dp_rank] = num_tokens
        num_tokens_across_dp[-2] = int(with_prefill)
        num_tokens_across_dp[-1] = int(not enable_dbo)
        dist.all_reduce(num_tokens_across_dp,
                        group=get_dp_group().device_group)
        with_prefill = bool(num_tokens_across_dp[-2])
        enable_dbo = not bool(num_tokens_across_dp[-1])
        num_tokens_across_dp = num_tokens_across_dp[:-2]

        if not with_prefill:
            max_num_token = num_tokens_across_dp.max().item()
            maybe_padded_num_tokens = self.select_torchair_padded_batch_size(
                max_num_token)
            num_tokens_across_dp = torch.full((self.dp_size, ),
                                              maybe_padded_num_tokens,
                                              dtype=torch.int32,
                                              device="npu")
        else:
            maybe_padded_num_tokens = num_tokens

        return maybe_padded_num_tokens, num_tokens_across_dp, with_prefill, enable_dbo

    def _build_attention_metadata(self, with_prefill, num_reqs, skip_attn):
        # NOTE: If torchair graph mode and not with_prefill,
        # we can't skip_attn, it will cause graph recompile.
        if not with_prefill:
            common_attn_metadata = TorchairCommonAttentionMetadata(
                num_reqs=num_reqs,
                num_actual_tokens=1,
                actual_seq_lengths_q=self.actual_seq_lengths_q,
                attn_mask=self.attn_mask,
                spec_attn_mask=self.spec_attn_mask,
                decode_token_per_req=self.decode_token_per_req,
            )
            attn_metadata = self.attn_metadata_builder.build_torchair_graph_dummy(
                common_attn_metadata)
        else:
            attn_metadata = super()._build_attention_metadata(
                with_prefill, num_reqs, skip_attn)
        return attn_metadata

    def _generate_dummy_run_hidden_states(self, with_prefill,
                                          is_torchair_compile, input_ids,
                                          positions, attn_metadata, num_tokens,
                                          intermediate_tensors, inputs_embeds):

        if not with_prefill:
            # Only mark static while compiling
            if is_torchair_compile:
                torch._dynamo.mark_static(input_ids)
                torch._dynamo.mark_static(positions)
                torch._dynamo.mark_static(attn_metadata.decode.block_table)
                torch._dynamo.mark_static(attn_metadata.decode.input_positions)
                torch._dynamo.mark_static(get_forward_context().mc2_mask)
                if hasattr(attn_metadata.decode, "sin"):
                    torch._dynamo.mark_static(attn_metadata.decode.sin)
                    torch._dynamo.mark_static(attn_metadata.decode.cos)
                torch._dynamo.mark_static(attn_metadata.slot_mapping)
                if self.speculative_config:
                    torch._dynamo.mark_static(attn_metadata.decode.attn_mask)
                for kv in self.kv_caches:
                    assert isinstance(kv, tuple), "kv_cache must be a tuple"
                    torch._dynamo.mark_static(kv[0])
                    torch._dynamo.mark_static(kv[1])
            if is_310p():
                converting_weight_acl_format(self.model, ACL_FORMAT_FRACTAL_NZ)

            compiled_model = self._get_torchair_lazy_compiled_model(num_tokens)
            model_kwargs = {}
            model_kwargs["kv_caches"] = self.kv_caches
            model_kwargs["attn_metadata"] = attn_metadata
            hidden_states = compiled_model(
                input_ids=input_ids,
                positions=positions,
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=None,
                **model_kwargs,
            )
        else:
            if is_310p():
                converting_weight_acl_format(self.model, ACL_FORMAT_FRACTAL_ND)
            hidden_states = super()._generate_dummy_run_hidden_states(
                with_prefill, is_torchair_compile, input_ids, positions,
                attn_metadata, num_tokens, intermediate_tensors, inputs_embeds)
        return hidden_states

    def _convert_torch_format(self, kv_cache):
        kv_cache = torch_npu.npu_format_cast(kv_cache, ACL_FORMAT_FRACTAL_ND)
        return kv_cache

    def _compile_torchair_graph(self, torchair_graph_batch_sizes) -> None:
        # Trigger torchair graph capture for specific shapes.
        # Capture the large shapes first so that the smaller shapes
        # can reuse the memory pool allocated for the large shapes.
        for idx, num_tokens in enumerate(reversed(torchair_graph_batch_sizes)):
            for _ in range(self.vllm_config.compilation_config.
                           cudagraph_num_of_warmups):
                self._dummy_run(num_tokens, is_torchair_compile=True)
            self._dummy_run(num_tokens, is_torchair_compile=True)
            logger.info("Batchsize %d is compiled successfully: %d/%d.",
                        num_tokens, idx + 1, len(torchair_graph_batch_sizes))

    def _capture_model(self):
        """Override from NPUModelRunner to use torchair graph capture."""
        # TODO(NeverRaR): Calling graph_capture(device=self.device) in
        # torchair graph capture can cause some issues, so now we just
        # temporarily split the codepath for the two different graph patterns.
        torchair_graph_batch_sizes = self.torchair_graph_batch_sizes
        graph_num = len(torchair_graph_batch_sizes)

        if self.use_cached_npu_graph and not check_torchair_cache_exist():
            # If caching is enabled but does not exist (either
            # use_cached_kv_cache_bytes is disabled or kv_cache_bytes are
            # different), we will compile the model twice. The first time is
            # used to generate the cache, and the second time is used to load the
            # cache to skip the overhead caused by Dynamo guard mechanism.
            logger.info(
                "Cache compilation for torchair graph is enabled. Now we compile graph to genetate"
                " torchair cache, this usually takes %.1f~%.1f mins.",
                0.5 * graph_num, 1.5 * graph_num)
            self._compile_torchair_graph(torchair_graph_batch_sizes)
            NPUPlatform.synchronize()
            # Note: We reset dynamo and reload the compiled torchair cached computation graph below
            # that was compiled above. This operation reduces graph launch time by 2-4ms and avoids
            # runtime errors caused by configuration mismatches in graph mode.
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

        if self.use_cached_kv_cache_bytes and self.new_kv_cache_bytes > 0:
            write_kv_cache_bytes_to_file(torch.distributed.get_rank(),
                                         self.new_kv_cache_bytes)

    def _use_aclgraph(self) -> bool:
        return False

    def _check_batch_sizes_consistency(self) -> None:
        if not dist.is_initialized():
            return

        local = torch.tensor(self.torchair_graph_batch_sizes,
                             device="cpu",
                             dtype=torch.int32)
        gathered_graph_batch_size = local.clone()
        dist.all_reduce(gathered_graph_batch_size,
                        group=get_dp_group().cpu_group)
        expected = local * self.dp_size

        if not torch.equal(gathered_graph_batch_size, expected):
            diff_idxs = (gathered_graph_batch_size != expected).nonzero(
                as_tuple=False).flatten().tolist()
            raise AssertionError(
                f"[Graph BatchSize Mismatch] Found mismatches at indices {diff_idxs}.\n"
                f"Local (rank {self.dp_rank}): {local.tolist()}\n"
                f"Sum over ranks:     {gathered_graph_batch_size.tolist()}\n"
                f"Expected if all equal: {[v * self.dp_size for v in local.tolist()]}"
            )

    def _update_graph_pad_size(self, with_prefill, graph_pad_size):
        if not with_prefill:
            self.graph_pad_size = graph_pad_size
        else:
            super()._update_graph_pad_size(with_prefill, graph_pad_size)

    def _update_input_ids_and_positions(self, input_ids, positions,
                                        num_input_tokens, with_prefill,
                                        padded_num_tokens_across_dp):
        """Override from NPUModelRunner to update input_ids and positions"""
        input_ids, positions = super()._update_input_ids_and_positions(
            input_ids, positions, num_input_tokens, with_prefill,
            padded_num_tokens_across_dp)

        if not with_prefill:
            input_ids = self.input_ids[:padded_num_tokens_across_dp]
            positions = self.positions[:padded_num_tokens_across_dp]
        return input_ids, positions

    def _generate_process_reqs_hidden_states(self, attn_metadata, with_prefill,
                                             padded_num_tokens_across_dp,
                                             input_ids, positions,
                                             intermediate_tensors,
                                             inputs_embeds):
        model_kwargs = {
            "kv_caches": self.kv_caches,
            "attn_metadata": attn_metadata
        }
        if not with_prefill:
            if is_310p():
                converting_weight_acl_format(self.model, ACL_FORMAT_FRACTAL_NZ)
            compiled_model = self._get_torchair_lazy_compiled_model(
                padded_num_tokens_across_dp)
            hidden_states = compiled_model(
                input_ids=input_ids,
                positions=positions,
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=inputs_embeds,
                **model_kwargs,
            )
        else:
            assert self.model is not None
            if is_310p():
                converting_weight_acl_format(self.model, ACL_FORMAT_FRACTAL_ND)

            hidden_states = self.model(
                input_ids=input_ids,
                positions=positions,
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=inputs_embeds,
                **model_kwargs,
            )
        return hidden_states

    def _get_torchair_lazy_compiled_model(self, batch_size: int):
        if batch_size < 0 or batch_size > self.torchair_graph_batch_sizes[-1]:
            raise ValueError(
                f"Bad graph batch size:{batch_size}! max_graph_batch_sizes:{self.torchair_graph_batch_sizes[-1]}"
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
        if get_ascend_config().torchair_graph_config.mode:
            config.mode = get_ascend_config().torchair_graph_config.mode
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
                    cache_dir=TORCHAIR_CACHE_DIR,
                    config=config,
                    ge_cache=False)
            return self.torchair_compiled_models[batch_size]

    def init_torchair_graph_batch_sizes(self):
        start_graph_batch_size = 4
        tp_size = get_tensor_model_parallel_world_size()

        # NOTE: When use all2all | mc2, We need to slice the `num_tokens` dimension into `tp_size` blocks
        start_graph_batch_size = max(start_graph_batch_size, tp_size)

        while (start_graph_batch_size <= self.max_num_reqs):
            self.torchair_graph_batch_sizes.append(start_graph_batch_size)
            start_graph_batch_size *= 2

    def select_torchair_padded_batch_size(self, batch_size: int):
        for padded_batch_size in self.torchair_graph_batch_sizes:
            if batch_size <= padded_batch_size:
                # we treat batch_size as num of requests
                return padded_batch_size
        raise ValueError(
            f"cur batch_size is invalid, torchair_graph_batch_sizes is "
            f"{self.torchair_graph_batch_sizes}, but cur batch_size is {batch_size}."
        )

    def update_torchair_graph_batch_sizes(self):
        # return graph_batch_sizes according to the max number of tokens
        # first pad according to the number of requests
        if len(self.torchair_graph_batch_sizes) == 0:
            self.torchair_graph_batch_sizes = [1, self.max_num_reqs]
        else:
            self.torchair_graph_batch_sizes = sorted(
                self.torchair_graph_batch_sizes)
            while self.torchair_graph_batch_sizes[-1] > self.max_num_reqs:
                self.torchair_graph_batch_sizes.pop()
                if len(self.torchair_graph_batch_sizes) == 0:
                    logger.warning(
                        "torch_graph_batch_sizes is invalid, reset it to [1, max_num_seqs]"
                    )
                    self.torchair_graph_batch_sizes = [1, self.max_num_reqs]
            if self.torchair_graph_batch_sizes[-1] < self.max_num_reqs:
                self.torchair_graph_batch_sizes.append(self.max_num_reqs)

        # padded max number tokens = max_num_req * decode_token_per_req
        self.torchair_graph_batch_sizes = [
            graph_batch_size * self.decode_token_per_req
            for graph_batch_size in self.torchair_graph_batch_sizes
        ]

        # NOTE: when enable_expert_parallel on A3, we need to check if `graph_batch_size` is divisible by `tp_size`
        # Because we use x_active_mask for dispatch/combine op on A3, which requires that input shape should be same
        # on all EP ranks
        if get_ascend_soc_version(
        ) == AscendSocVersion.A3 and self.parallel_config.enable_expert_parallel:
            self._align_graph_size_divisible_by_tp_size()

    def _align_graph_size_divisible_by_tp_size(self):
        tp_size = self.parallel_config.tensor_parallel_size
        new_graph_batch_sizes = []
        for graph_batch_size in self.torchair_graph_batch_sizes:
            cur_graph_batch_size = (graph_batch_size + tp_size -
                                    1) // tp_size * tp_size
            # MTP > 1: Cal LCMLeast Common Multiple with graph_batch_size and tp_size,
            # Both adapter multi-dp and FIA operator
            if self.speculative_config is not None and self.speculative_config.num_speculative_tokens > 1:
                cur_graph_batch_size = (tp_size * graph_batch_size) \
                                       // math.gcd(tp_size, graph_batch_size)
            if cur_graph_batch_size not in new_graph_batch_sizes and \
                cur_graph_batch_size <= self.scheduler_config.max_num_batched_tokens:
                new_graph_batch_sizes.append(cur_graph_batch_size)
            elif cur_graph_batch_size > self.scheduler_config.max_num_batched_tokens \
                    and self.decode_token_per_req > 1:
                logger.warning(
                    f"torchair_graph_batch_sizes {cur_graph_batch_size} is bigger than max_num_batched_tokens",
                    f"{self.scheduler_config.max_num_batched_tokens} will skip this batch size."
                )
        new_max_num_reqs = max(new_graph_batch_sizes)
        if self.max_num_reqs != new_max_num_reqs:
            logger.warning(f"max_num_reqs is updated to {new_max_num_reqs}")
            self.max_num_reqs = new_max_num_reqs
            self.scheduler_config.max_num_seqs = new_max_num_reqs

        if new_graph_batch_sizes != self.torchair_graph_batch_sizes:
            logger.warning(
                f"torchair_graph_batch_sizes are updated to {new_graph_batch_sizes}."
            )
            self.torchair_graph_batch_sizes = new_graph_batch_sizes

    def _build_drafter_prepare_inputs_torchair_param(self):
        return True

    def get_dp_padding(self, num_tokens):
        """Override from NPUModelRunner to get dp padding"""
        return 0, None
