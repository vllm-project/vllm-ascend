# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Bundled AOT cache for the Ascend FX fusion backend."""

import functools
from collections.abc import Callable

import torch
from torch._dynamo.backends.common import aot_autograd
from torch._dynamo.utils import counters
from torch._functorch._aot_autograd.schemas import SerializableAOTDispatchCompiler
from torch._inductor.compile_fx import graph_returns_tuple, make_graph_return_tuple
from torch._inductor.output_code import RegionalOutputCode
from torch.fx._graph_pickler import GraphPickler, Options
from vllm.logger import logger


class FusionCacheSerializationError(Exception):
    """The fused graph cannot be persisted; retain the uncached backend."""


class AscendFusionOutputCode(RegionalOutputCode):
    # AOT runtime wrappers pass one list of tensors and symbolic scalars.
    _boxed_call = True

    def prepare_for_serialization(self) -> None:
        if self._graph_module is not None:
            try:
                # Ascend and vLLM register their named ops before loading the cache.
                # PyTorch's default filter only accepts aten and fbgemm operators.
                self._serialized_graph_module = GraphPickler.dumps(self._graph_module, Options(ops_filter=None))
            except Exception as error:
                raise FusionCacheSerializationError(str(error)) from error
            self._graph_module = None


def compile_cached_fusion(graph: torch.fx.GraphModule, example_inputs: list, inner_compile: Callable) -> Callable:
    if not graph_returns_tuple(graph):
        return make_graph_return_tuple(
            graph, example_inputs, functools.partial(compile_cached_fusion, inner_compile=inner_compile)
        )

    # Triton kernel-side-table indices are process-local. Never persist them.
    if any(
        "triton_kernel_wrapper" in str(node.target)
        for module in graph.modules()
        if isinstance(module, torch.fx.GraphModule)
        for node in module.graph.nodes
    ):
        return aot_autograd(fw_compiler=inner_compile)(graph, example_inputs)

    def compile_serializable(graph, inputs):
        return AscendFusionOutputCode(inner_compile(graph, inputs))

    compiler = SerializableAOTDispatchCompiler(AscendFusionOutputCode, compile_serializable)
    try:
        with torch._functorch.config.patch(
            enable_autograd_cache=True, enable_remote_autograd_cache=False, bundled_autograd_cache=True
        ):
            hits = counters["aot_autograd"]["autograd_cache_hit"]
            # AotAutograd mutates a SerializableAOTDispatchCompiler supplied as
            # the backward compiler. Sharing it with the forward compiler would
            # let Dynamo's disable wrapper strip AOT's runtime wrappers on misses.
            backward_compiler = SerializableAOTDispatchCompiler(AscendFusionOutputCode, compile_serializable)
            result = aot_autograd(fw_compiler=compiler, bw_compiler=backward_compiler)(graph, example_inputs)
            if counters["aot_autograd"]["autograd_cache_hit"] > hits:
                logger.info_once("Loaded Ascend fusion AOT graph from cache.")
            return result
    except FusionCacheSerializationError as error:
        logger.warning("Ascend fusion graph cannot be cached; using uncached compilation: %s", error)
        return aot_autograd(fw_compiler=inner_compile)(graph, example_inputs)
