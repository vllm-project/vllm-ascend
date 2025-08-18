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

import vllm
from vllm.config import CompilationConfig
import vllm.compilation.backends
from vllm.compilation.compiler_interface import CompilerInterface
from vllm_ascend.compilation.graph_rewrite_pass_manager import GraphRewritePassManager
from vllm_ascend.compilation.compiler_interface import AscendAdaptor


def configure_post_pass(self):
    config = self.compilation_config
    self.graph_rewriter_pass_manager = GraphRewritePassManager()
    self.graph_rewriter_pass_manager.configure(self.vllm_config)

    # Post-grad custom passes are run using the post_grad_custom_post_pass
    # hook. If a pass for that hook exists, add it to the pass manager.
    inductor_config = config.inductor_compile_config
    PASS_KEY = "graph_rewriter_manager"
    inductor_config[PASS_KEY] = self.graph_rewriter_pass_manager


def make_compiler(compilation_config: CompilationConfig) -> CompilerInterface:
    return AscendAdaptor()


vllm.compilation.backends.make_compiler = make_compiler
vllm.compilation.backends.VllmBackend.configure_post_pass = configure_post_pass

