import hashlib
from typing import Any, Callable, Optional, Union

import torch
import torch_npu
import torchair
from torchair import patch_for_hcom
from vllm.config import CompilationLevel, VllmConfig
from vllm.logger import init_logger

logger = init_logger(__name__)

MAX_GEAR_NUM = 6

BLOCK_NUM_FLOATING_RANGE = 30


def get_torchair_config():
    patch_for_hcom()
    config = torchair.CompilerConfig()
    config.experimental_config.frozen_parameter = True
    config.experimental_config.tiling_schedule_optimize = True
    config.ge_config.optimization_switch = "InplaceAddRmsNormFusionPass:off"
    torch.npu.set_compile_mode(jit_compile=False)
    return config


class NPUCompilationConfig:
    level: int = 0
    """The level of compilation:

    - 0: no compilation.
    - 1: dynamo as is.
    - 2: dynamo once.
    - 3: piecewise compilation."""

    backend: Optional[str] = None
    """The backend for compilation."""

    use_aclgraph: bool = False
    """Whether to use aclgraph inside compilation.
    - False: aclgraph inside compilation is not used.
    - True: aclgraph inside compilation is used.
    In the vLLM V1 Engine, this flag only applies for
    CompilationLevel.PIECEWISE (aka -O3)."""

    aclgraph_num_of_warmups: int = 0
    """Number of warmup runs for aclgraph.
    It means the first several runs will be treated as warmup runs.
    Only after that, the execution will be recorded, and the recorded
    cudagraph will be used for subsequent runs."""

    aclgraph_capture_sizes: Optional[list[int]] = None
    """Sizes to capture aclgraph.
    - None (default): capture sizes are inferred from vllm config.
    - list[int]: capture sizes are specified as given."""

    use_ge_graph_cached: bool = False
    """Whether to use ge backend graph caching."""

    decode_gear_list: Optional[list[int]] = None
    """The gear size of the different static plots"""

    block_num_floating_range: int = BLOCK_NUM_FLOATING_RANGE
    """The compilation cache allows for the range of fluctuations"""

    def build_from_cli(self, raw_graph_config: dict[str, Any],
                       vllm_config: VllmConfig):
        """Parse the CLI value for the compilation config.
        -O1, -O2, -O3, etc. is handled in FlexibleArgumentParser.
        """
        self.level = raw_graph_config.get("level",
                                          CompilationLevel.NO_COMPILATION)
        self.backend = raw_graph_config.get("backend", None)
        self.use_aclgraph = raw_graph_config.get("use_aclgraph", False)
        self.aclgraph_num_of_warmups = raw_graph_config.get(
            "aclgraph_num_of_warmups", 0)
        self.aclgraph_capture_sizes = raw_graph_config.get(
            "aclgraph_capture_sizes", None)
        self.use_ge_graph_cached = raw_graph_config.get(
            "use_ge_graph_cached", False)
        self.decode_gear_list = raw_graph_config.get("decode_gear_list", None)
        self.block_num_floating_range = raw_graph_config.get(
            "block_num_floating_range", BLOCK_NUM_FLOATING_RANGE)

        if self.aclgraph_capture_sizes and not isinstance(
                self.aclgraph_capture_sizes, list):
            raise TypeError("aclgraph_capture_sizes must be a list")

        if self.decode_gear_list and not isinstance(self.decode_gear_list,
                                                    list):
            raise TypeError("decode_gear_list must be a list")

        self.update_gear_options(vllm_config)

        logger.info(f"the NPUCompilationConfig value is: {self}")

    def update_gear_options(self, vllm_config: VllmConfig):
        max_num_reqs = vllm_config.scheduler_config.max_num_seqs
        use_spec_decode = False if not vllm_config.speculative_config else (
            vllm_config.speculative_config.method == "deepseek_mtp"
            or vllm_config.speculative_config.method == "pangu_ultra_moe_mtp")
        max_batch_size = max_num_reqs if not use_spec_decode else max_num_reqs * (
            1 + vllm_config.speculative_config.num_speculative_tokens)
        if not self.decode_gear_list:
            self.decode_gear_list = [max_batch_size]

        if len(self.decode_gear_list) > MAX_GEAR_NUM:
            raise ValueError(f"Max gear num supported is {MAX_GEAR_NUM} now.")

        if self.decode_gear_list and max(
                self.decode_gear_list) > max_batch_size:
            decode_gear_list = [
                gear for gear in self.decode_gear_list
                if gear <= max_batch_size
            ]
            logger.warning(
                f"PTA_TORCHAIR_DECODE_GEAR_LIST({self.decode_gear_list}) becomes ({decode_gear_list}) due to max_batch_size({max_batch_size})"
            )
            self.decode_gear_list = decode_gear_list

        if len(self.decode_gear_list) < MAX_GEAR_NUM and max(
                self.decode_gear_list) < max_batch_size:
            self.decode_gear_list.append(max_batch_size)

    def init_backend(self, vllm_config: VllmConfig) -> Union[str, Callable]:
        if self.level == CompilationLevel.NO_COMPILATION:
            raise ValueError("No compilation level is set.")

        if self.level in [
                CompilationLevel.DYNAMO_AS_IS, CompilationLevel.DYNAMO_ONCE
        ]:
            if not self.backend or self.backend == "":
                config = get_torchair_config()
                npu_backend = torchair.get_npu_backend(compiler_config=config)
                logger.info(f"Using torchair backend!")
                return npu_backend

            logger.info(f"Using user-defined backend!")
            return self.backend

        assert self.level == CompilationLevel.PIECEWISE

        logger.info(f"Using omni backend!")
        from omni.adaptors.vllm.compilation.backends import OmniBackend
        return OmniBackend(vllm_config)

    def compute_hash(self) -> str:
        """
        WARNING: Whenever a new field is added to this config,
        ensure that it is included in the factors list if
        it affects the computation graph.
        """
        factors: list[Any] = [
            self.level, self.backend, self.block_num_floating_range
        ]
        return hashlib.sha256(str(factors).encode()).hexdigest()
