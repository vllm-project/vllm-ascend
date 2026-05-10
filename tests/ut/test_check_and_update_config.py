import os
import json
import pytest

from contextlib import ExitStack
from unittest.mock import MagicMock, patch

from vllm.config.compilation import CompilationMode, CUDAGraphMode
from tests.ut.base import PytestBase
from argparse import ArgumentError

from vllm.config import VllmConfig
from vllm.engine.arg_utils import EngineArgs
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.v1.attention.backends.registry import AttentionBackendEnum

SPECULATIVE_MODELS = [
    ("JackFram/llama-68m", "abhigoyal/vllm-medusa-llama-68m-random", False),
]
class TestCheckAndUpdateConfigPartial(PytestBase):
    """Tests for check_and_update_config method (lines 285-345)"""


    @pytest.mark.parametrize("additional_config, enforce_eager, parallel_config_enable_eplb, compilation_config_cudagraph_mode, speculative_config_target_model,speculative_config_draft_model,speculative_config_enforce_eager, attention_backend, expected_cudagraph_mode", [
        (True, True, False, CUDAGraphMode.FULL, "JackFram/llama-68m", "abhigoyal/vllm-medusa-llama-68m-random", False,  AttentionBackendEnum.XPU_MLA_SPARSE, CUDAGraphMode.PIECEWISE),
        (False, True, False, CUDAGraphMode.FULL, "JackFram/llama-68m", "abhigoyal/vllm-medusa-llama-68m-random", False,  AttentionBackendEnum.XPU_MLA_SPARSE, CUDAGraphMode.FULL),
        (True, False, False, CUDAGraphMode.FULL, "JackFram/llama-68m", "abhigoyal/vllm-medusa-llama-68m-random", False,  AttentionBackendEnum.XPU_MLA_SPARSE, CUDAGraphMode.PIECEWISE),
        (False, False, False, CUDAGraphMode.FULL, "JackFram/llama-68m", "abhigoyal/vllm-medusa-llama-68m-random", False,  AttentionBackendEnum.XPU_MLA_SPARSE, CUDAGraphMode.FULL),
        (True, True, False, CUDAGraphMode.FULL_DECODE_ONLY, "JackFram/llama-68m", "abhigoyal/vllm-medusa-llama-68m-random", False,  AttentionBackendEnum.XPU_MLA_SPARSE, CUDAGraphMode.PIECEWISE),
        (False, True, False, CUDAGraphMode.FULL_DECODE_ONLY, "JackFram/llama-68m", "abhigoyal/vllm-medusa-llama-68m-random", False,  AttentionBackendEnum.XPU_MLA_SPARSE, CUDAGraphMode.FULL_DECODE_ONLY),
        (True, False, False, CUDAGraphMode.FULL_DECODE_ONLY, "JackFram/llama-68m", "abhigoyal/vllm-medusa-llama-68m-random", False,  AttentionBackendEnum.XPU_MLA_SPARSE, CUDAGraphMode.PIECEWISE),
        (False, False, False, CUDAGraphMode.FULL_DECODE_ONLY, "JackFram/llama-68m", "abhigoyal/vllm-medusa-llama-68m-random", False,  AttentionBackendEnum.XPU_MLA_SPARSE, CUDAGraphMode.FULL_DECODE_ONLY),
        (True, True, False, CUDAGraphMode.FULL_AND_PIECEWISE, "JackFram/llama-68m", "abhigoyal/vllm-medusa-llama-68m-random", False,  AttentionBackendEnum.XPU_MLA_SPARSE, CUDAGraphMode.PIECEWISE),
        (False, True, False, CUDAGraphMode.FULL_AND_PIECEWISE, "JackFram/llama-68m", "abhigoyal/vllm-medusa-llama-68m-random", False,  AttentionBackendEnum.XPU_MLA_SPARSE, CUDAGraphMode.PIECEWISE),
        (True, False, False, CUDAGraphMode.FULL_AND_PIECEWISE, "JackFram/llama-68m", "abhigoyal/vllm-medusa-llama-68m-random", False,  AttentionBackendEnum.XPU_MLA_SPARSE, CUDAGraphMode.PIECEWISE),
        (False, False, False, CUDAGraphMode.FULL_AND_PIECEWISE, "JackFram/llama-68m", "abhigoyal/vllm-medusa-llama-68m-random", False,  AttentionBackendEnum.XPU_MLA_SPARSE, CUDAGraphMode.PIECEWISE),
   
    ])
    def test_cuda_graph_from_cli(self, additional_config, enforce_eager, parallel_config_enable_eplb, compilation_config_cudagraph_mode, speculative_config_target_model,speculative_config_draft_model,speculative_config_enforce_eager, attention_backend, expected_cudagraph_mode):
        parser = EngineArgs.add_cli_args(FlexibleArgumentParser())
        args = parser.parse_args([])
        vllm_config = EngineArgs.from_cli_args(args=args).create_engine_config()

        # Turn it off possible with flag.
        args = parser.parse_args(["--attention-backend",attention_backend.name])
        vllm_config = EngineArgs.from_cli_args(args=args).create_engine_config()
      
        if enforce_eager:
            args = parser.parse_args(["--enforce-eager"])
            vllm_config = EngineArgs.from_cli_args(args=args).create_engine_config()
          
        compilation_config = {
            "cudagraph_mode": compilation_config_cudagraph_mode.name
        }
        compilation_config_str = json.dumps(compilation_config)
        args = parser.parse_args(["--compilation-config", compilation_config_str])
        vllm_config = EngineArgs.from_cli_args(args=args).create_engine_config()

        if parallel_config_enable_eplb:
            args = parser.parse_args(["--enable-eplb"])
            vllm_config = EngineArgs.from_cli_args(args=args).create_engine_config()
      
        if additional_config:
            args = parser.parse_args([
                "--additional-config", '{"enable_cuda_graph": true}'
            ])
            vllm_config = EngineArgs.from_cli_args(args=args).create_engine_config()
  
        if speculative_config_enforce_eager:
            args = parser.parse_args([
                "--speculative-config", '{"model": speculative_config_draft_model, "enforce_eager": true,"num_speculative_tokens": 1}'
            ])
            vllm_config = EngineArgs.from_cli_args(args=args).create_engine_config()
    
        assert vllm_config.compilation_config.cudagraph_mode == expected_cudagraph_mode, (
            "V1 vllm_config.compilation_config.cudagraph_mode"
        )