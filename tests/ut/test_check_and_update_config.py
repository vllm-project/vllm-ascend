import os
import json
import pytest

from contextlib import ExitStack
from unittest.mock import MagicMock, patch

from vllm.config.compilation import CompilationMode, CUDAGraphMode
from tests.ut.base import PytestBase
from argparse import ArgumentError

from vllm.config import VllmConfig
from vllm.config.model import ModelConfig
from vllm.engine.arg_utils import EngineArgs
from vllm.utils.argparse_utils import FlexibleArgumentParser

# Import the actual backend classes from platform.py
from vllm_ascend.attention.attention_v1 import AscendAttentionBackend
from vllm_ascend.attention.mla_v1 import AscendMLABackend
from vllm_ascend.attention.sfa_v1 import AscendSFABackend

SPECULATIVE_MODELS = [
    ("JackFram/llama-68m", "abhigoyal/vllm-medusa-llama-68m-random", False),
]
class TestCheckAndUpdateConfigPartial(PytestBase):
    """Tests for check_and_update_config method (lines 285-345)"""


    @pytest.mark.parametrize("enforce_eager, parallel_config_enable_eplb, compilation_config_mode, compilation_config_cudagraph_mode, speculative_config_target_model,speculative_config_draft_model,speculative_config_enforce_eager, speculative_method, is_encoder_decoder, attention_backend, expected_cudagraph_mode", [
        # Test cases with enable_eplb=False - decoder-only models with platform.py attention backends
        # Platform.py backend: (False, False) -> AscendAttentionBackend
        (True, False, CompilationMode.VLLM_COMPILE, CUDAGraphMode.FULL, "JackFram/llama-68m", "abhigoyal/vllm-medusa-llama-68m-random", False, None, False, AscendAttentionBackend, CUDAGraphMode.NONE),
        (False, False, CompilationMode.VLLM_COMPILE, CUDAGraphMode.FULL, "JackFram/llama-68m", "abhigoyal/vllm-medusa-llama-68m-random", False, None, False, AscendAttentionBackend, CUDAGraphMode.FULL),
        (True, False, CompilationMode.VLLM_COMPILE, CUDAGraphMode.FULL_DECODE_ONLY, "JackFram/llama-68m", "abhigoyal/vllm-medusa-llama-68m-random", False, None, False, AscendAttentionBackend, CUDAGraphMode.NONE),
        (False, False, CompilationMode.VLLM_COMPILE, CUDAGraphMode.FULL_DECODE_ONLY, "JackFram/llama-68m", "abhigoyal/vllm-medusa-llama-68m-random", False, None, False, AscendAttentionBackend, CUDAGraphMode.FULL_DECODE_ONLY),
        (True, False, CompilationMode.VLLM_COMPILE, CUDAGraphMode.FULL_AND_PIECEWISE, "JackFram/llama-68m", "abhigoyal/vllm-medusa-llama-68m-random", False, None, False, AscendAttentionBackend, CUDAGraphMode.NONE),
        (False, False, CompilationMode.VLLM_COMPILE, CUDAGraphMode.FULL_AND_PIECEWISE, "JackFram/llama-68m", "abhigoyal/vllm-medusa-llama-68m-random", False, None, False, AscendAttentionBackend, CUDAGraphMode.PIECEWISE),
        
        # Platform.py backend: (True, False) -> AscendMLABackend
        (True, False, CompilationMode.VLLM_COMPILE, CUDAGraphMode.FULL, "JackFram/llama-68m", "abhigoyal/vllm-medusa-llama-68m-random", False, None, False, AscendMLABackend, CUDAGraphMode.NONE),
        (False, False, CompilationMode.VLLM_COMPILE, CUDAGraphMode.FULL, "JackFram/llama-68m", "abhigoyal/vllm-medusa-llama-68m-random", False, None, False, AscendMLABackend, CUDAGraphMode.FULL),
        (True, False, CompilationMode.VLLM_COMPILE, CUDAGraphMode.FULL_DECODE_ONLY, "JackFram/llama-68m", "abhigoyal/vllm-medusa-llama-68m-random", False, None, False, AscendMLABackend, CUDAGraphMode.NONE),
        (False, False, CompilationMode.VLLM_COMPILE, CUDAGraphMode.FULL_DECODE_ONLY, "JackFram/llama-68m", "abhigoyal/vllm-medusa-llama-68m-random", False, None, False, AscendMLABackend, CUDAGraphMode.FULL_DECODE_ONLY),
        (True, False, CompilationMode.VLLM_COMPILE, CUDAGraphMode.FULL_AND_PIECEWISE, "JackFram/llama-68m", "abhigoyal/vllm-medusa-llama-68m-random", False, None, False, AscendMLABackend, CUDAGraphMode.NONE),
        (False, False, CompilationMode.VLLM_COMPILE, CUDAGraphMode.FULL_AND_PIECEWISE, "JackFram/llama-68m", "abhigoyal/vllm-medusa-llama-68m-random", False, None, False, AscendMLABackend, CUDAGraphMode.PIECEWISE),
        
        # Platform.py backend: (True, True) -> AscendSFABackend (sparse)
        (True, False, CompilationMode.VLLM_COMPILE, CUDAGraphMode.FULL, "JackFram/llama-68m", "abhigoyal/vllm-medusa-llama-68m-random", False, None, False, AscendSFABackend, CUDAGraphMode.NONE),
        (False, False, CompilationMode.VLLM_COMPILE, CUDAGraphMode.FULL, "JackFram/llama-68m", "abhigoyal/vllm-medusa-llama-68m-random", False, None, False, AscendSFABackend, CUDAGraphMode.FULL),
        (True, False, CompilationMode.VLLM_COMPILE, CUDAGraphMode.FULL_DECODE_ONLY, "JackFram/llama-68m", "abhigoyal/vllm-medusa-llama-68m-random", False, None, False, AscendSFABackend, CUDAGraphMode.NONE),
        (False, False, CompilationMode.VLLM_COMPILE, CUDAGraphMode.FULL_DECODE_ONLY, "JackFram/llama-68m", "abhigoyal/vllm-medusa-llama-68m-random", False, None, False, AscendSFABackend, CUDAGraphMode.FULL_DECODE_ONLY),
        (True, False, CompilationMode.VLLM_COMPILE, CUDAGraphMode.FULL_AND_PIECEWISE, "JackFram/llama-68m", "abhigoyal/vllm-medusa-llama-68m-random", False, None, False, AscendSFABackend, CUDAGraphMode.NONE),
        (False, False, CompilationMode.VLLM_COMPILE, CUDAGraphMode.FULL_AND_PIECEWISE, "JackFram/llama-68m", "abhigoyal/vllm-medusa-llama-68m-random", False, None, False, AscendSFABackend, CUDAGraphMode.PIECEWISE),
        
        # Test cases with enable_eplb=True - decoder-only models with platform.py attention backends
        # Platform.py backend: (False, False) -> AscendAttentionBackend 
        #todo:fix
        (True, True, CompilationMode.VLLM_COMPILE, CUDAGraphMode.FULL, "JackFram/llama-68m", "abhigoyal/vllm-medusa-llama-68m-random", False, None, False, AscendAttentionBackend, CUDAGraphMode.NONE),
        (False, True, CompilationMode.VLLM_COMPILE, CUDAGraphMode.FULL, "JackFram/llama-68m", "abhigoyal/vllm-medusa-llama-68m-random", False, None, False, AscendAttentionBackend, CUDAGraphMode.FULL),
        (True, True, CompilationMode.VLLM_COMPILE, CUDAGraphMode.FULL_DECODE_ONLY, "JackFram/llama-68m", "abhigoyal/vllm-medusa-llama-68m-random", False, None, False, AscendAttentionBackend, CUDAGraphMode.NONE),
        (False, True, CompilationMode.VLLM_COMPILE, CUDAGraphMode.FULL_DECODE_ONLY, "JackFram/llama-68m", "abhigoyal/vllm-medusa-llama-68m-random", False, None, False, AscendAttentionBackend, CUDAGraphMode.FULL_DECODE_ONLY),
        (True, True, CompilationMode.VLLM_COMPILE, CUDAGraphMode.FULL_AND_PIECEWISE, "JackFram/llama-68m", "abhigoyal/vllm-medusa-llama-68m-random", False, None, False, AscendAttentionBackend, CUDAGraphMode.NONE),
        (False, True, CompilationMode.VLLM_COMPILE, CUDAGraphMode.FULL_AND_PIECEWISE, "JackFram/llama-68m", "abhigoyal/vllm-medusa-llama-68m-random", False, None, False, AscendAttentionBackend, CUDAGraphMode.PIECEWISE),
        
        # Platform.py backend: (True, False) -> AscendMLABackend
        (True, True, CompilationMode.VLLM_COMPILE, CUDAGraphMode.FULL, "JackFram/llama-68m", "abhigoyal/vllm-medusa-llama-68m-random", False, None, False, AscendMLABackend, CUDAGraphMode.NONE),
        (False, True, CompilationMode.VLLM_COMPILE, CUDAGraphMode.FULL, "JackFram/llama-68m", "abhigoyal/vllm-medusa-llama-68m-random", False, None, False, AscendMLABackend, CUDAGraphMode.FULL),
        (True, True, CompilationMode.VLLM_COMPILE, CUDAGraphMode.FULL_DECODE_ONLY, "JackFram/llama-68m", "abhigoyal/vllm-medusa-llama-68m-random", False, None, False, AscendMLABackend, CUDAGraphMode.NONE),
        (False, True, CompilationMode.VLLM_COMPILE, CUDAGraphMode.FULL_DECODE_ONLY, "JackFram/llama-68m", "abhigoyal/vllm-medusa-llama-68m-random", False, None, False, AscendMLABackend, CUDAGraphMode.FULL_DECODE_ONLY),
        (True, True, CompilationMode.VLLM_COMPILE, CUDAGraphMode.FULL_AND_PIECEWISE, "JackFram/llama-68m", "abhigoyal/vllm-medusa-llama-68m-random", False, None, False, AscendMLABackend, CUDAGraphMode.NONE),
        (False, True, CompilationMode.VLLM_COMPILE, CUDAGraphMode.FULL_AND_PIECEWISE, "JackFram/llama-68m", "abhigoyal/vllm-medusa-llama-68m-random", False, None, False, AscendMLABackend, CUDAGraphMode.PIECEWISE),
        
        # Platform.py backend: (True, True) -> AscendSFABackend (sparse)
        (True, True, CompilationMode.VLLM_COMPILE, CUDAGraphMode.FULL, "JackFram/llama-68m", "abhigoyal/vllm-medusa-llama-68m-random", False, None, False, AscendSFABackend, CUDAGraphMode.NONE),
        (False, True, CompilationMode.VLLM_COMPILE, CUDAGraphMode.FULL, "JackFram/llama-68m", "abhigoyal/vllm-medusa-llama-68m-random", False, None, False, AscendSFABackend, CUDAGraphMode.FULL),
        (True, True, CompilationMode.VLLM_COMPILE, CUDAGraphMode.FULL_DECODE_ONLY, "JackFram/llama-68m", "abhigoyal/vllm-medusa-llama-68m-random", False, None, False, AscendSFABackend, CUDAGraphMode.NONE),
        (False, True, CompilationMode.VLLM_COMPILE, CUDAGraphMode.FULL_DECODE_ONLY, "JackFram/llama-68m", "abhigoyal/vllm-medusa-llama-68m-random", False, None, False, AscendSFABackend, CUDAGraphMode.FULL_DECODE_ONLY),
        (True, True, CompilationMode.VLLM_COMPILE, CUDAGraphMode.FULL_AND_PIECEWISE, "JackFram/llama-68m", "abhigoyal/vllm-medusa-llama-68m-random", False, None, False, AscendSFABackend, CUDAGraphMode.NONE),
        (False, True, CompilationMode.VLLM_COMPILE, CUDAGraphMode.FULL_AND_PIECEWISE, "JackFram/llama-68m", "abhigoyal/vllm-medusa-llama-68m-random", False, None, False, AscendSFABackend, CUDAGraphMode.PIECEWISE),
        
        # Test cases with different speculative decoding methods - decoder-only models with platform.py attention backends
        # Suffix method configuration (from E2E test)
        # Platform.py backend: (False, False) -> AscendAttentionBackend
        (False, False, CompilationMode.VLLM_COMPILE, CUDAGraphMode.FULL, "JackFram/llama-68m", "JackFram/llama-68m", True, "suffix", False, AscendAttentionBackend, CUDAGraphMode.FULL),
        (False, True, CompilationMode.VLLM_COMPILE, CUDAGraphMode.FULL_DECODE_ONLY, "JackFram/llama-68m", "JackFram/llama-68m", True, "suffix", False, AscendAttentionBackend, CUDAGraphMode.FULL_DECODE_ONLY),
        
        # # Eagle3 method configuration (from E2E test)
        # # Platform.py backend: (True, False) -> AscendMLABackend
        (False, False, CompilationMode.VLLM_COMPILE, CUDAGraphMode.FULL, "JackFram/llama-68m", "abhigoyal/vllm-medusa-llama-68m-random", True, "eagle3", False, AscendMLABackend, CUDAGraphMode.FULL),
        (False, True, CompilationMode.VLLM_COMPILE, CUDAGraphMode.FULL_DECODE_ONLY, "JackFram/llama-68m", "abhigoyal/vllm-medusa-llama-68m-random", True, "eagle3", False, AscendMLABackend, CUDAGraphMode.FULL_DECODE_ONLY),
        
        # # Ngram method configuration (from E2E test)
        # # Platform.py backend: (True, True) -> AscendSFABackend (sparse)
        (False, False, CompilationMode.VLLM_COMPILE, CUDAGraphMode.FULL, "JackFram/llama-68m", "JackFram/llama-68m", True, "ngram", False, AscendSFABackend, CUDAGraphMode.FULL),
        (False, True, CompilationMode.VLLM_COMPILE, CUDAGraphMode.FULL_DECODE_ONLY, "JackFram/llama-68m", "JackFram/llama-68m", True, "ngram", False, AscendSFABackend, CUDAGraphMode.FULL_DECODE_ONLY),
        
        # Test cases with encoder-decoder models
        # Platform.py backend: (True, True) -> AscendSFABackend (sparse)
        (False, False, CompilationMode.VLLM_COMPILE, CUDAGraphMode.FULL, "t5-small", "t5-small", False, None, True, AscendSFABackend, CUDAGraphMode.FULL),
        # Platform.py backend: (False, False) -> AscendAttentionBackend
        (False, False, CompilationMode.VLLM_COMPILE, CUDAGraphMode.FULL_DECODE_ONLY, "t5-small", "t5-small", False, None, True, AscendAttentionBackend, CUDAGraphMode.FULL_DECODE_ONLY),
        # Platform.py backend: (True, True) -> AscendSFABackend (sparse)
        (True, False, CompilationMode.VLLM_COMPILE, CUDAGraphMode.FULL_AND_PIECEWISE, "t5-small", "t5-small", False, None, True, AscendSFABackend, CUDAGraphMode.NONE),
        
        # Test cases with different compilation modes
        # Platform.py backend: (True, True) -> AscendSFABackend (sparse)
        (False, False, CompilationMode.NONE, CUDAGraphMode.FULL, "JackFram/llama-68m", "abhigoyal/vllm-medusa-llama-68m-random", False, None, False, AscendSFABackend, CUDAGraphMode.FULL),
        # Platform.py backend: (False, False) -> AscendAttentionBackend
        (False, False, CompilationMode.PYTORCH_INDUCTOR, CUDAGraphMode.FULL_DECODE_ONLY, "JackFram/llama-68m", "abhigoyal/vllm-medusa-llama-68m-random", False, None, False, AscendAttentionBackend, CUDAGraphMode.FULL_DECODE_ONLY),
    ])
    def test_cuda_graph_from_cli(self, enforce_eager, parallel_config_enable_eplb, compilation_config_mode, compilation_config_cudagraph_mode, speculative_config_target_model,speculative_config_draft_model,speculative_config_enforce_eager, speculative_method, is_encoder_decoder, attention_backend, expected_cudagraph_mode):
        # Mock list_filtered_repo_files and is_encoder_decoder
        with ExitStack() as stack:
            stack.enter_context(patch('vllm.transformers_utils.repo_utils.list_filtered_repo_files', return_value=[]))
            
            # Patch the is_encoder_decoder function and property to use the test case value
            # This ensures the scheduler_config uses the correct value from test parametrization
            stack.enter_context(patch('vllm.config.model.is_encoder_decoder', return_value=is_encoder_decoder))
            
            # For the cached_property, we need to create a mock property
            mock_is_encoder_decoder = property(lambda self: is_encoder_decoder)
            stack.enter_context(patch.object(ModelConfig, 'is_encoder_decoder', mock_is_encoder_decoder))
            parser = EngineArgs.add_cli_args(FlexibleArgumentParser())
            
            # Build a single list of CLI arguments to test combination of parameters
            cli_args = []
            
            # Add attention backend - map actual backend classes to CLI strings
            if attention_backend == AscendAttentionBackend:
                cli_backend = "CUSTOM"
            elif attention_backend == AscendMLABackend:
                cli_backend = "TRITON_MLA"
            elif attention_backend == AscendSFABackend:
                cli_backend = "XPU_MLA_SPARSE"
            else:
                raise ValueError(f"Unknown attention backend: {attention_backend}")
            cli_args.extend(["--attention-backend", cli_backend])
            
            # Add enforce eager if needed
            if enforce_eager:
                cli_args.append("--enforce-eager")
            
            # Add compilation config
            compilation_config = {
                "mode": compilation_config_mode.name,
                "cudagraph_mode": compilation_config_cudagraph_mode.name
            }
            compilation_config_str = json.dumps(compilation_config)
            cli_args.extend(["--compilation-config", compilation_config_str])
            
            # Add EPLB if needed
            if parallel_config_enable_eplb:
                cli_args.append("--enable-eplb")
            
            # Add speculative config if needed
            if speculative_config_enforce_eager:
                speculative_config = {
                    "enforce_eager": True,
                    "num_speculative_tokens": 1
                }
                # Add method-specific configuration based on E2E test examples
                if speculative_method is not None:
                    speculative_config["method"] = speculative_method
                    
                    # Configure method-specific parameters
                    if speculative_method == "suffix":
                        # Suffix method configuration from E2E test
                        speculative_config["num_speculative_tokens"] = 8
                    elif speculative_method == "eagle3":
                        # Eagle3 method configuration from E2E test
                        speculative_config["model"] = speculative_config_draft_model
                        speculative_config["num_speculative_tokens"] = 2
                    elif speculative_method == "ngram":
                        # Ngram method configuration from E2E test
                        speculative_config["prompt_lookup_max"] = 5
                        speculative_config["prompt_lookup_min"] = 3
                        speculative_config["num_speculative_tokens"] = 3
                    elif speculative_method == "mtp":
                        # MTP method uses draft model
                        speculative_config["model"] = speculative_config_draft_model
                
                speculative_config_str = json.dumps(speculative_config)
                cli_args.extend(["--speculative-config", speculative_config_str])
            
            # Parse all arguments at once and create engine config
            args = parser.parse_args(cli_args)
            vllm_config = EngineArgs.from_cli_args(args=args).create_engine_config()
            # print(f"############# 1 {vllm_config.compilation_config.cudagraph_mode= } {expected_cudagraph_mode=}")
            # print(f"############# 2 {vllm_config=}")
            # print(f"############# 3 {args=}")
            # print(f"############# 4 {vllm_config.model_config.is_encoder_decoder=}")
            assert vllm_config.compilation_config.cudagraph_mode == expected_cudagraph_mode, "V1 vllm_config.compilation_config.cudagraph_mode"

