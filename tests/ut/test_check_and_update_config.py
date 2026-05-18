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

class TestCheckAndUpdateConfigPartial(PytestBase):
    """Tests for check_and_update_config method (lines 285-345)"""


    @pytest.mark.parametrize("enforce_eager, parallel_config_tensor_parallel_size, compilation_config_mode, compilation_config_cudagraph_mode, speculative_config_enforce_eager, speculative_method, is_encoder_decoder, attention_backend, expected_cudagraph_mode", [
        # Basic test cases covering all attention backends, tensor parallel sizes, and cudagraph modes
        # AscendAttentionBackend tests
        (True, 1, CompilationMode.VLLM_COMPILE, CUDAGraphMode.FULL, False, None, False, AscendAttentionBackend, CUDAGraphMode.NONE),
        (False, 1, CompilationMode.VLLM_COMPILE, CUDAGraphMode.FULL, False, None, False, AscendAttentionBackend, CUDAGraphMode.FULL),
        (True, 2, CompilationMode.VLLM_COMPILE, CUDAGraphMode.FULL_DECODE_ONLY, False, None, False, AscendAttentionBackend, CUDAGraphMode.NONE),
        (False, 2, CompilationMode.VLLM_COMPILE, CUDAGraphMode.FULL_DECODE_ONLY, False, None, False, AscendAttentionBackend, CUDAGraphMode.FULL_DECODE_ONLY),
        (True, None, CompilationMode.VLLM_COMPILE, CUDAGraphMode.FULL_AND_PIECEWISE, False, None, False, AscendAttentionBackend, CUDAGraphMode.NONE),
        (False, None, CompilationMode.VLLM_COMPILE, CUDAGraphMode.FULL_AND_PIECEWISE, False, None, False, AscendAttentionBackend, CUDAGraphMode.PIECEWISE),
        
        AscendMLABackend tests
        (True, 1, CompilationMode.VLLM_COMPILE, CUDAGraphMode.FULL_DECODE_ONLY, False, None, False, AscendMLABackend, CUDAGraphMode.NONE),
        (False, 1, CompilationMode.VLLM_COMPILE, CUDAGraphMode.FULL_DECODE_ONLY, False, None, False, AscendMLABackend, CUDAGraphMode.FULL_DECODE_ONLY),
        (True, 2, CompilationMode.VLLM_COMPILE, CUDAGraphMode.FULL_AND_PIECEWISE, False, None, False, AscendMLABackend, CUDAGraphMode.NONE),
        (False, 2, CompilationMode.VLLM_COMPILE, CUDAGraphMode.FULL_AND_PIECEWISE, False, None, False, AscendMLABackend, CUDAGraphMode.PIECEWISE),
        (True, None, CompilationMode.VLLM_COMPILE, CUDAGraphMode.FULL, False, None, False, AscendMLABackend, CUDAGraphMode.NONE),
        (False, None, CompilationMode.VLLM_COMPILE, CUDAGraphMode.FULL, False, None, False, AscendMLABackend, CUDAGraphMode.FULL),
        
        # AscendSFABackend tests
        (True, 1, CompilationMode.VLLM_COMPILE, CUDAGraphMode.FULL_AND_PIECEWISE, False, None, False, AscendSFABackend, CUDAGraphMode.NONE),
        (False, 1, CompilationMode.VLLM_COMPILE, CUDAGraphMode.FULL_AND_PIECEWISE, False, None, False, AscendSFABackend, CUDAGraphMode.PIECEWISE),
        (True, 2, CompilationMode.VLLM_COMPILE, CUDAGraphMode.FULL, False, None, False, AscendSFABackend, CUDAGraphMode.NONE),
        (False, 2, CompilationMode.VLLM_COMPILE, CUDAGraphMode.FULL, False, None, False, AscendSFABackend, CUDAGraphMode.FULL),
        (True, None, CompilationMode.VLLM_COMPILE, CUDAGraphMode.FULL_DECODE_ONLY, False, None, False, AscendSFABackend, CUDAGraphMode.NONE),
        (False, None, CompilationMode.VLLM_COMPILE, CUDAGraphMode.FULL_DECODE_ONLY, False, None, False, AscendSFABackend, CUDAGraphMode.FULL_DECODE_ONLY),
        
        # Speculative decoding tests with balanced distribution
        # Suffix method with different backends and tensor parallel sizes
        (False, 1, CompilationMode.VLLM_COMPILE, CUDAGraphMode.FULL, True, "suffix", False, AscendAttentionBackend, CUDAGraphMode.FULL),
        (False, 2, CompilationMode.VLLM_COMPILE, CUDAGraphMode.FULL_DECODE_ONLY, True, "suffix", False, AscendMLABackend, CUDAGraphMode.FULL_DECODE_ONLY),
        (False, None, CompilationMode.VLLM_COMPILE, CUDAGraphMode.FULL_AND_PIECEWISE, True, "suffix", False, AscendSFABackend, CUDAGraphMode.PIECEWISE),
        
        # Eagle3 method with different backends and tensor parallel sizes
        (False, 1, CompilationMode.VLLM_COMPILE, CUDAGraphMode.FULL_DECODE_ONLY, True, "eagle3", False, AscendMLABackend, CUDAGraphMode.FULL_DECODE_ONLY),
        (False, 2, CompilationMode.VLLM_COMPILE, CUDAGraphMode.FULL_AND_PIECEWISE, True, "eagle3", False, AscendSFABackend, CUDAGraphMode.PIECEWISE),
        (False, None, CompilationMode.VLLM_COMPILE, CUDAGraphMode.FULL, True, "eagle3", False, AscendAttentionBackend, CUDAGraphMode.FULL),
        
        # Ngram method with different backends and tensor parallel sizes
        (False, 1, CompilationMode.VLLM_COMPILE, CUDAGraphMode.FULL_AND_PIECEWISE, True, "ngram", False, AscendSFABackend, CUDAGraphMode.PIECEWISE),
        (False, 2, CompilationMode.VLLM_COMPILE, CUDAGraphMode.FULL, True, "ngram", False, AscendAttentionBackend, CUDAGraphMode.FULL),
        (False, None, CompilationMode.VLLM_COMPILE, CUDAGraphMode.FULL_DECODE_ONLY, True, "ngram", False, AscendMLABackend, CUDAGraphMode.FULL_DECODE_ONLY),
        
        Encoder-decoder model tests with balanced distribution
        (False, 1, CompilationMode.VLLM_COMPILE, CUDAGraphMode.FULL, False, None, True, AscendAttentionBackend, CUDAGraphMode.PIECEWISE),
        (False, 2, CompilationMode.VLLM_COMPILE, CUDAGraphMode.FULL_DECODE_ONLY, False, None, True, AscendMLABackend, CUDAGraphMode.PIECEWISE),
        # (False, None, CompilationMode.VLLM_COMPILE, CUDAGraphMode.FULL_AND_PIECEWISE, False, None, True, AscendSFABackend, CUDAGraphMode.PIECEWISE),

        
        # todo error
        # (True, 1, CompilationMode.VLLM_COMPILE, CUDAGraphMode.FULL_DECODE_ONLY, False, None, True, AscendSFABackend, CUDAGraphMode.FULL_DECODE_ONLY),
        # (True, 2, CompilationMode.VLLM_COMPILE, CUDAGraphMode.FULL_AND_PIECEWISE, False, None, True, AscendAttentionBackend, CUDAGraphMode.FULL_AND_PIECEWISE),
        # (True, None, CompilationMode.VLLM_COMPILE, CUDAGraphMode.FULL, False, None, True, AscendMLABackend, CUDAGraphMode.FULL),
        
        # # Different compilation modes with balanced distribution
        (False, 1, CompilationMode.NONE, CUDAGraphMode.FULL, False, None, False, AscendAttentionBackend, CUDAGraphMode.FULL),
        (False, 2, CompilationMode.NONE, CUDAGraphMode.FULL_DECODE_ONLY, False, None, False, AscendMLABackend, CUDAGraphMode.FULL_DECODE_ONLY),
        (False, None, CompilationMode.NONE, CUDAGraphMode.FULL_AND_PIECEWISE, False, None, False, AscendSFABackend, CUDAGraphMode.NONE),
        
        (False, 1, CompilationMode.STOCK_TORCH_COMPILE, CUDAGraphMode.FULL_DECODE_ONLY, False, None, False, AscendMLABackend, CUDAGraphMode.NONE),
        (False, 2, CompilationMode.STOCK_TORCH_COMPILE, CUDAGraphMode.FULL_AND_PIECEWISE, False, None, False, AscendSFABackend, CUDAGraphMode.NONE),
        (False, None, CompilationMode.STOCK_TORCH_COMPILE, CUDAGraphMode.FULL, False, None, False, AscendAttentionBackend, CUDAGraphMode.NONE),
    ])
    def test_cuda_graph_from_cli(self, enforce_eager, parallel_config_tensor_parallel_size, compilation_config_mode, compilation_config_cudagraph_mode, speculative_config_enforce_eager, speculative_method, is_encoder_decoder, attention_backend, expected_cudagraph_mode):
        # Mock list_filtered_repo_files and is_encoder_decoder
        with ExitStack() as stack:
            stack.enter_context(patch('vllm.transformers_utils.repo_utils.list_filtered_repo_files', return_value=[]))
            
            # Patch the is_encoder_decoder function and property to use the test case value
            # This ensures the scheduler_config uses the correct value from test parametrization
            stack.enter_context(patch('vllm.config.model.is_encoder_decoder', return_value=is_encoder_decoder))
            
            # For the cached_property, we need to create a mock property
            mock_is_encoder_decoder = property(lambda self: is_encoder_decoder)
            stack.enter_context(patch.object(ModelConfig, 'is_encoder_decoder', mock_is_encoder_decoder))
            
            # Mock create_speculative_config to completely avoid model weight loading
            # This approach doesn't create a real SpeculativeConfig object
            def mock_create_speculative_config(*args, **kwargs):
                # args[0] is the self parameter when called as a method
                self_obj = args[0] if args else None
                if not self_obj or not hasattr(self_obj, 'speculative_config') or self_obj.speculative_config is None:
                    return None
                
                # Create a simple mock object that satisfies the interface
                mock_spec_config = MagicMock()
                
                # Set all attributes from the speculative config
                for key, value in self_obj.speculative_config.items():
                    setattr(mock_spec_config, key, value)
                
                # Add required attributes that might be accessed
                setattr(mock_spec_config, 'method', self_obj.speculative_config.get('method'))
                setattr(mock_spec_config, 'num_speculative_tokens', self_obj.speculative_config.get('num_speculative_tokens', 1))
                setattr(mock_spec_config, 'model', self_obj.speculative_config.get('model'))
                setattr(mock_spec_config, 'draft_model_config', MagicMock())
                setattr(mock_spec_config, 'draft_parallel_config', MagicMock())
                setattr(mock_spec_config, 'target_model_config', MagicMock())
                setattr(mock_spec_config, 'target_parallel_config', MagicMock())
                
                return mock_spec_config
            
            stack.enter_context(patch.object(EngineArgs, 'create_speculative_config', side_effect=mock_create_speculative_config))
            
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
            
            # Add tensor parallel size if specified
            if parallel_config_tensor_parallel_size is not None:
                cli_args.extend(["--tensor-parallel-size", str(parallel_config_tensor_parallel_size)])
            
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
                    # Use mock model name since we don't need actual weights
                    speculative_config["model"] = "mock-eagle3-model"
                    speculative_config["num_speculative_tokens"] = 2
                elif speculative_method == "ngram":
                    # Ngram method configuration from E2E test
                    speculative_config["prompt_lookup_max"] = 5
                    speculative_config["prompt_lookup_min"] = 3
                    speculative_config["num_speculative_tokens"] = 3
                elif speculative_method == "mtp":
                    # MTP method uses draft model
                    # Use mock model name since we don't need actual weights
                    speculative_config["model"] = "mock-mtp-model"
                
                speculative_config_str = json.dumps(speculative_config)
                cli_args.extend(["--speculative-config", speculative_config_str])
            
            # Parse all arguments at once and create engine config
            args = parser.parse_args(cli_args)
            vllm_config = EngineArgs.from_cli_args(args=args).create_engine_config()
            assert vllm_config.compilation_config.cudagraph_mode == expected_cudagraph_mode, "V1 vllm_config.compilation_config.cudagraph_mode"
