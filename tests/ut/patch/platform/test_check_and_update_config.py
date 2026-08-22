import json
from contextlib import ExitStack
from unittest.mock import MagicMock, patch

import pytest
from vllm.config.compilation import CompilationMode, CUDAGraphMode
from vllm.config.model import ModelConfig
from vllm.engine.arg_utils import EngineArgs
from vllm.utils.argparse_utils import FlexibleArgumentParser

from tests.ut.base import PytestBase


class TestCheckAndUpdateConfigPartial(PytestBase):
    """Tests for check_and_update_config method (lines 285-345)"""

    @pytest.mark.parametrize(
        "enforce_eager, parallel_config_tensor_parallel_size, compilation_config_mode, \
            compilation_config_cudagraph_mode, speculative_config_enforce_eager, \
            speculative_method, is_encoder_decoder, expected_cudagraph_mode",
        [
            # Basic test cases covering tensor parallel sizes, and cudagraph modes
            (True, 1, CompilationMode.VLLM_COMPILE, CUDAGraphMode.FULL, False, None, False, CUDAGraphMode.NONE),
            (False, 1, CompilationMode.VLLM_COMPILE, CUDAGraphMode.FULL, False, None, False, CUDAGraphMode.FULL),
            (
                True,
                2,
                CompilationMode.VLLM_COMPILE,
                CUDAGraphMode.FULL_DECODE_ONLY,
                False,
                None,
                False,
                CUDAGraphMode.NONE,
            ),
            (
                False,
                2,
                CompilationMode.VLLM_COMPILE,
                CUDAGraphMode.FULL_DECODE_ONLY,
                False,
                None,
                False,
                CUDAGraphMode.FULL_DECODE_ONLY,
            ),
            (
                True,
                None,
                CompilationMode.VLLM_COMPILE,
                CUDAGraphMode.FULL_AND_PIECEWISE,
                False,
                None,
                False,
                CUDAGraphMode.NONE,
            ),
            (
                False,
                None,
                CompilationMode.VLLM_COMPILE,
                CUDAGraphMode.FULL_AND_PIECEWISE,
                False,
                None,
                False,
                CUDAGraphMode.PIECEWISE,
            ),
            # Tests for None values in compilation_config_mode and compilation_config_cudagraph_mode
            # Both None - should use platform defaults
            (False, 1, None, None, False, None, False, CUDAGraphMode.PIECEWISE),
            # Only cudagraph_mode specified
            (False, 1, None, CUDAGraphMode.FULL, False, None, False, CUDAGraphMode.FULL),
            (False, 2, None, CUDAGraphMode.FULL_DECODE_ONLY, False, None, False, CUDAGraphMode.FULL_DECODE_ONLY),
            # Only mode specified
            (False, 1, CompilationMode.VLLM_COMPILE, None, False, None, False, CUDAGraphMode.PIECEWISE),
            (False, 1, CompilationMode.NONE, None, False, None, False, CUDAGraphMode.NONE),
            # Speculative decoding tests
            # Suffix method
            (False, 1, CompilationMode.VLLM_COMPILE, CUDAGraphMode.FULL, True, "suffix", False, CUDAGraphMode.FULL),
            (
                False,
                2,
                CompilationMode.VLLM_COMPILE,
                CUDAGraphMode.FULL_DECODE_ONLY,
                True,
                "suffix",
                False,
                CUDAGraphMode.FULL_DECODE_ONLY,
            ),
            (
                False,
                None,
                CompilationMode.VLLM_COMPILE,
                CUDAGraphMode.FULL_AND_PIECEWISE,
                True,
                "suffix",
                False,
                CUDAGraphMode.PIECEWISE,
            ),
            # Eagle3 method
            (
                False,
                1,
                CompilationMode.VLLM_COMPILE,
                CUDAGraphMode.FULL_DECODE_ONLY,
                True,
                "eagle3",
                False,
                CUDAGraphMode.FULL_DECODE_ONLY,
            ),
            (
                False,
                2,
                CompilationMode.VLLM_COMPILE,
                CUDAGraphMode.FULL_AND_PIECEWISE,
                True,
                "eagle3",
                False,
                CUDAGraphMode.PIECEWISE,
            ),
            (False, None, CompilationMode.VLLM_COMPILE, CUDAGraphMode.FULL, True, "eagle3", False, CUDAGraphMode.FULL),
            # Ngram method
            (
                False,
                1,
                CompilationMode.VLLM_COMPILE,
                CUDAGraphMode.FULL_AND_PIECEWISE,
                True,
                "ngram",
                False,
                CUDAGraphMode.PIECEWISE,
            ),
            (False, 2, CompilationMode.VLLM_COMPILE, CUDAGraphMode.FULL, True, "ngram", False, CUDAGraphMode.FULL),
            (
                False,
                None,
                CompilationMode.VLLM_COMPILE,
                CUDAGraphMode.FULL_DECODE_ONLY,
                True,
                "ngram",
                False,
                CUDAGraphMode.FULL_DECODE_ONLY,
            ),
            # Encoder-decoder model tests
            (False, 1, CompilationMode.VLLM_COMPILE, CUDAGraphMode.FULL, False, None, True, CUDAGraphMode.PIECEWISE),
            (
                False,
                2,
                CompilationMode.VLLM_COMPILE,
                CUDAGraphMode.FULL_DECODE_ONLY,
                False,
                None,
                True,
                CUDAGraphMode.PIECEWISE,
            ),
            # Different compilation modes
            (False, 1, CompilationMode.NONE, CUDAGraphMode.FULL, False, None, False, CUDAGraphMode.FULL),
            (
                False,
                2,
                CompilationMode.NONE,
                CUDAGraphMode.FULL_DECODE_ONLY,
                False,
                None,
                False,
                CUDAGraphMode.FULL_DECODE_ONLY,
            ),
            (
                False,
                None,
                CompilationMode.NONE,
                CUDAGraphMode.FULL_AND_PIECEWISE,
                False,
                None,
                False,
                CUDAGraphMode.NONE,
            ),
            (
                False,
                1,
                CompilationMode.STOCK_TORCH_COMPILE,
                CUDAGraphMode.FULL_DECODE_ONLY,
                False,
                None,
                False,
                CUDAGraphMode.NONE,
            ),
            (
                False,
                2,
                CompilationMode.STOCK_TORCH_COMPILE,
                CUDAGraphMode.FULL_AND_PIECEWISE,
                False,
                None,
                False,
                CUDAGraphMode.NONE,
            ),
            (
                False,
                None,
                CompilationMode.STOCK_TORCH_COMPILE,
                CUDAGraphMode.FULL,
                False,
                None,
                False,
                CUDAGraphMode.NONE,
            ),
        ],
    )
    def test_cuda_graph_from_cli(
        self,
        enforce_eager,
        parallel_config_tensor_parallel_size,
        compilation_config_mode,
        compilation_config_cudagraph_mode,
        speculative_config_enforce_eager,
        speculative_method,
        is_encoder_decoder,
        expected_cudagraph_mode,
    ):
        # Mock list_filtered_repo_files and is_encoder_decoder
        with ExitStack() as stack:
            stack.enter_context(patch("vllm.transformers_utils.repo_utils.list_filtered_repo_files", return_value=[]))

            # Patch the is_encoder_decoder function and property to use the test case value
            # This ensures the scheduler_config uses the correct value from test parametrization
            stack.enter_context(patch("vllm.config.model.is_encoder_decoder", return_value=is_encoder_decoder))

            # For the cached_property, we need to create a mock property
            mock_is_encoder_decoder = property(lambda self: is_encoder_decoder)
            stack.enter_context(patch.object(ModelConfig, "is_encoder_decoder", mock_is_encoder_decoder))

            # Mock create_speculative_config to completely avoid model weight loading
            # This approach doesn't create a real SpeculativeConfig object
            def mock_create_speculative_config(*args, **kwargs):
                # args[0] is the self parameter when called as a method
                self_obj = args[0] if args else None
                if not self_obj or not hasattr(self_obj, "speculative_config") or self_obj.speculative_config is None:
                    return None

                # Create a simple mock object that satisfies the interface
                mock_spec_config = MagicMock()

                # Set all attributes from the speculative config
                for key, value in self_obj.speculative_config.items():
                    setattr(mock_spec_config, key, value)

                # Add required attributes that might be accessed
                mock_spec_config.method = self_obj.speculative_config.get("method")
                mock_spec_config.num_speculative_tokens = self_obj.speculative_config.get("num_speculative_tokens", 1)
                mock_spec_config.model = self_obj.speculative_config.get("model")
                mock_spec_config.draft_model_config = MagicMock()
                mock_spec_config.draft_parallel_config = MagicMock()
                mock_spec_config.target_model_config = MagicMock()
                mock_spec_config.target_parallel_config = MagicMock()

                return mock_spec_config

            stack.enter_context(
                patch.object(EngineArgs, "create_speculative_config", side_effect=mock_create_speculative_config)
            )

            parser = EngineArgs.add_cli_args(FlexibleArgumentParser())

            # Build a single list of CLI arguments to test combination of parameters
            cli_args = []

            # Add enforce eager if needed
            if enforce_eager:
                cli_args.append("--enforce-eager")

            # Add compilation config if specified
            if compilation_config_mode is not None or compilation_config_cudagraph_mode is not None:
                compilation_config = {}
                if compilation_config_mode is not None:
                    compilation_config["mode"] = compilation_config_mode.name
                if compilation_config_cudagraph_mode is not None:
                    compilation_config["cudagraph_mode"] = compilation_config_cudagraph_mode.name
                compilation_config_str = json.dumps(compilation_config)
                cli_args.extend(["--compilation-config", compilation_config_str])

            # Add tensor parallel size if specified
            if parallel_config_tensor_parallel_size is not None:
                cli_args.extend(["--tensor-parallel-size", str(parallel_config_tensor_parallel_size)])

            # Add speculative config if needed
            speculative_config = {"enforce_eager": speculative_config_enforce_eager, "num_speculative_tokens": 1}
            # Add method-specific configuration based on E2E test examples
            if speculative_method is not None:
                speculative_config["method"] = speculative_method

            speculative_config_str = json.dumps(speculative_config)
            cli_args.extend(["--speculative-config", speculative_config_str])

            # Parse all arguments at once and create engine config
            args = parser.parse_args(cli_args)
            vllm_config = EngineArgs.from_cli_args(args=args).create_engine_config()
            assert vllm_config.compilation_config.cudagraph_mode == expected_cudagraph_mode, (
                "V1 vllm_config.compilation_config.cudagraph_mode"
            )
