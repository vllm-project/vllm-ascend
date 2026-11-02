"""Execution defaults scoped to a selected SplitQuant  profile."""
import os


def apply_execution_profile(vllm_config):
    if os.getenv('SPLITQUANT_DENSE') != '1':
        return
    mode = os.getenv('SPLITQUANT_EXECUTION_MODE')
    if mode is None:
        return
    if mode not in ('graph', 'eager'):
        raise ValueError('SPLITQUANT_EXECUTION_MODE must be graph or eager')
    model = vllm_config.model_config
    if model is None:
        return
    from vllm.config import CompilationLevel, CUDAGraphMode
    compilation = vllm_config.compilation_config
    # An explicit eager request takes precedence over the Graph default.
    if mode == 'eager' or model.enforce_eager:
        model.enforce_eager = True
        compilation.level = CompilationLevel.NO_COMPILATION
        compilation.cudagraph_mode = CUDAGraphMode.NONE
        return
    compilation.level = CompilationLevel.PIECEWISE
    compilation.cudagraph_mode = CUDAGraphMode.FULL_DECODE_ONLY
    compilation.use_inductor = False
    compilation.cudagraph_capture_sizes = [1]
    compilation.max_capture_size = 1
