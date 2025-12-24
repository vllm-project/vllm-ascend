from collections.abc import Callable
from typing import Any

import torch
import torch.fx as fx
import vllm.compilation.backends
import vllm.compilation.piecewise_backend
from torch._dispatch.python import enable_python_dispatcher
from vllm.compilation.backends import VllmBackend
from vllm.compilation.counter import compilation_counter
from vllm.compilation.piecewise_backend import RangeEntry
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.config.utils import Range
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.import_utils import resolve_obj_by_qualname

logger = init_logger(__name__)


class AscendPiecewiseCompileInterpreter(torch.fx.Interpreter):
    """Code adapted from `torch.fx.passes.shape_prop.ShapeProp`.
    It runs the given graph with fake inputs, and compile some
    submodules specified by `compile_submod_names` with the given
    compilation configs.

    NOTE: the order in `compile_submod_names` matters, because
    it will be used to determine the order of the compiled piecewise
    graphs. The first graph will handle logging, and the last graph
    has some special cudagraph output handling.
    """

    def __init__(
        self,
        module: torch.fx.GraphModule,
        compile_submod_names: list[str],
        vllm_config: VllmConfig,
        vllm_backend: "VllmBackend",
    ):
        super().__init__(module)
        from torch._guards import detect_fake_mode

        self.fake_mode = detect_fake_mode()
        self.compile_submod_names = compile_submod_names
        self.compilation_config = vllm_config.compilation_config
        self.vllm_config = vllm_config
        self.vllm_backend = vllm_backend
        # When True, it annoyingly dumps the torch.fx.Graph on errors.
        self.extra_traceback = False

    def run(self, *args):
        # maybe instead just assert inputs are fake?
        fake_args = [
            self.fake_mode.from_tensor(t) if isinstance(t, torch.Tensor) else t
            for t in args
        ]
        with self.fake_mode, enable_python_dispatcher():
            return super().run(*fake_args)

    def call_module(
        self,
        target: torch.fx.node.Target,
        args: tuple[torch.fx.node.Argument, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        assert isinstance(target, str)

        output = super().call_module(target, args, kwargs)

        if target in self.compile_submod_names:
            index = self.compile_submod_names.index(target)
            submod = self.fetch_attr(target)

            sym_shape_indices = [
                i for i, x in enumerate(args) if isinstance(x, torch.SymInt)
            ]
            max_num_batched_tokens = self.vllm_config.scheduler_config.max_num_batched_tokens
            r1 = Range(start=1, end=max_num_batched_tokens)
            compiled_graph_for_dynamic_shape = (
                self.vllm_backend.compiler_manager.compile(
                    submod,
                    args,
                    self.vllm_backend.inductor_config,
                    self.compilation_config,
                    graph_index=index,
                    num_graphs=len(self.compile_submod_names),
                    compile_range=r1,
                ))

            # Lazy import here to avoid circular import
            from vllm.compilation.piecewise_backend import PiecewiseBackend

            piecewise_backend = PiecewiseBackend(
                submod,
                self.vllm_config,
                index,
                len(self.compile_submod_names),
                sym_shape_indices,
                compiled_graph_for_dynamic_shape,
                self.vllm_backend,
            )

            if (self.compilation_config.cudagraph_mode.
                    has_piecewise_cudagraphs() and
                    not self.compilation_config.use_inductor_graph_partition):
                # We're using Dynamo-based piecewise splitting, so we wrap
                # the whole subgraph with a static graph wrapper.
                from vllm.compilation.cuda_graph import CUDAGraphOptions

                # resolve the static graph wrapper class (e.g. CUDAGraphWrapper
                # class) as platform dependent.
                static_graph_wrapper_class = resolve_obj_by_qualname(
                    current_platform.get_static_graph_wrapper_cls())

                # Always assign PIECEWISE runtime mode to the
                # CUDAGraphWrapper for piecewise_backend, to distinguish
                # it from the FULL cudagraph runtime mode, no matter it
                # is wrapped on a full or piecewise fx graph.
                self.module.__dict__[target] = static_graph_wrapper_class(
                    runnable=piecewise_backend,
                    vllm_config=self.vllm_config,
                    runtime_mode=CUDAGraphMode.PIECEWISE,
                    cudagraph_options=CUDAGraphOptions(
                        debug_log_enable=piecewise_backend.is_first_graph,
                        gc_disable=not piecewise_backend.is_first_graph,
                        weak_ref_output=piecewise_backend.is_last_graph,
                    ),
                )
            else:
                self.module.__dict__[target] = piecewise_backend

            compilation_counter.num_piecewise_capturable_graphs_seen += 1

        return output


class AscendPiecewiseBackend:

    def __init__(
        self,
        graph: fx.GraphModule,
        vllm_config: VllmConfig,
        piecewise_compile_index: int,
        total_piecewise_compiles: int,
        sym_shape_indices: list[int],
        compiled_graph_for_general_shape: Callable,
        vllm_backend: VllmBackend,
    ):
        """
        The backend for piecewise compilation.
        It mainly handles the compilation of static shapes and
        dispatching based on runtime shape.

        We will compile `self.graph` once for the general shape,
        and then compile for different shapes specified in
        `compilation_config.compile_sizes`.
        """
        self.compiled_graph_for_general_shape = compiled_graph_for_general_shape
        self.graph = graph
        self.vllm_config = vllm_config
        self.compilation_config = vllm_config.compilation_config
        self.piecewise_compile_index = piecewise_compile_index
        self.total_piecewise_compiles = total_piecewise_compiles
        self.vllm_backend = vllm_backend

        self.is_first_graph = piecewise_compile_index == 0
        self.is_last_graph = piecewise_compile_index == total_piecewise_compiles - 1

        self.is_full_graph = total_piecewise_compiles == 1
        self.is_encoder_compilation = vllm_backend.is_encoder

        self.compile_ranges = self.compilation_config.get_compile_ranges()
        if self.is_encoder_compilation:
            # For encoder compilation we use the max int32 value
            # to set the upper bound of the compile ranges
            max_int32 = 2**31 - 1
            last_compile_range = self.compile_ranges[-1]
            assert (last_compile_range.end ==
                    vllm_config.scheduler_config.max_num_batched_tokens)
            self.compile_ranges[-1] = Range(start=last_compile_range.start,
                                            end=max_int32)

        log_string = f"PiecewiseBackend: compile_ranges: {self.compile_ranges}"
        logger.debug_once(log_string)

        self.compile_sizes = self.compilation_config.compile_sizes
        log_string = f"PiecewiseBackend: compile_sizes: {self.compile_sizes}"
        logger.debug_once(log_string)

        self.sym_shape_indices = sym_shape_indices

        # the entries for ranges that we need to either
        self.range_entries: dict[Range, RangeEntry] = {}

        # to_be_compiled_ranges tracks the remaining ranges to compile,
        # and updates during the compilation process, so we need to copy it
        self.to_be_compiled_ranges: set[Range] = set(self.compile_ranges)

        # We only keep compilation management inside this class directly.
        for size in self.compile_sizes:
            range = Range(start=size, end=size)
            if range not in self.compile_ranges:
                self.range_entries[range] = RangeEntry(compile_range=range, )
                self.to_be_compiled_ranges.add(range)

        for range in self.compile_ranges:
            self.range_entries[range] = RangeEntry(compile_range=range, )

    def _find_range_for_shape(self, runtime_shape: int) -> Range | None:
        # First we try to find the range entry for the concrete compile size
        # If not found, we search for the range entry
        # that contains the runtime shape.
        if runtime_shape in self.compile_sizes:
            return self.range_entries[Range(start=runtime_shape,
                                            end=runtime_shape)]
        else:
            for range in self.compile_ranges:
                if runtime_shape in range:
                    return self.range_entries[range]
        return None

    def __call__(self, *args) -> Any:
        runtime_shape = args[self.sym_shape_indices[0]]
        range_entry = self._find_range_for_shape(runtime_shape)

        assert range_entry is not None, (
            f"Shape out of considered range: {runtime_shape} "
            "[1, max_num_batched_tokens]")

        return self.compiled_graph_for_general_shape(*args)


vllm.compilation.backends.PiecewiseCompileInterpreter = AscendPiecewiseCompileInterpreter
vllm.compilation.piecewise_backend.PiecewiseBackend.__init__ = AscendPiecewiseBackend.__init__
vllm.compilation.piecewise_backend.PiecewiseBackend.__call__ = AscendPiecewiseBackend.__call__
