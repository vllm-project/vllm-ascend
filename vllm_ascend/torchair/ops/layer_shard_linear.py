from dataclasses import dataclass
from typing import Callable, Optional, Union

import torch
import torch.distributed as dist
from torch.nn.parameter import Parameter
from vllm.distributed.parallel_state import GroupCoordinator
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.utils import set_weight_attrs

from vllm_ascend.utils import dispose_tensor


@dataclass
class LayerMetadata:
    """Metadata for a layer.
    """
    layer: Optional[LinearBase]  # The layer object.
    post_method: Callable[[
        torch.nn.Module
    ], None]  # The `process_weights_after_loading` method from the quant method.
    weight: torch.Tensor  # The weight tensor.
    window_idx: int  # The index of the window.


@dataclass
class SharedWindowMetadata:
    """Metadata for a shared window.
    """
    weight: torch.Tensor  # The weight tensor to be shared by layers.
    data_layer_idx: int  # The index of the layer this window's weight is equal to.
    work: Optional[torch.distributed.Work]  # The asynchronous broadcast work.


@dataclass
class SeriesMetadata:
    """Metadata for a series.
    """
    group: GroupCoordinator
    start_layer: int
    end_layer: int
    num_layers: int
    prefetch_step: int
    dummy_weight: torch.Tensor  # Dummy weight to replace the loaded weight matrix. All the layers in the series share the same dummy weight tensor.
    layers: list[LayerMetadata]
    shared_windows: list[
        SharedWindowMetadata]  # Shared windows for prefetching. The window size is (`prefetch_step` + 1), as only the weights for the next (`prefetch_step` + 1) layers need to be stored.
    window_offset: int  # The index of the window for the next coming layer.

    def is_source(self, layer_idx) -> bool:
        return layer_idx % self.group.world_size == self.group.rank_in_group

    def post_process_after_loading(self):
        # This method only needs to be called once per series.
        if self.shared_windows:
            return
        for layer_idx in range(self.start_layer, self.end_layer):
            layer = self.layers[layer_idx - self.start_layer]
            is_source = self.is_source(layer_idx)
            # If the weight uses dummy weight, make a copy temporary such that the post method call won't affect other layers which also uses dummy weight.
            if not is_source:
                layer.weight.set_(torch.empty_like(self.dummy_weight))
            # Broadcast to get the true weight.
            dist.broadcast(layer.weight,
                           src=self.group.ranks[layer_idx %
                                                self.group.world_size],
                           group=self.group.device_group)
            assert layer.layer is not None
            # Call `process_weights_after_loading` from the quant method.
            layer.post_method(layer.layer)
            step = layer_idx - self.start_layer
            if step < self.prefetch_step:
                # Build the windows for the first `prefetch_step` layers. The weights can be used for the first `prefetch_step` layers in `forward()`, so also clone the weights.
                self.shared_windows.append(
                    SharedWindowMetadata(
                        weight=layer.weight.clone().detach(),
                        data_layer_idx=layer_idx,
                        work=None,
                    ))
                layer.window_idx = step
                # When the layer not intended to be stored in this device, link to the corresponding window's tensor.
                if not is_source:
                    layer.weight.set_(self.shared_windows[-1].weight)
            else:
                # Build one more window for prefetch. The weight is useless, so just keep the shape.
                if step == self.prefetch_step:
                    self.shared_windows.append(
                        SharedWindowMetadata(
                            weight=torch.empty_like(layer.weight),
                            data_layer_idx=-1,
                            work=None,
                        ))
                # When the layer not intended to be stored in this device, dispose the tensor.
                if not is_source:
                    dispose_tensor(layer.weight)

        dispose_tensor(self.dummy_weight)

    def get_shared_window(self, layer_idx: int):
        assert self.shared_windows
        return self.shared_windows[self.layers[layer_idx -
                                               self.start_layer].window_idx]

    def reach_layer(self, layer_idx: int):
        # The index of the layer to be prefetched.
        next_layer_idx = (layer_idx + self.prefetch_step
                          ) % self.num_layers + self.start_layer
        next_layer = self.layers[next_layer_idx - self.start_layer]
        # The index of the window to store the weight for the coming layer.
        next_layer.window_idx = self.window_offset
        window = self.shared_windows[next_layer.window_idx]
        # When the layer not intended to be stored in this device, link to the corresponding window's tensor.
        if not self.is_source(next_layer_idx):
            next_layer.weight.set_(window.weight)
        # Update `window_offset` by rolling one step.
        self.window_offset = (self.window_offset + 1) % (self.prefetch_step +
                                                         1)
        assert window.data_layer_idx != next_layer_idx
        window.data_layer_idx = next_layer_idx
        # Start asynchronous broadcast work.
        window.work = dist.broadcast(
            next_layer.weight,
            src=self.group.ranks[next_layer_idx % self.group.world_size],
            group=self.group.device_group,
            async_op=True)


_series_dict: dict[str, SeriesMetadata] = {}


def register_layer_to_series(
    name: str,
    group: GroupCoordinator,
    start_layer: int,
    end_layer: int,
    prefetch_step: int,
    layer_idx: int,
    layer: LinearBase,
) -> SeriesMetadata:
    global _series_dict
    if name not in _series_dict:
        num_layers = end_layer - start_layer
        assert num_layers > 0
        assert prefetch_step >= 0 and prefetch_step <= num_layers - 2
        _series_dict[name] = SeriesMetadata(
            group=group,
            start_layer=start_layer,
            end_layer=end_layer,
            num_layers=num_layers,
            prefetch_step=prefetch_step,
            dummy_weight=torch.empty_like(layer.weight),
            layers=[
                LayerMetadata(
                    layer=None,
                    post_method=lambda layer: None,
                    weight=torch.empty([]),
                    window_idx=-1,
                ) for _ in range(num_layers)
            ],
            shared_windows=[],
            window_offset=prefetch_step,
        )
    series = _series_dict[name]
    assert layer.quant_method is not None
    series.layers[layer_idx - start_layer] = LayerMetadata(
        layer=layer,
        post_method=layer.quant_method.process_weights_after_loading,
        weight=layer.weight,
        window_idx=-1,
    )
    # Discard the original `process_weights_after_loading` method such that it won't be called by others.
    layer.quant_method.process_weights_after_loading = lambda layer: None
    # When the layer not intended to be stored in this device, dispose the tensor.
    if not series.is_source(layer_idx):
        dispose_tensor(layer.weight)
    return series


@CustomOp.register("layer_shard_linear")
class LayerShardLinear(LinearBase):
    """Linear layer with sharding storage.

    Each device in the parallel group evenly stores a set of disjoint layers. All layers must have the same structure. A set of isomorphic layers is defined as a "series". Assuming there are n devices, the weight matrix of the i-th layer will be stored on the (i % n)-th device.

    After loading the model, you must call `post_process_after_loading_for_series()` from any layer of this series to complete the initialization.

    Each time a new layer is reached, you must call `reach_layer()` from that layer to prefetch the weights. The argument `prefetch_step` is a non-negative integer k that manages asynchronous weight prefetching. Each call to this layer's `reach_layer()` method will trigger an asynchronous prefetch for the weights of the k-th subsequent layer.

    Note: The layers are managed as a circular buffer. The index of the layer to prefetch is determined by the formula:
    - total_layers = end_layer - start_layer
    - prefetch_layer_idx = (layer_idx + prefetch_step) % total_layers + start_layer

    To hold the weights for the current layer and the k prefetched layers, a pool of (k + 1) shared tensor buffers will be created for this series.

    Arguments:
        input_size: first dimension of matrix.
        output_size: second dimension of matrix.
        bias: If true, add bias.
        skip_bias_add: This was added to enable performance optimization where
                       bias can be fused with other element-wise operations.
                       We skip adding bias but instead return it.
        params_dtype: Data type for the parameters.
        quant_config: Quantization configure.
        prefix: The name of the layer in the state dict, including all parents
                        (e.g. model.layers.0.self_attn.o_proj)
        return_bias: If true, return bias together with outputs in forward pass.
        series_name: This name identifies which series this layer belongs to.
        group: The group coordinator for handling asynchronous communications. It is recommended to create a new group coordinator for each new series.
        start_layer: The index of the first layer in the series (inclusive).
        end_layer: The index of the last layer in the series (exclusive). Thus, the series includes all layers with indices in the range [start_layer, end_layer).
        layer_idx: The index of the current layer.
        prefetch_step: An integer that manages asynchronous weight prefetching.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True,
        skip_bias_add: bool = False,
        params_dtype: Optional[torch.dtype] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        *,
        return_bias: bool = True,
        series_name: str,
        group: GroupCoordinator,
        start_layer: int,
        end_layer: int,
        layer_idx: int,
        prefetch_step: int = 0,
    ):
        self.input_size = input_size
        self.output_size = output_size
        self.output_partition_sizes = [output_size]
        super().__init__(input_size,
                         output_size,
                         skip_bias_add,
                         params_dtype,
                         quant_config,
                         prefix,
                         return_bias=return_bias)
        assert self.quant_method is not None
        self.quant_method.create_weights(
            layer=self,
            input_size_per_partition=self.input_size,
            output_partition_sizes=[self.output_size],
            input_size=self.input_size,
            output_size=self.output_size,
            params_dtype=self.params_dtype,
            weight_loader=self.weight_loader)
        if bias:
            self.bias = Parameter(
                torch.empty(self.output_size, dtype=params_dtype))
            set_weight_attrs(self.bias, {
                "output_dim": 0,
                "weight_loader": self.weight_loader,
            })
        else:
            self.register_parameter("bias", None)

        self.layer_idx = layer_idx
        self.series = register_layer_to_series(
            name=series_name,
            group=group,
            start_layer=start_layer,
            end_layer=end_layer,
            prefetch_step=prefetch_step,
            layer_idx=layer_idx,
            layer=self,
        )

    def weight_loader(self, param: Parameter, loaded_weight: torch.Tensor):
        # Skip loading matrix weight when not intended to be stored on this device.
        if param is self.weight and not self.series.is_source(self.layer_idx):
            return
        assert not getattr(param, "is_gguf_weight", False)
        assert not getattr(param, "is_gguf_weight_type", False)
        # If the weight on disk does not have a shape, give it one
        # (such scales for AutoFp8).
        if len(loaded_weight.shape) == 0:
            loaded_weight = loaded_weight.reshape(1)
        assert param.size() == loaded_weight.size(), (
            f"Tried to load weights of size {loaded_weight.size()}"
            f"to a parameter of size {param.size()}")
        param.data.copy_(loaded_weight)

    def post_process_after_loading_for_series(self):
        self.series.post_process_after_loading()

    def reach_layer(self):
        self.series.reach_layer(self.layer_idx)

    def forward(
        self,
        input,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, Optional[Parameter]]]:
        # Find the asynchronous broadcast work and wait for it.
        window = self.series.get_shared_window(self.layer_idx)
        # Make sure the data in the corresponding shared window is for the current layer.
        assert window.data_layer_idx == self.layer_idx
        if window.work is not None:
            window.work.wait()
            window.work = None
        # Matrix multiply.
        bias_ = None if self.skip_bias_add else self.bias
        output = self.quant_method.apply(self, input, bias=bias_)
        output_bias = self.bias if self.skip_bias_add else None
        if not self.return_bias:
            return output
        return output, output_bias

    def extra_repr(self) -> str:
        s = f"input_features={self.input_size}"
        s += f", output_features={self.output_size}"
        s += f", bias={self.bias is not None}"
        return s
