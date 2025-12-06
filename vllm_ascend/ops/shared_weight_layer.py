from dataclasses import dataclass
from typing import Callable, Optional

import torch
import torch.distributed as dist
from vllm.distributed.parallel_state import GroupCoordinator
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.models.utils import extract_layer_index


def dispose_tensor(x: torch.Tensor):
    x.set_(torch.empty([], device=x.device, dtype=x.dtype))


@dataclass
class LayerMetadata:
    """Metadata for a layer.
    """
    layer_idx: int  # The index of the layer.
    layer: LinearBase  # The layer object.
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
    """Metadata for a weight shared series.
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

        self.layers.sort(key=lambda x: x.layer_idx)
        self.num_layers = len(self.layers)
        assert self.num_layers > 0, "No layers in the series"
        assert self.prefetch_step >= 0 and self.prefetch_step <= max(
            0, self.num_layers -
            2), "prefetch_step must be in [0, num_layers - 2]"
        self.start_layer = self.layers[0].layer_idx
        self.end_layer = self.layers[-1].layer_idx + 1

        for layer_idx in range(self.start_layer, self.end_layer):
            layer = self.layers[layer_idx - self.start_layer]
            assert layer.layer_idx == layer_idx, "layer_idx must be consecutive"
            is_source = self.is_source(layer_idx)
            # If the weight uses dummy weight, make a copy temporary such that the post method call won't affect other layers which also uses dummy weight.
            if not is_source:
                layer.weight.set_(torch.empty_like(self.dummy_weight))
            # Broadcast to get the true weight.
            dist.broadcast(layer.weight,
                           src=self.group.ranks[layer_idx %
                                                self.group.world_size],
                           group=self.group.device_group)
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
        # Dispose the dummy tensor since it's no longer needed.
        dispose_tensor(self.dummy_weight)

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

    def wait_weight(self, layer_idx: int):
        # Find the asynchronous broadcast work and wait for it.
        assert self.shared_windows
        window = self.shared_windows[self.layers[layer_idx -
                                                 self.start_layer].window_idx]
        # Make sure the data in the corresponding shared window is for the current layer.
        assert window.data_layer_idx == layer_idx
        if window.work is not None:
            window.work.wait()
            window.work = None


@dataclass
class LayerExternalMetadata:
    """External metadata for a layer.
    """
    series: SeriesMetadata
    layer_idx: int


_series_dict: dict[str, SeriesMetadata] = {}

_layer_external_dict: dict[int, LayerExternalMetadata] = {}


def _create_forward_wrapper(forward: Callable, series: SeriesMetadata,
                            layer_idx: int) -> Callable:

    def wrapped_forward(*args, **kwargs):
        # Wait for the weight.
        series.wait_weight(layer_idx)
        return forward(*args, **kwargs)

    return wrapped_forward


"""
Register linear layers into a shared storage series.

In a parallel group, each device stores a distinct, non-overlapping subset of layers from the series. All layers in a series must have the same structure (are isomorphic). The weight matrix for the i-th layer is stored on device (i % n), where n is the number of devices.

After loading the model, you must call `post_process_after_loading_for_shared_weight_series(layer)` on any layer of this series to complete the initialization.

During execution, each time a new layer is reached, you must call `reach_layer_for_shared_weight_series(layer)` for that layer to prefetch the weights. The argument `prefetch_step` is a non-negative integer k that manages asynchronous weight prefetching. Each call to `reach_layer_for_shared_weight_series(current_layer)` method will trigger an asynchronous prefetch for the weights of the k-th subsequent layer after `current_layer` within the series.

Note: The layers are managed as a circular buffer. The index of the layer to prefetch is determined by the formula:
- start_layer is the index of the first layer in the series (inclusive).
- end_layer is the index of the last layer in the series (exclusive). Thus, the series includes all layers with indices in the range [start_layer, end_layer).
- total_layers = end_layer - start_layer
- prefetch_layer_idx = (layer_idx + prefetch_step) % total_layers + start_layer

To hold the weights for the current layer and the k prefetched layers, a pool of (k + 1) shared tensor buffers will be created for this series.

Arguments:
    series_name: This name identifies which series this layer belongs to.
    group: The group coordinator for handling asynchronous communications. It is recommended to create a new group coordinator for each new series.
    layer: The linear layer object to register.
    prefetch_step: An integer that manages asynchronous weight prefetching. Setting it to 0 or 1 can cover most cases.
"""


def register_layer_to_shared_weight_series(
    series_name: str,
    group: GroupCoordinator,
    layer: LinearBase,
    prefetch_step: int = 1,
):
    global _series_dict
    if series_name not in _series_dict:
        _series_dict[series_name] = SeriesMetadata(
            group=group,
            start_layer=0,
            end_layer=0,
            num_layers=0,
            prefetch_step=prefetch_step,
            dummy_weight=torch.empty_like(layer.weight),
            layers=[],
            shared_windows=[],
            window_offset=prefetch_step,
        )
    series = _series_dict[series_name]
    assert layer.quant_method is not None
    layer_idx = extract_layer_index(layer.prefix)
    series.layers.append(
        LayerMetadata(
            layer_idx=layer_idx,
            layer=layer,
            post_method=layer.quant_method.process_weights_after_loading,
            weight=layer.weight,
            window_idx=-1,
        ))
    # Discard the original `process_weights_after_loading` method such that it won't be called by others.
    layer.quant_method.process_weights_after_loading = lambda layer: None
    # When the layer not intended to be stored in this device, dispose the tensor and skip weight loading.
    if not series.is_source(layer_idx):
        dispose_tensor(layer.weight)
        layer.weight.weight_loader = lambda *args, **kwargs: None
    layer.forward = _create_forward_wrapper(layer.forward, series, layer_idx)
    global _layer_external_dict
    _layer_external_dict[id(layer)] = LayerExternalMetadata(
        series=series,
        layer_idx=layer_idx,
    )


def post_process_after_loading_for_shared_weight_series(layer: LinearBase):
    ext = _layer_external_dict[id(layer)]
    ext.series.post_process_after_loading()


def reach_layer_for_shared_weight_series(layer: LinearBase):
    ext = _layer_external_dict[id(layer)]
    ext.series.reach_layer(ext.layer_idx)


def is_hidden_layer(vllm_config, layer: LinearBase) -> bool:
    num_hidden_layers = vllm_config.model_config.hf_config.num_hidden_layers
    layer_idx = extract_layer_index(layer.prefix)
    return layer_idx < num_hidden_layers