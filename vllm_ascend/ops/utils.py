from vllm_ascend.ops.shared_weight_layer import (
    is_hidden_layer, post_process_after_loading_for_shared_weight_series,
    reach_layer_for_shared_weight_series,
    register_layer_to_shared_weight_series)
from vllm.model_executor.models.utils import extract_layer_index
from vllm_ascend.distributed.parallel_state import get_shared_weight_group
from typing import Dict, Any, Optional


class Flashcomm2OShardManager:
    """Manages sharded layers for the FlashComm2 O-Shard feature.

    This class is implemented to centralize all logic related to Flashcomm2OShard layers.
    Its main responsibilities are:
    1.  Registering Attention `o_proj` layers that require O-Sharding.
    2.  Storing and managing these layers in a dictionary mapping layer indices
        to layer objects (`layer_index -> layer`).
    3.  Providing a high-level API for external callers to use at key stages
        like model initialization, computation, and weight loading.

    Attributes:
        _shard_layers: A dictionary to store the registered sharded layers,
            mapping a layer index (int) to its corresponding layer object.
    """

    def __init__(self):
        self._shard_layers: Dict[int, Any] = {}

    def register_layer(self,
                       layer: Any,
                       vllm_config: Any,
                       prefetch_step: int = 1):
        """Registers a layer for O-Sharding.

        This method first checks if the O-Shard feature is enabled and if the
        provided layer qualifies as a target (e.g., a hidden layer). If so,
        it performs two actions:
        1. Caches the layer internally in the `_shard_layers` dictionary.
        2. Calls the underlying `register_layer_to_shared_weight_series`
           function to register it for communication.

        Args:
            layer: The layer object to be registered.
            vllm_config: The vLLM model configuration object, used to determine
                if the layer is a target for sharding.
            prefetch_step: The prefetch step to be used when registering the
                layer to the shared weight series.
        """
        # Check if the layer is a target for sharding.
        if is_hidden_layer(vllm_config, layer):
            layer_idx = extract_layer_index(layer.prefix)
            self._shard_layers[layer_idx] = layer

            register_layer_to_shared_weight_series(
                series_name="o_proj",
                group=get_shared_weight_group(),
                layer=layer,
                prefetch_step=prefetch_step)

    def get_layer(self, layer_idx: int) -> Optional[Any]:
        """Safely retrieves a registered layer by its index.

        Args:
            layer_idx: The index of the layer to retrieve.

        Returns:
            The layer object if found, otherwise None.
        """
        return self._shard_layers.get(layer_idx)

    def trigger_broadcast_for_layer(self, layer_prefix: str, vllm_config: Any):
        """Triggers a broadcast for a specific layer during model computation.

        This method is intended to be called within a layer's forward pass.
        It extracts the layer index from the prefix, retrieves the corresponding
        registered layer object, and then triggers the broadcast operation
        if all conditions are met.

        Args:
            layer_prefix: The name prefix of the current layer being computed.
            vllm_config: The vLLM model configuration object.
        """
        layer_idx = extract_layer_index(layer_prefix)
        target_layer = self.get_layer(layer_idx)

        # Ensure the layer exists and meets the sharding criteria.
        if target_layer and is_hidden_layer(vllm_config, target_layer):
            reach_layer_for_shared_weight_series(target_layer)

    def post_process_after_loading(self):
        """Performs post-processing on all registered layers after weight loading.

        This should be called once after the model weights have been fully loaded.
        """
        # Iterate through all registered layers to preform post_process_after_loading
        for layer_idx in sorted(self._shard_layers.keys()):
            layer = self._shard_layers[layer_idx]
            post_process_after_loading_for_shared_weight_series(layer)


flashcomm2_oshard_manager = Flashcomm2OShardManager()
