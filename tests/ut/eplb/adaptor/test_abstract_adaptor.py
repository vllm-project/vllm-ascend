import pytest

from vllm_ascend.eplb.adaptor.abstract_adaptor import EplbAdaptor


class DummyAdaptor(EplbAdaptor):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.args = kwargs

    def get_rank_expert_workload(self):
        return "workload"

    def get_init_expert_map(self, num_moe_layers):
        return {"layers": num_moe_layers}

    def do_update_expert_map(self, layer_id, updated_expert_map):
        return {"layer_id": layer_id, "map": updated_expert_map}

    def do_update_expert_weight(self, layer_id, local_expert_to_replace,
                                buffer_tensor_id):
        return {
            "layer_id": layer_id,
            "replace": local_expert_to_replace,
            "buffer": buffer_tensor_id,
        }


def test_base_class_methods_raise():
    adaptor = EplbAdaptor()
    with pytest.raises(NotImplementedError):
        adaptor.get_rank_expert_workload()
    with pytest.raises(NotImplementedError):
        adaptor.get_init_expert_map(1)
    with pytest.raises(NotImplementedError):
        adaptor.do_update_expert_map(1, {})
    with pytest.raises(NotImplementedError):
        adaptor.do_update_expert_weight(1, "x", "y")


def test_dummy_adaptor_init_and_args():
    adaptor = DummyAdaptor(test_arg=123)
    assert adaptor.args["test_arg"] == 123


def test_get_rank_expert_workload():
    adaptor = DummyAdaptor()
    result = adaptor.get_rank_expert_workload()
    assert result == "workload"


def test_get_init_expert_map():
    adaptor = DummyAdaptor()
    result = adaptor.get_init_expert_map(5)
    assert isinstance(result, dict)
    assert result["layers"] == 5


def test_do_update_expert_map():
    adaptor = DummyAdaptor()
    updated = {"expert": 1}
    result = adaptor.do_update_expert_map(2, updated)
    assert result["layer_id"] == 2
    assert result["map"] == updated


def test_do_update_expert_weight():
    adaptor = DummyAdaptor()
    result = adaptor.do_update_expert_weight(1, "expertA", "bufferX")
    assert result["layer_id"] == 1
    assert result["replace"] == "expertA"
    assert result["buffer"] == "bufferX"
