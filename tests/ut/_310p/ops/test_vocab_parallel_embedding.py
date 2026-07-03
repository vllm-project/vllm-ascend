from types import SimpleNamespace

import torch

from vllm_ascend._310p.ops import vocab_parallel_embedding as embedding_ops


def test_310p_embedding_general_path_uses_nz_weight(monkeypatch):
    method = embedding_ops.AscendUnquantizedEmbeddingMethod310()
    layer = SimpleNamespace(
        weight=torch.zeros((4, 3)),
        weight_nz=torch.ones((4, 3)),
    )
    x = torch.ones((2, 3))

    def fake_linear(input_, weight, bias=None):
        assert input_ is x
        assert weight is layer.weight_nz
        assert bias is None
        return torch.empty((2, 4))

    monkeypatch.setattr(embedding_ops.F, "linear", fake_linear)

    output = method.apply(layer, x)

    assert output.shape == (2, 4)


def test_310p_embedding_deepseek_ocr2_path_uses_nd_weight(monkeypatch):
    monkeypatch.setattr(embedding_ops, "is_deepseek_ocr2_310p_model", lambda: True)

    method = embedding_ops.AscendUnquantizedEmbeddingMethod310()
    layer = SimpleNamespace(
        weight=torch.arange(12, dtype=torch.float32).reshape(4, 3),
        weight_nz=torch.empty((4, 3)),
    )
    x = torch.ones((2, 3))

    method.process_weights_after_loading(layer)
    output = method.apply(layer, x)

    assert torch.equal(output, torch.matmul(x, layer.weight.t()))


def test_310p_embedding_deepseek_ocr2_path_preserves_input_dtype(monkeypatch):
    monkeypatch.setattr(embedding_ops, "is_deepseek_ocr2_310p_model", lambda: True)

    method = embedding_ops.AscendUnquantizedEmbeddingMethod310()
    layer = SimpleNamespace(
        weight=torch.arange(12, dtype=torch.float16).reshape(4, 3),
    )
    x = torch.ones((2, 3), dtype=torch.bfloat16)

    method.process_weights_after_loading(layer)
    output = method.apply(layer, x)

    assert output.dtype == x.dtype
