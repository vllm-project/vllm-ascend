import inspect
import math

import torch

from vllm_ascend.ascend_config import _is_deepseek_v4_dsa_model
from vllm_ascend.turboquant import tq_latent_store


def _sylvester_hadamard(order: int) -> torch.Tensor:
    matrix = torch.ones(1, 1, dtype=torch.float64)
    while matrix.shape[0] < order:
        matrix = torch.cat(
            (
                torch.cat((matrix, matrix), dim=1),
                torch.cat((matrix, -matrix), dim=1),
            ),
            dim=0,
        )
    return matrix / math.sqrt(order)


def test_shared_kv_hadamard_attention_is_basis_invariant():
    torch.manual_seed(0)
    head_dim = 512
    transform = _sylvester_hadamard(head_dim)
    signs = torch.where(torch.arange(head_dim) % 3 == 0, -1.0, 1.0).to(torch.float64)
    transform = signs.unsqueeze(1) * transform

    query = torch.randn(3, head_dim, dtype=torch.float64)
    shared_kv = torch.randn(7, head_dim, dtype=torch.float64)
    scale = 1.0 / math.sqrt(head_dim)

    reference = torch.softmax(query @ shared_kv.T * scale, dim=-1) @ shared_kv
    transformed_query = query @ transform
    transformed_kv = shared_kv @ transform
    transformed_output = torch.softmax(transformed_query @ transformed_kv.T * scale, dim=-1) @ transformed_kv
    restored_output = transformed_output @ transform.T

    torch.testing.assert_close(restored_output, reference, rtol=1e-10, atol=1e-10)


def test_turboquant_legacy_and_compact_slot_layouts():
    head_dim = 512
    assert tq_latent_store.packed_bytes(head_dim) == 256
    assert tq_latent_store.base_slot_size(head_dim) == 320
    assert tq_latent_store.compact_slot_size(head_dim) == 258
    output_mode = inspect.signature(tq_latent_store.compress_kernel).parameters["output_mode"]
    assert output_mode.default == tq_latent_store.COMPRESS_OUTPUT_LEGACY
    assert (1024 - tq_latent_store.compact_slot_size(head_dim)) / 1024 == 0.748046875


def test_dsa_turboquant_is_scoped_to_deepseek_v4():
    deepseek_v4 = type(
        "ModelConfig",
        (),
        {"hf_text_config": type("HFConfig", (), {"model_type": "deepseek_v4"})()},
    )()
    other_dsa = type(
        "ModelConfig",
        (),
        {
            "hf_text_config": type(
                "HFConfig", (), {"model_type": "other", "compress_ratios": [4]}
            )()
        },
    )()

    assert _is_deepseek_v4_dsa_model(deepseek_v4)
    assert not _is_deepseek_v4_dsa_model(other_dsa)
    assert not _is_deepseek_v4_dsa_model(None)


def test_deepseek_shared_kv_centroid_order_matches_signed_nibbles():
    tq_latent_store._build(torch.device("cpu"), 512)
    rotated_centroids = torch.cat(
        (tq_latent_store._CENT[8:], tq_latent_store._CENT[:8])
    )
    kernel_centroids = torch.tensor(
        [
            0.00547294,
            0.01680406,
            0.02857605,
            0.04108622,
            0.05492980,
            0.07101817,
            0.09115373,
            0.12037795,
            -0.12091285,
            -0.09111122,
            -0.07112455,
            -0.05513602,
            -0.04132067,
            -0.02874970,
            -0.01700489,
            -0.00568677,
        ],
        dtype=torch.float32,
    )

    torch.testing.assert_close(rotated_centroids.cpu(), kernel_centroids)
