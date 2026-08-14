from types import SimpleNamespace
from unittest.mock import patch

import torch

from vllm_ascend.ops.dots3_note_audio_attention import (
    AscendDots3NoteAudioAttentionBackend,
)


def test_dots3_note_audio_attention_uses_unpad_attention():
    backend = SimpleNamespace(
        num_heads=2,
        num_kv_heads=2,
        head_size=4,
        scale=0.5,
    )
    query = torch.randn(1, 3, 2, 4)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    expected = query.reshape(-1, 2, 4) + 1

    with patch(
        "vllm_ascend.ops.dots3_note_audio_attention.DeviceOperator.npu_flash_attention",
        return_value=expected,
    ) as mock_attention:
        output = AscendDots3NoteAudioAttentionBackend.forward_oot(
            backend,
            query,
            key,
            value,
            torch.tensor([0, 3], dtype=torch.int32),
        )

    torch.testing.assert_close(output, expected.reshape(1, 3, 2, 4))
    kwargs = mock_attention.call_args.kwargs
    assert kwargs["query"].shape == (3, 2, 4)
    assert kwargs["key"].shape == (3, 2, 4)
    assert kwargs["value"].shape == (3, 2, 4)
    assert kwargs["seq_lens_cpu"].dtype == torch.int32
    assert kwargs["seq_lens_cpu"].tolist() == [3]
    assert kwargs["head_num"] == 2
    assert kwargs["num_kv_heads"] == 2
    assert kwargs["scale_value"] == 0.5
