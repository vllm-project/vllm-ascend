# test_rejection_sample.py

from unittest.mock import patch

import pytest
import torch

# 假设这些函数定义在 spec_decode.py 中
from vllm_ascend.sample.rejection_sampler import (
    expand_batch_to_tokens, expand_pytorch, rejection_greedy_sample_pytorch,
    rejection_random_sample_pytorch, rejection_sample,
    sample_recovered_tokens_pytorch)

# 全局常量
PLACEHOLDER_TOKEN_ID = -1
GREEDY_TEMPERATURE = 0.0
MAX_SPEC_LEN = 8  # 用于 expand_batch_to_tokens 的 MAX_NUM_TOKENS


# 模拟 SamplingMetadata
class MockSamplingMetadata:

    def __init__(self,
                 all_greedy=False,
                 all_random=False,
                 temperature=None,
                 generators=None):
        self.all_greedy = all_greedy
        self.all_random = all_random
        self.temperature = temperature or (GREEDY_TEMPERATURE
                                           if all_greedy else 1.0)
        self.generators = generators or {}


def test_rejection_greedy_sample_pytorch():
    """测试贪婪拒绝采样：草稿不匹配时停止，否则追加 bonus token"""
    batch_size = 2
    max_spec_len = 3
    output_token_ids = torch.full((batch_size, max_spec_len + 1),
                                  PLACEHOLDER_TOKEN_ID)

    cu_num_draft_tokens = torch.tensor([2, 4])  # 请求0: 2个草稿；请求1: 2个草稿
    draft_token_ids = torch.tensor([10, 11, 20, 21])
    target_argmax = torch.tensor([10, 99, 20, 22])  # 第二个位置不匹配
    bonus_token_ids = torch.tensor([[100], [200]])

    is_greedy = torch.tensor([True, True])

    rejection_greedy_sample_pytorch(
        output_token_ids,
        cu_num_draft_tokens,
        draft_token_ids,
        target_argmax,
        bonus_token_ids,
        is_greedy,
        max_spec_len,
    )

    # 请求0: [10, 99] + bonus 100 → [10, 99, 100, _]
    # 请求1: [20] 匹配，[21]!=22 → 拒绝，不再继续，bonus 不加
    assert output_token_ids[0, 0].item() == 10
    assert output_token_ids[0, 1].item() == 99
    assert output_token_ids[0, 2].item() == 100  # bonus
    assert output_token_ids[1, 0].item() == 20
    assert output_token_ids[1, 1].item() == PLACEHOLDER_TOKEN_ID  # 被拒绝
    assert output_token_ids[1, 2].item() == PLACEHOLDER_TOKEN_ID  # no bonus


def test_rejection_random_sample_pytorch():
    """测试随机拒绝采样：基于 uniform prob 决定是否接受"""
    batch_size = 2
    max_spec_len = 3
    output_token_ids = torch.full((batch_size, max_spec_len + 1),
                                  PLACEHOLDER_TOKEN_ID)

    cu_num_draft_tokens = torch.tensor([2, 1])
    draft_token_ids = torch.tensor([10, 11, 20])
    draft_probs = torch.tensor([
        [0.0, 0.6, 0.0, 0.4],  # vocab_size=4
        [0.1, 0.2, 0.3, 0.4],
        [0.5, 0.5, 0.0, 0.0],
    ])
    target_probs = torch.tensor([
        [0.0, 0.8, 0.0, 0.2],
        [0.2, 0.1, 0.3, 0.4],
        [0.9, 0.1, 0.0, 0.0],
    ])
    bonus_token_ids = torch.tensor([[100], [200]])
    recovered_token_ids = torch.tensor([1, 2, 3])  # 替代 token
    uniform_probs = torch.tensor([0.7, 0.6, 0.5])
    is_greedy = torch.tensor([False, False])
    vocab_size = 4

    rejection_random_sample_pytorch(
        output_token_ids,
        cu_num_draft_tokens,
        draft_token_ids,
        draft_probs,
        target_probs,
        bonus_token_ids,
        recovered_token_ids,
        uniform_probs,
        is_greedy,
        max_spec_len,
        vocab_size,
        IS_NGRAM=False,
    )

    # 请求0:
    #   pos0: p=0.6, q=0.8 → ratio=1.33 > 0.7 → 接受 → 10
    #   pos1: p=0.4, q=0.4 → ratio=1.0 > 0.6 → 接受 → 11
    #   bonus: 100
    assert output_token_ids[0, 0].item() == 10
    assert output_token_ids[0, 1].item() == 11
    assert output_token_ids[0, 2].item() == 100

    # 请求1:
    #   pos0: p=0.5, q=0.9 → ratio=1.8 > 0.5 → 接受 → 20
    #   bonus: 200
    assert output_token_ids[1, 0].item() == 20
    assert output_token_ids[1, 1].item() == 200


def test_expand_pytorch():
    """测试 expand_pytorch 功能"""
    input_ptr = torch.tensor([10, 20, 30])
    cu_num_tokens_ptr = torch.tensor([2, 5, 7])  # 累计
    output_ptr = torch.empty(7, dtype=torch.int32)

    expand_pytorch(
        output_ptr,
        input_ptr,
        cu_num_tokens_ptr,
        replace_from=0,
        replace_to=0,
        MAX_NUM_TOKENS=MAX_SPEC_LEN,
    )

    expected = torch.tensor([10, 10, 20, 20, 20, 30, 30])
    assert torch.equal(output_ptr, expected)


def test_expand_batch_to_tokens():
    """测试 expand_batch_to_tokens 封装"""
    x = torch.tensor([10, 20, 30])
    cu_num_tokens = torch.tensor([2, 5, 7])
    num_tokens = 7

    with patch("your_module.spec_decode.expand_pytorch") as mock_kernel:
        expand_batch_to_tokens(x, cu_num_tokens, num_tokens)
        mock_kernel.assert_called_once()
        args = mock_kernel.call_args[0]
        assert (args[1] == x).all()
        assert (args[2] == cu_num_tokens).all()

    # 实际运行
    result = expand_batch_to_tokens(x, cu_num_tokens, num_tokens)
    expected = torch.tensor([10, 10, 20, 20, 20, 30, 30])
    assert torch.equal(result, expected)


def test_sample_recovered_tokens_pytorch_ngram():
    """测试 n-gram 模式下的 recovered token 采样"""
    output_token_ids = torch.empty(2, dtype=torch.int32)
    cu_num_draft_tokens = torch.tensor([1, 2])
    draft_token_ids = torch.tensor([1, 2])
    draft_probs = None
    target_probs = torch.tensor([
        [0.1, 0.2, 0.7],  # draft=1 → 排除后最大是2
        [0.3, 0.3, 0.4],  # draft=2 → 排除后最大是0或1，q 决定
    ])
    q = torch.tensor([
        [0.1, 0.2, 0.7],  # req0
        [0.5, 0.4, 0.1],  # req1
    ])
    vocab_size = 3

    sample_recovered_tokens_pytorch(
        output_token_ids,
        cu_num_draft_tokens,
        draft_token_ids,
        draft_probs,
        target_probs,
        q,
        vocab_size,
        IS_NGRAM=True,
    )

    # req0: pos0: 排除 token1 → [0.1, 0, 0.7] → prob/q = [1, 0, 1] → argmax=2
    # req1: pos0: 排除 token2 → [0.3, 0.3, 0] → prob/q = [0.6, 0.75, 0] → argmax=1
    #      pos1: same q → [0.6, 0.75] → still 1
    assert output_token_ids[0].item() == 2
    assert output_token_ids[1].item() == 1


def test_sample_recovered_tokens_pytorch_autoregressive():
    """测试自回归模型 recovered token 采样"""
    output_token_ids = torch.empty(2, dtype=torch.int32)
    cu_num_draft_tokens = torch.tensor([1, 1])
    draft_token_ids = torch.tensor([0, 1])
    draft_probs = torch.tensor([
        [0.6, 0.1, 0.3],
        [0.2, 0.7, 0.1],
    ])
    target_probs = torch.tensor([
        [0.8, 0.1, 0.1],
        [0.3, 0.6, 0.1],
    ])
    q = torch.tensor([
        [0.5, 0.3, 0.2],
        [0.1, 0.8, 0.1],
    ])
    vocab_size = 3

    sample_recovered_tokens_pytorch(
        output_token_ids,
        cu_num_draft_tokens,
        draft_token_ids,
        draft_probs,
        target_probs,
        q,
        vocab_size,
        IS_NGRAM=False,
    )

    # req0: prob = max(0.8-0.6, 0), (0.1-0.1), (0.1-0.3) → [0.2, 0, 0]
    #       prob/q = [0.4, 0, 0] → argmax=0
    # req1: prob = [0.1, -0.1→0, 0] → [0.1, 0, 0] → prob/q = [1.0, 0, 0] → argmax=0
    assert output_token_ids[0].item() == 0
    assert output_token_ids[1].item() == 0


@pytest.fixture
def setup_inputs():
    """构建标准输入"""
    batch_size = 2
    vocab_size = 4
    device = torch.device("cpu")

    num_draft_tokens = [2, 1]
    cu_num_draft_tokens = torch.tensor([0, 2, 3],
                                       dtype=torch.int32,
                                       device=device)  # 前缀和
    num_tokens = 3

    draft_token_ids = torch.tensor([1, 2, 3], dtype=torch.int32, device=device)
    draft_probs = torch.tensor([
        [0.5, 0.5, 0.0, 0.0],
        [0.0, 0.9, 0.1, 0.0],
        [0.8, 0.2, 0.0, 0.0],
    ],
                               device=device)
    target_probs = torch.tensor([
        [0.7, 0.3, 0.0, 0.0],
        [0.0, 0.95, 0.05, 0.0],
        [0.4, 0.6, 0.0, 0.0],
    ],
                                device=device)
    bonus_token_ids = torch.tensor([[10], [20]],
                                   dtype=torch.int32,
                                   device=device)

    return {
        "draft_token_ids": draft_token_ids,
        "num_draft_tokens": num_draft_tokens,
        "max_spec_len": 4,
        "cu_num_draft_tokens": cu_num_draft_tokens,
        "draft_probs": draft_probs,
        "target_probs": target_probs,
        "bonus_token_ids": bonus_token_ids,
        "device": device,
        "vocab_size": vocab_size,
        "num_tokens": num_tokens,
        "batch_size": batch_size,
    }


def test_rejection_sample_greedy_all(setup_inputs):
    """全贪婪模式：只运行 greedy 分支"""
    inputs = setup_inputs
    metadata = MockSamplingMetadata(all_greedy=True)
    inputs["sampling_metadata"] = metadata

    with patch("torch.Generator") as mock_gen:
        output = rejection_sample(**inputs, sampling_metadata=metadata)

    assert output.shape == (2, 5)  # [batch, max_spec_len+1]
    assert output.dtype == torch.int32

    # 请求0: draft=[1,2], target_argmax=[0,1] → 第一个就不匹配 → 只写 target_argmax[0]=0
    #        但因为不匹配，bonus 不加
    assert output[0, 0].item() == 0
    assert output[0, 1].item() == PLACEHOLDER_TOKEN_ID

    # 请求1: draft=[3], target_argmax=[3] → 匹配 → 写3，加 bonus=20
    assert output[1, 0].item() == 3
    assert output[1, 1].item() == 20


def test_rejection_sample_random(setup_inputs):
    """随机采样模式：运行完整流程"""
    inputs = setup_inputs
    gen1, gen2 = torch.Generator(), torch.Generator()
    metadata = MockSamplingMetadata(all_greedy=False,
                                    all_random=False,
                                    generators={
                                        0: gen1,
                                        1: gen2
                                    })
    inputs["sampling_metadata"] = metadata

    # Mock 生成 uniform 随机数
    with patch("torch.empty") as mock_empty, \
         patch("torch.Tensor.exponential_") as mock_exp, \
         patch("your_module.spec_decode.generate_uniform_probs") as mock_unif:

        unif_probs = torch.tensor([0.6, 0.96, 0.5])
        mock_unif.return_value = unif_probs

        output = rejection_sample(**inputs, sampling_metadata=metadata)

    # 请求0:
    #   pos0: draft=1, p=0.5, q=0.7 → ratio=1.4 > 0.6 → accept → 1
    #   pos1: draft=2, p=0.9, q=0.95 → ratio≈1.055 > 0.96 → accept → 2
    #   bonus: 10
    assert output[0, 0].item() == 1
    assert output[0, 1].item() == 2
    assert output[0, 2].item() == 10

    # 请求1:
    #   pos0: draft=3, p=0.8, q=0.4 → ratio=0.5 < 0.5? → no (>=) → accept → 3
    #   bonus: 20
    assert output[1, 0].item() == 3
    assert output[1, 1].item() == 20


def test_rejection_sample_with_recovered_token_rejection(setup_inputs):
    """测试拒绝后使用 recovered token"""
    inputs = setup_inputs
    metadata = MockSamplingMetadata(all_greedy=False)

    # 修改 uniform prob 使第一次拒绝
    with patch("your_module.spec_decode.generate_uniform_probs") as mock_unif:
        mock_unif.return_value = torch.tensor([0.8, 0.96, 0.5])

        # Mock sample_recovered_tokens
        with patch(
                "your_module.spec_decode.sample_recovered_tokens") as mock_rec:
            mock_rec.return_value = torch.tensor([99, 88, 77])

            output = rejection_sample(**inputs, sampling_metadata=metadata)

    # 请求0: 第一个就被拒绝 → 使用 recovered=99
    assert output[0, 0].item() == 99
    assert output[0, 1].item() == PLACEHOLDER_TOKEN_ID  # 后续不继续
    # bonus 不加

    # 请求1: 仍然接受
    assert output[1, 0].item() == 3
    assert output[1, 1].item() == 20
