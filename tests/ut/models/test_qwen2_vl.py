import pytest
import torch
from pytest_mock import MockerFixture
from vllm.model_executor.layers.activation import QuickGELU

from tests.ut.base import PytestBase
from vllm_ascend.models.qwen2_vl import (AscendQwen2VisionAttention,
                                         AscendQwen2VisionBlock)


class TestAscendQwen2VisionAttention(PytestBase):

    def init_attention(
        self,
        mocker,
        embed_dim=1000,
        num_heads=10,
        projection_size=100,
        quant_config=None,
        prefix="",
    ):
        mocker_attn = mocker.patch(
            "vllm_ascend.models.qwen2_vl.Qwen2VisionAttention.__init__")

        attention = AscendQwen2VisionAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            projection_size=projection_size,
            quant_config=quant_config,
            prefix=prefix,
        )
        args, kwargs = mocker_attn.call_args
        assert args == (embed_dim, num_heads, projection_size, None, "")
        assert not kwargs
        attention.num_attention_heads_per_partition = num_heads
        return attention

    def test_attn_init_should_normal(self, mocker: MockerFixture):
        embed_dim = 1000
        num_heads = 10
        projection_size = 100
        quant_config = None
        prefix = ""
        vit = self.init_attention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            projection_size=projection_size,
            quant_config=quant_config,
            prefix=prefix,
            mocker=mocker,
        )
        assert vit.hidden_size_per_attention_head == 10

    def test_attn_init_should_raise_error(self, mocker: MockerFixture):
        embed_dim = 1000
        num_heads = 7
        projection_size = 100
        quant_config = None
        prefix = ""
        with pytest.raises(AssertionError):
            # projection_size should divided by num heads
            self.init_attention(
                mocker=mocker,
                embed_dim=embed_dim,
                num_heads=num_heads,
                projection_size=projection_size,
                quant_config=quant_config,
                prefix=prefix,
            )

    def test_attn_forward(self, mocker: MockerFixture):
        attention = self.init_attention(mocker=mocker)
        mocker.patch("torch.nn.Module.__setattr__")
        mocker.patch("torch.nn.Module.__getattr__")
        mocker.patch("torch.nn.Module.__delattr__")
        x = torch.rand((100, 3, 10 * 3 * 128))  # s,b, head*3*head_dim
        cu_seqlens = torch.tensor([10, 50, 100])
        cos = torch.rand((1, 100, 1, 128))
        sin = torch.rand((1, 100, 1, 128))

        qkv = lambda x: (x, 0)  # noqa
        split_qkv = lambda x: [  #noqa
            torch.rand((100, 3, 10, 128)) for i in range(3)
        ]  # noqa
        npu_rotary_mul = lambda q, cos, sin: q  # noqa
        _npu_flash_attention_unpad = lambda **kwargs: kwargs["out"]  # noqa
        proj = lambda x: (x, 0)  # noqa

        mocker_qkv = mocker.patch.object(attention, "qkv", side_effect=qkv)
        mocker_split_qkv = mocker.patch.object(
            attention,
            "split_qkv",
            side_effect=split_qkv,
        )
        mocker_npu_rotary_mul = mocker.patch("torch_npu.npu_rotary_mul",
                                             side_effect=npu_rotary_mul)
        mocker_npu_flash_attention_unpad = mocker.patch(
            "torch_npu._npu_flash_attention_unpad",
            side_effect=_npu_flash_attention_unpad,
        )
        mocker_proj = mocker.patch.object(attention, "proj", side_effect=proj)
        attention.__dict__["qkv"] = mocker_qkv
        attention.__dict__["split_qkv"] = mocker_split_qkv
        attention.__dict__["npu_rotary_mul"] = mocker_npu_rotary_mul
        attention.__dict__["_npu_flash_attention_unpad"] = (
            mocker_npu_flash_attention_unpad)
        attention.__dict__["proj"] = mocker_proj

        output = attention.forward(
            x=x,
            cu_seqlens=cu_seqlens,
            cos=cos,
            sin=sin,
        )
        qkv_args, qkv_kwargs = mocker_qkv.call_args
        assert qkv_args == (x, )
        assert not qkv_kwargs

        split_qkv_args, split_qkv_kwargs = mocker_split_qkv.call_args
        assert split_qkv_args == (x, )
        assert not split_qkv_kwargs

        npu_rotary_mul_args, npu_rotary_mul_kwargs = mocker_npu_rotary_mul.call_args
        assert npu_rotary_mul_args[1:] == (cos, sin)
        assert npu_rotary_mul_args[0].shape == torch.Size([3, 100, 10, 128])
        assert not npu_rotary_mul_kwargs

        assert output.shape == torch.Size([100, 3, 1280])


class TestAscendQwen2VisionBlock(PytestBase):

    def init_vision_block(
        self,
        mocker,
        dim=100,
        num_heads=10,
        mlp_ratio=0.5,
    ):
        mocker_vit = mocker.patch(
            "vllm.model_executor.models.qwen2_vl.Qwen2VisionBlock.__init__",
            return_value=None,
        )

        mocker_attn = mocker.patch(
            "vllm_ascend.models.qwen2_vl.AscendQwen2VisionAttention.__init__",
            return_value=None,
        )

        mocker.patch("torch.nn.Module.__setattr__")
        mocker.patch("torch.nn.Module.__getattr__")
        mocker.patch("torch.nn.Module.__delattr__")
        vision_block = AscendQwen2VisionBlock(
            dim=dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
        )
        args, kwargs = mocker_vit.call_args
        assert args == (dim, num_heads, mlp_ratio, QuickGELU, None, None, "")
        assert not kwargs

        args1, kwargs1 = mocker_attn.call_args
        assert not args1
        assert kwargs1 == {
            "embed_dim": dim,
            "num_heads": num_heads,
            "projection_size": dim,
            "quant_config": None,
            "prefix": ".attn",
        }
        return vision_block

    def test_init_vision_block_should_normal(
        self,
        mocker: MockerFixture,
    ):
        vision_block = self.init_vision_block(mocker)
        assert isinstance(vision_block, AscendQwen2VisionBlock)

    def test_vision_block_forward(self, mocker: MockerFixture):
        x = torch.randint(1, 100, (100, 3, 1280))  # s,b,d
        cu_seqlens = torch.tensor([10, 50, 100])
        cos = torch.rand((1, 100, 1, 128))
        sin = torch.rand((1, 100, 1, 128))
        vision_block = self.init_vision_block(mocker)
        mocker_attn = mocker.patch.object(vision_block, "attn", return_value=x)
        mocker_mlp = mocker.patch.object(vision_block, "mlp", return_value=x)
        vision_block.__dict__["attn"] = mocker_attn
        vision_block.__dict__["mlp"] = mocker_mlp

        output = vision_block.forward(x.clone(), cu_seqlens, cos, sin)

        _, attn_kwargs = mocker_attn.call_args
        assert attn_kwargs == {
            "cu_seqlens": cu_seqlens,
            "cos": cos,
            "sin": sin,
        }

        assert torch.all(x * 3 == output)
