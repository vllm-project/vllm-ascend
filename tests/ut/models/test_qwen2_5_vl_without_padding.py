import pytest
import torch
import torch.nn.functional as F
from pytest_mock import MockerFixture
from vllm.model_executor.models.qwen2_5_vl import \
    Qwen2_5_VLForConditionalGeneration

from tests.ut.base import PytestBase
from vllm_ascend.models.qwen2_5_vl_without_padding import (
    AscendQwen2_5_VisionAttention_Without_Padding,
    AscendQwen2_5_VisionBlock_Without_Padding,
    AscendQwen2_5_VisionPatchEmbed_Without_Padding,
    AscendQwen2_5_VisionTransformer_Without_Padding,
    AscendQwen2_5_VLForConditionalGeneration_Without_Padding)


class TestAscendQwen2_5_VisionAttention_Without_Padding(PytestBase):

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
            "vllm_ascend.models.qwen2_5_vl_without_padding.Qwen2_5_VisionAttention.__init__"
        )

        attention = AscendQwen2_5_VisionAttention_Without_Padding(
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

    def test_vit_init_should_normal(self, mocker: MockerFixture):
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
        assert vit.embed_dim == 1000
        assert vit.hidden_size_per_attention_head == 10

    def test_vit_init_should_raise_error(self, mocker: MockerFixture):
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

    def test_vit_forward(self, mocker: MockerFixture):
        mocker.patch("torch.nn.Module.__setattr__")
        mocker.patch("torch.nn.Module.__getattr__")
        mocker.patch("torch.nn.Module.__delattr__")
        attention = self.init_attention(mocker=mocker)
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


class TestAscendQwen2_5_VisionBlock_Without_Padding(PytestBase):

    def init_vision_block(
        self,
        mocker,
        dim=100,
        num_heads=10,
        mlp_hidden_dim=100,
    ):
        mocker_vit = mocker.patch(
            "vllm.model_executor.models.qwen2_5_vl.Qwen2_5_VisionBlock.__init__",
            return_value=None,
        )

        mocker_attn = mocker.patch(
            "vllm_ascend.models.qwen2_5_vl_without_padding.AscendQwen2_5_VisionAttention_Without_Padding.__init__",
            return_value=None,
        )

        mocker.patch("torch.nn.Module.__setattr__")
        mocker.patch("torch.nn.Module.__getattr__")
        mocker.patch("torch.nn.Module.__delattr__")
        vision_block = AscendQwen2_5_VisionBlock_Without_Padding(
            dim=dim,
            num_heads=num_heads,
            mlp_hidden_dim=mlp_hidden_dim,
        )
        args, kwargs = mocker_vit.call_args
        assert args == (dim, num_heads, mlp_hidden_dim, F.silu, None, None, "")
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
        assert isinstance(vision_block,
                          AscendQwen2_5_VisionBlock_Without_Padding)

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


class TestAscendQwen2_5_VisionPatchEmbed_Without_Padding(PytestBase):

    def test_forward(self):
        patch_embed = AscendQwen2_5_VisionPatchEmbed_Without_Padding()

        ret = patch_embed(torch.rand((120, 1176)))
        assert ret.shape == (120, 1152)


class TestAscendQwen2_5_VisionTransformer_Without_Padding(PytestBase):

    input_data = torch.tensor([[0.1, 0.2], [0.3, 0.4]])

    def init_vision_transformer(
        self,
        mocker,
    ):
        norm_eps = 1e-6
        vision_config = mocker.MagicMock()
        vision_config.patch_size = 16
        vision_config.temporal_patch_size = 2
        vision_config.in_channels = 3
        vision_config.hidden_act = "gelu"
        vision_config.depth = 0
        vision_config.hidden_size = 1280
        vision_config.num_heads = 16

        mocker.patch("torch.nn.Module.__setattr__")
        mocker.patch("torch.nn.Module.__getattr__")
        mocker.patch("torch.nn.Module.__delattr__")
        mocker_vit = mocker.patch(
            "vllm.model_executor.models.qwen2_5_vl.Qwen2_5_VisionTransformer.__init__",
            return_value=None,
        )
        mocker_vision_rotary_embedding = mocker.patch(
            "vllm_ascend.models.qwen2_5_vl.AscendQwen2_5_VisionRotaryEmbedding.__init__",
            return_value=None,
        )
        mocker.patch(
            "vllm_ascend.models.qwen2_5_vl_without_padding.AscendQwen2_5_VisionBlock_Without_Padding.__init__",
            return_value=None,
        )
        mocker.patch(
            "vllm_ascend.models.qwen2_5_vl_without_padding.AscendQwen2_5_VisionPatchEmbed_Without_Padding.__init__",
            return_value=None,
        )
        mocker.patch(
            "vllm_ascend.models.qwen2_5_vl_without_padding.parallel_state.get_tensor_model_parallel_world_size",
            return_value=1,
        )
        mocker.patch(
            "vllm_ascend.models.qwen2_5_vl_without_padding.parallel_state.get_tensor_model_parallel_rank",
            return_value=0,
        )
        mocker.patch("vllm.distributed.utils.divide", return_value=100)

        vision_transformer = AscendQwen2_5_VisionTransformer_Without_Padding(
            vision_config,
            norm_eps,
        )
        args, kwargs = mocker_vit.call_args
        assert args == (vision_config, norm_eps, None, "")
        assert not kwargs
        mocker_vision_rotary_embedding.assert_called_once()
        return vision_transformer

    def test_init_vision_transformer(self, mocker: MockerFixture):
        vision_transformer = self.init_vision_transformer(mocker)
        assert isinstance(vision_transformer,
                          AscendQwen2_5_VisionTransformer_Without_Padding)

    @pytest.mark.parametrize(
        "interleaved, expected",
        [
            (
                False,
                torch.tensor([
                    input_data[0, 0].cos(),
                    input_data[0, 1].cos(),
                    input_data[0, 0].cos(),
                    input_data[0, 1].cos(),
                    input_data[1, 0].cos(),
                    input_data[1, 1].cos(),
                    input_data[1, 0].cos(),
                    input_data[1, 1].cos(),
                ]),
            ),
            (
                True,
                torch.tensor([
                    input_data[0, 0].cos(),
                    input_data[0, 0].cos(),
                    input_data[0, 1].cos(),
                    input_data[0, 1].cos(),
                    input_data[1, 0].cos(),
                    input_data[1, 0].cos(),
                    input_data[1, 1].cos(),
                    input_data[1, 1].cos(),
                ]),
            ),
        ],
    )
    def test_cal_cos_sin(self, interleaved, expected, mocker: MockerFixture):
        vision_transformer = self.init_vision_transformer(mocker)
        vision_transformer.__dict__["interleaved"] = interleaved
        vision_transformer.__dict__["hidden_size_per_attention_head"] = 2
        vision_transformer.hidden_size_per_attention_head = 4
        cos_new, _ = vision_transformer.cal_cos_sin(self.input_data)
        assert cos_new.shape == (1, 4, 1, 2)
        assert torch.allclose(cos_new.view(-1), expected)

    def test_forward(self, mocker: MockerFixture):
        vision_transformer = self.init_vision_transformer(mocker)
        x = torch.randn(1, 3, 224, 224)
        grid_thw = torch.tensor([[1, 4, 4]])
        mocker_patch_embed = mocker.patch.object(
            vision_transformer,
            "patch_embed",
            side_effect=lambda _: torch.randn(16, 512),  # noqa
        )
        mocker_rot_pos_emb = mocker.patch.object(
            vision_transformer,
            "rot_pos_emb",
            side_effect=lambda _: torch.randn(16, 64),  # noqa
        )
        mocker_get_window_index = mocker.patch.object(
            vision_transformer,
            "get_window_index",
            side_effect=lambda _: (torch.arange(8), [4, 8, 12, 16]),  # noqa
        )
        mocker_cal_cos_sin = mocker.patch.object(
            vision_transformer,
            "cal_cos_sin",
            side_effect=lambda _:
            (torch.randn(16, 32), torch.randn(16, 32)),  # noqa
        )
        mocker_merger = mocker.patch.object(
            vision_transformer,
            "merger",
            side_effect=lambda _: torch.randn(16, 256),  # noqa
        )
        vision_transformer.__dict__["vision_blocks"] = [
            lambda *args, **kwargs: torch.randn(16, 1, 512)  # noqa
        ]
        vision_transformer.__dict__["patch_embed"] = mocker_patch_embed
        vision_transformer.__dict__["rot_pos_emb"] = mocker_rot_pos_emb
        vision_transformer.__dict__[
            "get_window_index"] = mocker_get_window_index
        vision_transformer.__dict__["cal_cos_sin"] = mocker_cal_cos_sin
        vision_transformer.__dict__["merger"] = mocker_merger
        vision_transformer.__dict__["fullatt_block_indexes"] = [0, 2]
        vision_transformer.__dict__["spatial_merge_unit"] = 2
        ret = vision_transformer.forward(x, grid_thw)
        assert ret.shape == (8, 256)
        mocker_patch_embed.assert_called_with(x)
        mocker_rot_pos_emb.assert_called_with(grid_thw)
        mocker_get_window_index.assert_called_with(grid_thw)
        mocker_cal_cos_sin.assert_called_once()
        mocker_merger.assert_called_once()


class TestAscendQwen2_5_VLForConditionalGeneration_Without_Padding(PytestBase):

    def test_init_vl_for_conditional_generation(self, mocker: MockerFixture):
        vllm_config = mocker.MagicMock()
        vllm_config.vision_config = "vision_config"
        vllm_config.rms_norm_eps = 1e-5
        mocker.patch("torch.nn.Module.__setattr__")
        mocker.patch("torch.nn.Module.__getattr__")
        mocker.patch("torch.nn.Module.__delattr__")
        mocker_vl = mocker.patch(
            "vllm.model_executor.models.qwen2_5_vl.Qwen2_5_VLForConditionalGeneration.__init__",
            return_value=None,
        )
        mocker_vit = mocker.patch(
            "vllm_ascend.models.qwen2_5_vl_without_padding.AscendQwen2_5_VisionTransformer_Without_Padding.__init__",
            return_value=None,
        )

        vl_for_conditional_generation = AscendQwen2_5_VLForConditionalGeneration_Without_Padding(
            vllm_config=vllm_config)
        args, kwargs = mocker_vl.call_args
        assert not args
        assert kwargs == {"vllm_config": vllm_config, "prefix": ""}
        mocker_vit.assert_called_once()
        assert isinstance(
            vl_for_conditional_generation,
            AscendQwen2_5_VLForConditionalGeneration_Without_Padding,
        )

    def test_overridden_methods(self):
        self.assert_method_overridden(
            AscendQwen2_5_VLForConditionalGeneration_Without_Padding,
            Qwen2_5_VLForConditionalGeneration,
            "_process_image_input",
        )

        self.assert_method_overridden(
            AscendQwen2_5_VLForConditionalGeneration_Without_Padding,
            Qwen2_5_VLForConditionalGeneration,
            "_process_video_input",
        )

    @staticmethod
    def assert_method_overridden(subclass, parent, method_name: str):
        """assert subclass override parent method"""
        parent_func = parent.__dict__.get(method_name)
        child_func = subclass.__dict__.get(method_name)

        assert child_func is not None, f"{subclass.__name__} should defined {method_name}"
        assert child_func is not parent_func, f"{method_name} should override in {subclass.__name__}"
