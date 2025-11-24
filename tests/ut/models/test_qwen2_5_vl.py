from pytest_mock import MockerFixture

from tests.ut.base import PytestBase
from vllm_ascend.models.qwen2_5_vl import \
    AscendQwen2_5_VLForConditionalGeneration


class TestAscendQwen2_5_VLForConditionalGeneration(PytestBase):

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
            "vllm_ascend.models.qwen2_5_vl.AscendQwen2_5_VisionTransformer.__init__",
            return_value=None,
        )

        vl_for_conditional_generation = AscendQwen2_5_VLForConditionalGeneration(
            vllm_config=vllm_config)
        args, kwargs = mocker_vl.call_args
        assert not args
        assert kwargs == {"vllm_config": vllm_config, "prefix": ""}
        mocker_vit.assert_called_once()
        assert isinstance(
            vl_for_conditional_generation,
            AscendQwen2_5_VLForConditionalGeneration,
        )
