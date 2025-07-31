import pytest
from pytest_mock import MockFixture

from tests.ut.base import PytestBase
from vllm_ascend.multistream.decorator import set_multistream_support


class Context:

    def __init__(self, attn_metadata=None):
        self.attn_metadata = attn_metadata


class TestDecorator(PytestBase):

    @pytest.mark.parametrize(
        'layer_context, microbatch_context, expected_metadata', [
            ((-1, None, None), -1, {
                "original": True
            }),
            ((-1, None, None), 0, {
                "original": True
            }),
            ((0, None, None), -1, {
                "original": True
            }),
            ((0, None, [{
                "new": True
            }]), 0, {
                "new": True
            }),
        ])
    def test_decorator(self, mocker: MockFixture, layer_context,
                       microbatch_context, expected_metadata):

        def context_func():
            return Context(attn_metadata={"original": True})

        mocker.patch(
            'vllm_ascend.multistream.decorator.get_multistream_layer_context',
            return_value=layer_context)
        mocker.patch(
            'vllm_ascend.multistream.decorator.get_multistream_microbatch_context',
            return_value=microbatch_context)

        context = set_multistream_support()(context_func)()
        assert context.attn_metadata == expected_metadata
