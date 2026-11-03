import pytest
import torch

from vllm_ascend.worker.utils import disable_compilation


@pytest.mark.parametrize("initial", [False, True])
@pytest.mark.parametrize("fail", [False, True])
def test_disable_compilation_restores_nested_modules(initial, fail):
    backbone = torch.nn.Module()
    backbone.do_not_compile = initial
    language_model = torch.nn.Module()
    language_model.model = backbone
    model = torch.nn.Module()
    model.language_model = language_model
    # Include the root and a shared module with a different initial state.
    model.do_not_compile = not initial
    model.shared_backbone = backbone

    def run():
        with disable_compilation(model):
            assert model.do_not_compile is True
            assert backbone.do_not_compile is True
            with disable_compilation(model):
                assert backbone.do_not_compile is True
            assert backbone.do_not_compile is True
            if fail:
                raise RuntimeError("warmup failed")

    if fail:
        with pytest.raises(RuntimeError, match="warmup failed"):
            run()
    else:
        run()

    assert model.do_not_compile is (not initial)
    assert backbone.do_not_compile is initial
    assert not hasattr(language_model, "do_not_compile")


def test_disable_compilation_without_compiled_modules():
    model = torch.nn.Sequential(torch.nn.Identity())
    with disable_compilation(model):
        assert all(not hasattr(module, "do_not_compile") for module in model.modules())
    assert all(not hasattr(module, "do_not_compile") for module in model.modules())
