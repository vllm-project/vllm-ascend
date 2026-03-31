import torch


# torch_npu.argsort does not sipport bool now, it will support it in the future.
# TODO When the operator of argsort is ready, this patch must be removed.
def _argsort(tensor, *args, **kwargs):
    if tensor.dtype == torch.bool:
        # If it is not stable, it will have redundant outputs.
        kwargs["stable"] = True
        return torch.argsort(tensor.to(torch.int32), *args, **kwargs)
    else:
        return torch.argsort(tensor, *args, **kwargs)


class _TorchWrapper:
    def __init__(self):
        self._raw_torch = torch

    def __getattr__(self, name):
        if name == "argsort":
            return _argsort
        else:
            return getattr(self._raw_torch, name)


_is_patched = False


# patch argsort only for torch in gdn_attn
def patch_torch_npu_argsort():
    global _is_patched
    if not _is_patched:
        import vllm.v1.attention.backends.gdn_attn as gdn_attn

        gdn_attn.torch = _TorchWrapper()
        _is_patched = True
