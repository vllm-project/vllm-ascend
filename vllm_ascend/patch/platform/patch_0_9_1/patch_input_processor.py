from typing import Union

from packaging.version import Version
from transformers import ProcessorMixin
from transformers import __version__ as TRANSFORMERS_VERSION
from typing_extensions import TypeVar
from vllm.inputs.registry import InputProcessingContext, InputContext

_P = TypeVar("_P", bound=ProcessorMixin, default=ProcessorMixin)


# Patch for InputProcessingContext to handle a specific version of transformers
# Remove this after we drop support for vLLM v0.9.1
def get_hf_processor(
    self,
    typ: Union[type[_P], tuple[type[_P], ...]] = ProcessorMixin,
    /,
    **kwargs: object,
) -> _P:
    # Transformers 4.53.0 has issue with passing tokenizer to
    # initialize processor. We disable it for this version.
    # See: https://github.com/vllm-project/vllm/issues/20224
    if Version(TRANSFORMERS_VERSION) != Version("4.53.0"):
        kwargs["tokenizer"] = self.tokenizer
    return InputContext.get_hf_processor(
        typ,
        **kwargs,
    )


InputProcessingContext.get_hf_processor = get_hf_processor
