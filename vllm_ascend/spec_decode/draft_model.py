from vllm.v1.spec_decode.eagle import DraftModelProposer as VllmDraftModelProposer

from vllm_ascend.spec_decode.eagle_proposer import SpecDecodeBaseProposer
from vllm_ascend.utils import fix_new_class_closures, make_cls

# to reuse all functions
DraftModelProposer = make_cls(
    VllmDraftModelProposer.__name__, (SpecDecodeBaseProposer,), VllmDraftModelProposer.__dict__
)
fix_new_class_closures(DraftModelProposer)

# to pass check of isinstance check
VllmDraftModelProposer.register(DraftModelProposer)
