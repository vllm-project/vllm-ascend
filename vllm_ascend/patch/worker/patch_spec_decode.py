import vllm.v1.spec_decode.draft_model as vllm_draft_model
import vllm.v1.spec_decode.eagle as vllm_eagle

from vllm_ascend.utils import add_abc_meta, fix_new_class_closures

# add metaclass ABCMeta
EagleProposerWithMeta = add_abc_meta(vllm_eagle.EagleProposer)
fix_new_class_closures(EagleProposerWithMeta)

vllm_eagle.EagleProposer = EagleProposerWithMeta


DraftModelProposerWithMeta = add_abc_meta(vllm_draft_model.DraftModelProposer)
fix_new_class_closures(DraftModelProposerWithMeta)

vllm_draft_model.DraftModelProposer = DraftModelProposerWithMeta
