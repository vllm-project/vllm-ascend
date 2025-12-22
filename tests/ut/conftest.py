# limitations under the License.
# This file is a part of the vllm-ascend project.
#
import sys
from unittest.mock import MagicMock

from vllm_ascend.utils import adapt_patch  # noqa E402

sys.modules['torch_npu'] = MagicMock(npu=MagicMock(current_device=MagicMock(
    return_value=0)))
sys.modules['torch_npu._inductor'] = MagicMock()

triton_runtime = MagicMock()
triton_runtime.driver.active.utils.get_device_properties.return_value = {
    'num_aic': 8,
    'num_vectorcore': 8,
}
sys.modules['triton.runtime'] = triton_runtime

adapt_patch()
adapt_patch(True)
