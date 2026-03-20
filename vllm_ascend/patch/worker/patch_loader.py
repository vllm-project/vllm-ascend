import vllm
from vllm.model_executor.models.utils import AutoWeightsLoader


class AutoWeightsLoaderWithTrans(AutoWeightsLoader):
    def _preprocess(self):
        for k, v in self.module.named_parameters():
            if getattr(v, "transposed", False):
                setattr(v, "transposed", False)  # noqa: B010
                v.data = v.data.transpose(1, 2)

    def load_weights(self, model_path: str):
        self._preprocess()
        super().load_weights(model_path)


vllm.model_executor.models.utils.AutoWeightsLoader = AutoWeightsLoaderWithTrans
