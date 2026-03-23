import vllm
from vllm.model_executor.models.utils import AutoWeightsLoader


class AutoWeightsLoaderWithPreprocess(AutoWeightsLoader):
    def _preprocess(self):
        for _, v in self.module.named_parameters():
            if getattr(v, "transposed", False):
                setattr(v, "transposed", False)  # noqa: B010
                v.data = v.data.transpose(1, 2)

    def load_weights(self, *args, **kwargs):
        self._preprocess()
        super().load_weights(*args, **kwargs)


vllm.model_executor.models.utils.AutoWeightsLoader = AutoWeightsLoaderWithPreprocess
