# -*- coding: UTF-8 -*-
# Loader for the DeepSeek-V4-Flash-DSpark quant adapter.
# Place at: msmodelslim/model/deepseek_v4/dspark_loader.py
from msmodelslim.model.plugin_factory.base_loader import BaseModelAdapterLoader


class DeepseekV4DSparkAdapterLoader(BaseModelAdapterLoader):
    ADAPTER_CLASS_PATH = "msmodelslim.model.deepseek_v4.dspark_adapter:DeepSeekV4DSparkModelAdapter"
