# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
add_op_to_compiled_list()

if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnn PRIVATE
<<<<<<< HEAD
=======
<<<<<<<< HEAD:csrc/attention/quant_lightning_indexer_custom/op_host/CMakeLists.txt
>>>>>>> db4cc4304 ([BugFix] avoid experimental QLI overriding native QLI (#10900))
<<<<<<<< HEAD:csrc/attention/vllm_quant_lightning_indexer/op_host/CMakeLists.txt
        vllm_quant_lightning_indexer_def.cpp
========
        quant_lightning_indexer_custom_def.cpp
>>>>>>>> 6e9fabbc3 ([BugFix] Fix duplicate symbol for custom QuantLightningIndexer operator (#10744)):csrc/attention/quant_lightning_indexer_custom/op_host/CMakeLists.txt
<<<<<<< HEAD
=======
========
        vllm_quant_lightning_indexer_def.cpp
>>>>>>>> db4cc4304 ([BugFix] avoid experimental QLI overriding native QLI (#10900)):csrc/attention/vllm_quant_lightning_indexer/op_host/CMakeLists.txt
>>>>>>> db4cc4304 ([BugFix] avoid experimental QLI overriding native QLI (#10900))
     )
endif()

add_ops_compile_options(
<<<<<<< HEAD
=======
<<<<<<<< HEAD:csrc/attention/quant_lightning_indexer_custom/op_host/CMakeLists.txt
>>>>>>> db4cc4304 ([BugFix] avoid experimental QLI overriding native QLI (#10900))
<<<<<<<< HEAD:csrc/attention/vllm_quant_lightning_indexer/op_host/CMakeLists.txt
    OP_NAME VllmQuantLightningIndexer
========
    OP_NAME QuantLightningIndexerCustom
>>>>>>>> 6e9fabbc3 ([BugFix] Fix duplicate symbol for custom QuantLightningIndexer operator (#10744)):csrc/attention/quant_lightning_indexer_custom/op_host/CMakeLists.txt
<<<<<<< HEAD
=======
========
    OP_NAME VllmQuantLightningIndexer
>>>>>>>> db4cc4304 ([BugFix] avoid experimental QLI overriding native QLI (#10900)):csrc/attention/vllm_quant_lightning_indexer/op_host/CMakeLists.txt
>>>>>>> db4cc4304 ([BugFix] avoid experimental QLI overriding native QLI (#10900))
    OPTIONS --cce-auto-sync=off
            -Wno-deprecated-declarations
            -Werror
            -mllvm -cce-vf-remove-membar=false
            -mllvm -cce-aicore-hoist-movemask=false
)

if (NOT BUILD_OPS_RTY_KERNEL)
<<<<<<< HEAD
=======
<<<<<<<< HEAD:csrc/attention/quant_lightning_indexer_custom/op_host/CMakeLists.txt
>>>>>>> db4cc4304 ([BugFix] avoid experimental QLI overriding native QLI (#10900))
<<<<<<<< HEAD:csrc/attention/vllm_quant_lightning_indexer/op_host/CMakeLists.txt
    add_modules_sources(OPTYPE vllm_quant_lightning_indexer ACLNNTYPE aclnn)
endif()
========
    add_modules_sources(OPTYPE quant_lightning_indexer_custom ACLNNTYPE aclnn)
endif()
>>>>>>>> 6e9fabbc3 ([BugFix] Fix duplicate symbol for custom QuantLightningIndexer operator (#10744)):csrc/attention/quant_lightning_indexer_custom/op_host/CMakeLists.txt
<<<<<<< HEAD
=======
========
    add_modules_sources(OPTYPE vllm_quant_lightning_indexer ACLNNTYPE aclnn)
endif()
>>>>>>>> db4cc4304 ([BugFix] avoid experimental QLI overriding native QLI (#10900)):csrc/attention/vllm_quant_lightning_indexer/op_host/CMakeLists.txt
>>>>>>> db4cc4304 ([BugFix] avoid experimental QLI overriding native QLI (#10900))
