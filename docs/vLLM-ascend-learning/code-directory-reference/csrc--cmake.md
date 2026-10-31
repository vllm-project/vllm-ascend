# csrc/cmake：目录与文件功能

[返回总索引](../code-directory-guide.md) · [学习文档入口](../README.md)

本分册覆盖 **73 个条目**。说明对应当前本地源码；右栏中的定义、注释或配置项用于追溯依据。

用途说明不等于运行验证；按文件角色归类的条目不保证已逐行确认全部算法。

## `csrc/cmake/`

算子工程的构建模块、脚本和外部依赖配置。

| 子目录 | 职责 |
| --- | --- |
| [modules/](../../../csrc/cmake/modules) | modules 子模块；所属范围：算子工程的构建模块、脚本和外部依赖配置。 |
| [scripts/](../../../csrc/cmake/scripts) | scripts 子模块；所属范围：算子工程的构建模块、脚本和外部依赖配置。 |
| [third_party/](../../../csrc/cmake/third_party) | third_party 子模块；所属范围：算子工程的构建模块、脚本和外部依赖配置。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [Third_Party_Open_Source_Software_List.yaml](../../../csrc/cmake/Third_Party_Open_Source_Software_List.yaml) | 所在模块的结构化配置或参数模板。 | 顶层项：`observability` |
| [aclnn_ops_transformer.h.in](../../../csrc/cmake/aclnn_ops_transformer.h.in) | 算子工程构建、依赖查找或安装生成模板。 | 依据文件格式及所在目录归类。 |
| [build_cache.cmake](../../../csrc/cmake/build_cache.cmake) | 算子工程构建、依赖查找或安装生成模板。 | 构建项：`vllm_ascend_build_cache_command` |
| [build_empty_package.cmake](../../../csrc/cmake/build_empty_package.cmake) | 算子工程构建、依赖查找或安装生成模板。 | 构建项：`cpack_empty_package` |
| [config.cmake](../../../csrc/cmake/config.cmake) | 算子工程构建、依赖查找或安装生成模板。 | CMake 构建规则。 |
| [config_utest.cmake](../../../csrc/cmake/config_utest.cmake) | 算子工程构建、依赖查找或安装生成模板。 | CMake 构建规则。 |
| [custom_build.cmake](../../../csrc/cmake/custom_build.cmake) | 算子工程构建、依赖查找或安装生成模板。 | 构建项：`op_host_aclnn`, `op_host_aclnnInner`, `op_host_aclnnExc` |
| [dependencies.cmake](../../../csrc/cmake/dependencies.cmake) | 算子工程构建、依赖查找或安装生成模板。 | CMake 构建规则。 |
| [func.cmake](../../../csrc/cmake/func.cmake) | 算子工程构建、依赖查找或安装生成模板。 | 构建项：`filter_copy_files`, `add_target_source`, `op_add_subdirectory` |
| [func_examples.cmake](../../../csrc/cmake/func_examples.cmake) | 算子工程构建、依赖查找或安装生成模板。 | 构建项：`add_execute_example` |
| [func_utest.cmake](../../../csrc/cmake/func_utest.cmake) | 算子工程构建、依赖查找或安装生成模板。 | 构建项：`_OpsTestUt_OpApi_Wno`, `_OpsTestUt_OpTiling_Wno`, `_OpsTestUt_OpKernel_Wno` |
| [gen_ops_info.cmake](../../../csrc/cmake/gen_ops_info.cmake) | 算子工程构建、依赖查找或安装生成模板。 | 构建项：`kernel_src_copy`, `add_ops_impl_target`, `add_ops_info_target` |
| [intf.cmake](../../../csrc/cmake/intf.cmake) | 算子工程构建、依赖查找或安装生成模板。 | CMake 构建规则。 |
| [intf_pub.cmake](../../../csrc/cmake/intf_pub.cmake) | 算子工程构建、依赖查找或安装生成模板。 | 构建项：`intf_pub`, `intf_pub_cxx14`, `intf_pub_cxx17` |
| [intf_pub_examples.cmake](../../../csrc/cmake/intf_pub_examples.cmake) | 算子工程构建、依赖查找或安装生成模板。 | CMake 构建规则。 |
| [intf_pub_linux.cmake](../../../csrc/cmake/intf_pub_linux.cmake) | 算子工程构建、依赖查找或安装生成模板。 | 构建项：`intf_pub`, `intf_pub_cxx14`, `intf_pub_cxx17` |
| [intf_pub_llt_gccnative.cmake](../../../csrc/cmake/intf_pub_llt_gccnative.cmake) | 算子工程构建、依赖查找或安装生成模板。 | 构建项：`intf_llt_pub`, `intf_llt_pub_asan`, `intf_llt_pub_asan_cxx14` |
| [intf_pub_utest.cmake](../../../csrc/cmake/intf_pub_utest.cmake) | 算子工程构建、依赖查找或安装生成模板。 | 构建项：`intf_pub_utest` |
| [makeself_built_in.cmake](../../../csrc/cmake/makeself_built_in.cmake) | 算子工程构建、依赖查找或安装生成模板。 | CMake 构建规则。 |
| [makeself_custom.cmake](../../../csrc/cmake/makeself_custom.cmake) | 算子工程构建、依赖查找或安装生成模板。 | CMake 构建规则。 |
| [obj_func.cmake](../../../csrc/cmake/obj_func.cmake) | 算子工程构建、依赖查找或安装生成模板。 | 构建项：`add_modules_sources`, `add_modules_sources_with_soc`, `${OPHOST_NAME}_infer_obj` |
| [opbuild.cmake](../../../csrc/cmake/opbuild.cmake) | 算子工程构建、依赖查找或安装生成模板。 | 构建项：`gen_opbuild_target`, `gen_op_host_${OPBUILD_PREFIX}`, `append_versioned_aclnn_outputs` |
| [package.cmake](../../../csrc/cmake/package.cmake) | 算子工程构建、依赖查找或安装生成模板。 | 构建项：`pack_custom`, `pack_tiling_sink`, `pack_built_in` |
| [rty_obj_func.cmake](../../../csrc/cmake/rty_obj_func.cmake) | 算子工程构建、依赖查找或安装生成模板。 | 构建项：`add_infer_modules`, `${OPHOST_NAME}_infer_obj`, `add_tiling_modules` |
| [runtimeKB.cmake](../../../csrc/cmake/runtimeKB.cmake) | 算子工程构建、依赖查找或安装生成模板。 | CMake 构建规则。 |
| [static.cmake](../../../csrc/cmake/static.cmake) | 算子工程构建、依赖查找或安装生成模板。 | 构建项：`${OPHOST_NAME}_static`, `${OPAPI_NAME}_static`, `resource_${compute_unit}_static` |
| [symbol.cmake](../../../csrc/cmake/symbol.cmake) | 算子工程构建、依赖查找或安装生成模板。 | 构建项：`gen_common_symbol`, `gen_ophost_symbol`, `${OPHOST_NAME}` |
| [ut.cmake](../../../csrc/cmake/ut.cmake) | 算子工程构建、依赖查找或安装生成模板。 | 构建项：`register_op_name`, `add_optiling_ut_modules`, `${OP_TILING_MODULE_NAME}_common_obj` |
| [variables.cmake](../../../csrc/cmake/variables.cmake) | 算子工程构建、依赖查找或安装生成模板。 | 构建项：`${OPHOST_NAME}_opdef_aclnn_obj`, `${OPHOST_NAME}_opdef_aclnn_inner_obj`, `${OPHOST_NAME}_opdef_aclnn_exclude_obj` |

## `csrc/cmake/modules/`

modules 子模块；所属范围：算子工程的构建模块、脚本和外部依赖配置。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [FindOPBASE.cmake](../../../csrc/cmake/modules/FindOPBASE.cmake) | 算子工程构建、依赖查找或安装生成模板。 | 构建项：`opsbase` |
| [FindPython.cmake](../../../csrc/cmake/modules/FindPython.cmake) | 算子工程构建、依赖查找或安装生成模板。 | 构建项：`get_python` |
| [Findaicpu.cmake](../../../csrc/cmake/modules/Findaicpu.cmake) | 算子工程构建、依赖查找或安装生成模板。 | CMake 构建规则。 |
| [Findalog.cmake](../../../csrc/cmake/modules/Findalog.cmake) | 算子工程构建、依赖查找或安装生成模板。 | 构建项：`slog_a`, `alog_a`, `alog_headers` |
| [Finddlog.cmake](../../../csrc/cmake/modules/Finddlog.cmake) | 算子工程构建、依赖查找或安装生成模板。 | 构建项：`dlog`, `dlog_headers` |
| [Findjson.cmake](../../../csrc/cmake/modules/Findjson.cmake) | 算子工程构建、依赖查找或安装生成模板。 | 构建项：`json` |
| [Findmetadef.cmake](../../../csrc/cmake/modules/Findmetadef.cmake) | 算子工程构建、依赖查找或安装生成模板。 | 构建项：`register`, `opp_registry`, `exe_graph` |
| [Findnnopbase.cmake](../../../csrc/cmake/modules/Findnnopbase.cmake) | 算子工程构建、依赖查找或安装生成模板。 | 构建项：`nnopbase` |
| [Findoptiling.cmake](../../../csrc/cmake/modules/Findoptiling.cmake) | 算子工程构建、依赖查找或安装生成模板。 | CMake 构建规则。 |
| [Findplatform.cmake](../../../csrc/cmake/modules/Findplatform.cmake) | 算子工程构建、依赖查找或安装生成模板。 | 构建项：`platform` |
| [Findruntime.cmake](../../../csrc/cmake/modules/Findruntime.cmake) | 算子工程构建、依赖查找或安装生成模板。 | CMake 构建规则。 |
| [Findsecurec.cmake](../../../csrc/cmake/modules/Findsecurec.cmake) | 算子工程构建、依赖查找或安装生成模板。 | 构建项：`c_sec_headers`, `c_sec_static`, `c_sec` |
| [Findtilingapi.cmake](../../../csrc/cmake/modules/Findtilingapi.cmake) | 算子工程构建、依赖查找或安装生成模板。 | 构建项：`tiling_api` |
| [Findunified_dlog.cmake](../../../csrc/cmake/modules/Findunified_dlog.cmake) | 算子工程构建、依赖查找或安装生成模板。 | 构建项：`unified_dlog`, `unified_dlog_headers` |

## `csrc/cmake/scripts/`

scripts 子模块；所属范围：算子工程的构建模块、脚本和外部依赖配置。

| 子目录 | 职责 |
| --- | --- |
| [custom/](../../../csrc/cmake/scripts/custom) | custom 子模块；所属范围：算子工程的构建模块、脚本和外部依赖配置。 |
| [examples/](../../../csrc/cmake/scripts/examples) | examples 子模块；所属范围：算子工程的构建模块、脚本和外部依赖配置。 |
| [utest/](../../../csrc/cmake/scripts/utest) | utest 子模块；所属范围：算子工程的构建模块、脚本和外部依赖配置。 |
| [util/](../../../csrc/cmake/scripts/util) | util 子模块；所属范围：算子工程的构建模块、脚本和外部依赖配置。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [check_version_compatible.py](../../../csrc/cmake/scripts/check_version_compatible.py) | 算子工程 检查 的生成、配置、编译或打包辅助；接口见右栏。 | 注释：版本兼容性检查；定义：`VersionChecker` |
| [convert_yaml.py](../../../csrc/cmake/scripts/convert_yaml.py) | 算子工程 卷积 的生成、配置、编译或打包辅助；接口见右栏。 | 注释：test_config.yaml 格式转换；定义：`load_test_config`, `extract_src_and_exclude`, `write_new_format` |
| [fix_format.sh](../../../csrc/cmake/scripts/fix_format.sh) | 算子工程 格式化 的构建、打包或处理脚本；按命令内容确认执行效果。 | Shell 脚本；按工程阶段和文件名归类。 |
| [parse_changed_files.py](../../../csrc/cmake/scripts/parse_changed_files.py) | 算子工程 parse_changed_files 的生成、配置、编译或打包辅助；接口见右栏。 | 注释：获取修改文件应触发的测试范围.；定义：`Module`, `Parser` |
| [prepare.sh](../../../csrc/cmake/scripts/prepare.sh) | 算子工程 准备 的构建、打包或处理脚本；按命令内容确认执行效果。 | 命令类别：`cmake`, `make` |

## `csrc/cmake/scripts/custom/`

custom 子模块；所属范围：算子工程的构建模块、脚本和外部依赖配置。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [help.info](../../../csrc/cmake/scripts/custom/help.info) | help.info 附属文件；当前按文件格式及目录定位用途。 | 依据文件格式及所在目录归类。 |
| [install.sh](../../../csrc/cmake/scripts/custom/install.sh) | 算子工程 install 的构建、打包或处理脚本；按命令内容确认执行效果。 | 函数：`log`, `upgrade`, `upgrade_proto` |
| [upgrade.sh](../../../csrc/cmake/scripts/custom/upgrade.sh) | 算子工程 upgrade 的构建、打包或处理脚本；按命令内容确认执行效果。 | 函数：`log`, `upgrade`, `upgrade_file` |

## `csrc/cmake/scripts/examples/`

examples 子模块；所属范围：算子工程的构建模块、脚本和外部依赖配置。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [get_opapi_abs_path.py](../../../csrc/cmake/scripts/examples/get_opapi_abs_path.py) | 算子工程 get_opapi_abs_path 的生成、配置、编译或打包辅助；接口见右栏。 | 注释：获取 opapi 二进制绝对路径；定义：`OpApiMgr` |
| [get_soc_info.py](../../../csrc/cmake/scripts/examples/get_soc_info.py) | 算子工程 get_soc_info 的生成、配置、编译或打包辅助；接口见右栏。 | 注释：获取 Soc 相关信息；定义：`SocInfoMgr` |

## `csrc/cmake/scripts/utest/`

utest 子模块；所属范围：算子工程的构建模块、脚本和外部依赖配置。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [gen_coverage.py](../../../csrc/cmake/scripts/utest/gen_coverage.py) | 算子工程 gen_coverage 的生成、配置、编译或打包辅助；接口见右栏。 | 注释：生成覆盖率；定义：`GenCoverage` |
| [gen_tiling_data_stub.py](../../../csrc/cmake/scripts/utest/gen_tiling_data_stub.py) | 算子工程 gen_tiling_data_stub 的生成、配置、编译或打包辅助；接口见右栏。 | 注释：生成 TilingData 桩；定义：`Process`, `check_if_new_tiling_file_path_existed`, `process_class_fields`, `find_classes` |

## `csrc/cmake/scripts/util/`

util 子模块；所属范围：算子工程的构建模块、脚本和外部依赖配置。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [ascendc_bin_param_build.py](../../../csrc/cmake/scripts/util/ascendc_bin_param_build.py) | 算子工程 ascendc_bin_param_build 的生成、配置、编译或打包辅助；接口见右栏。 | 定义：`ParamInfo`, `BinParamBuilder`, `get_tiling_keys`, `trans_soc_verion`, `parse_op_debug_confg` |
| [ascendc_gen_options.py](../../../csrc/cmake/scripts/util/ascendc_gen_options.py) | 算子工程 ascendc_gen_options 的生成、配置、编译或打包辅助；接口见右栏。 | 定义：`write_options_to_file`, `gen_compile_options` |
| [ascendc_impl_build.py](../../../csrc/cmake/scripts/util/ascendc_impl_build.py) | 算子工程 ascendc_impl_build 的生成、配置、编译或打包辅助；接口见右栏。 | 定义：`AdpBuilder`, `OpFileNotExistsError`, `optype_snake`, `optype_snake_ex`, `write_scripts` |
| [ascendc_ops_config.py](../../../csrc/cmake/scripts/util/ascendc_ops_config.py) | 算子工程 配置 的生成、配置、编译或打包辅助；接口见右栏。 | 定义：`OpConfig`, `load_json`, `get_specified_suffix_file`, `add_dict_key` |
| [const_var.py](../../../csrc/cmake/scripts/util/const_var.py) | 算子工程 const_var 的生成、配置、编译或打包辅助；接口见右栏。 | 定义：`conv_soc_ver` |
| [gen_version_info.sh](../../../csrc/cmake/scripts/util/gen_version_info.sh) | 算子工程 gen_version_info 的构建、打包或处理脚本；按命令内容确认执行效果。 | Shell 脚本；按工程阶段和文件名归类。 |
| [opdesc_parser.py](../../../csrc/cmake/scripts/util/opdesc_parser.py) | 算子工程 opdesc_parser 的生成、配置、编译或打包辅助；接口见右栏。 | 定义：`OpDesc`, `get_op_desc` |
| [parse_ini_to_json.py](../../../csrc/cmake/scripts/util/parse_ini_to_json.py) | 算子工程 parse_ini_to_json 的生成、配置、编译或打包辅助；接口见右栏。 | 定义：`parse_ini_files`, `check_file_size`, `parse_ini_to_obj` |

## `csrc/cmake/third_party/`

third_party 子模块；所属范围：算子工程的构建模块、脚本和外部依赖配置。

| 子目录 | 职责 |
| --- | --- |
| [build/](../../../csrc/cmake/third_party/build) | build 子模块；所属范围：算子工程的构建模块、脚本和外部依赖配置。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [.gitkeep](../../../csrc/cmake/third_party/.gitkeep) | 保留目录的占位文件，无运行逻辑。 | 依据文件格式及所在目录归类。 |
| [abseil-cpp.cmake](../../../csrc/cmake/third_party/abseil-cpp.cmake) | 第三方依赖的版本/引用或随附资料。 | CMake 构建规则。 |
| [ascend_protobuf.cmake](../../../csrc/cmake/third_party/ascend_protobuf.cmake) | 第三方依赖的版本/引用或随附资料。 | 构建项：`ascend_protobuf_build_transformer` |
| [gtest.cmake](../../../csrc/cmake/third_party/gtest.cmake) | 第三方依赖的版本/引用或随附资料。 | 构建项：`gtest`, `gmock`, `gtest_main` |
| [json.cmake](../../../csrc/cmake/third_party/json.cmake) | 第三方依赖的版本/引用或随附资料。 | 构建项：`json`, `json` |
| [makeself-fetch.cmake](../../../csrc/cmake/third_party/makeself-fetch.cmake) | 第三方依赖的版本/引用或随附资料。 | CMake 构建规则。 |
| [protobuf.cmake](../../../csrc/cmake/third_party/protobuf.cmake) | 第三方依赖的版本/引用或随附资料。 | CMake 构建规则。 |
| [secure_c.cmake](../../../csrc/cmake/third_party/secure_c.cmake) | 第三方依赖的版本/引用或随附资料。 | 构建项：`c_sec` |

## `csrc/cmake/third_party/build/`

build 子模块；所属范围：算子工程的构建模块、脚本和外部依赖配置。

| 子目录 | 职责 |
| --- | --- |
| [modules/](../../../csrc/cmake/third_party/build/modules) | modules 子模块；所属范围：算子工程的构建模块、脚本和外部依赖配置。 |

## `csrc/cmake/third_party/build/modules/`

modules 子模块；所属范围：算子工程的构建模块、脚本和外部依赖配置。

| 子目录 | 职责 |
| --- | --- |
| [patch/](../../../csrc/cmake/third_party/build/modules/patch) | patch 子模块；所属范围：算子工程的构建模块、脚本和外部依赖配置。 |

## `csrc/cmake/third_party/build/modules/patch/`

patch 子模块；所属范围：算子工程的构建模块、脚本和外部依赖配置。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [protobuf-hide_absl_symbols.patch](../../../csrc/cmake/third_party/build/modules/patch/protobuf-hide_absl_symbols.patch) | 第三方依赖的版本/引用或随附资料。 | 依据文件格式及所在目录归类。 |
| [protobuf_25.1_change_version.patch](../../../csrc/cmake/third_party/build/modules/patch/protobuf_25.1_change_version.patch) | 第三方依赖的版本/引用或随附资料。 | 依据文件格式及所在目录归类。 |
