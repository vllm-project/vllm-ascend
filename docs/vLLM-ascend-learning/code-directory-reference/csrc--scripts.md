# csrc/scripts：目录与文件功能

[返回总索引](../code-directory-guide.md) · [学习文档入口](../README.md)

本分册覆盖 **69 个条目**。说明对应当前本地源码；右栏中的定义、注释或配置项用于追溯依据。

用途说明不等于运行验证；按文件角色归类的条目不保证已逐行确认全部算法。

## `csrc/scripts/`

算子代码生成、编译产物处理和安装打包脚本。

| 子目录 | 职责 |
| --- | --- |
| [opgen/](../../../csrc/scripts/opgen) | opgen 子模块；所属范围：算子代码生成、编译产物处理和安装打包脚本。 |
| [package/](../../../csrc/scripts/package) | 打包 子模块；所属范围：算子代码生成、编译产物处理和安装打包脚本。 |
| [util/](../../../csrc/scripts/util) | util 子模块；所属范围：算子代码生成、编译产物处理和安装打包脚本。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [build_cache.py](../../../csrc/scripts/build_cache.py) | 算子工程 缓存 的生成、配置、编译或打包辅助；接口见右栏。 | 定义：`_LockTimeoutError`, `snapshot_key`, `run`, `build_parser` |

## `csrc/scripts/opgen/`

opgen 子模块；所属范围：算子代码生成、编译产物处理和安装打包脚本。

| 子目录 | 职责 |
| --- | --- |
| [template/](../../../csrc/scripts/opgen/template) | template 子模块；所属范围：算子代码生成、编译产物处理和安装打包脚本。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [opgen_standalone.py](../../../csrc/scripts/opgen/opgen_standalone.py) | 算子工程 opgen_standalone 的生成、配置、编译或打包辅助；接口见右栏。 | 定义：`OpGenerator`, `execute`, `register_parser`, `main` |

## `csrc/scripts/opgen/template/`

template 子模块；所属范围：算子代码生成、编译产物处理和安装打包脚本。

| 子目录 | 职责 |
| --- | --- |
| [add/](../../../csrc/scripts/opgen/template/add) | add 子模块；所属范围：算子代码生成、编译产物处理和安装打包脚本。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [CMakeLists.txt](../../../csrc/scripts/opgen/template/CMakeLists.txt) | 定义本目录源码、依赖、目标和安装/构建步骤。 | CMake 构建规则。 |

## `csrc/scripts/opgen/template/add/`

add 子模块；所属范围：算子代码生成、编译产物处理和安装打包脚本。

| 子目录 | 职责 |
| --- | --- |
| [examples/](../../../csrc/scripts/opgen/template/add/examples) | examples 子模块；所属范围：算子代码生成、编译产物处理和安装打包脚本。 |
| [op_host/](../../../csrc/scripts/opgen/template/add/op_host) | 该算子的主机侧定义、shape/参数校验、tiling 分块和注册。 |
| [op_kernel/](../../../csrc/scripts/opgen/template/add/op_kernel) | 该算子在设备上的 kernel、分块计算、搬运和流水实现。 |
| [tests/](../../../csrc/scripts/opgen/template/add/tests) | tests 子模块；所属范围：算子代码生成、编译产物处理和安装打包脚本。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [CMakeLists.txt](../../../csrc/scripts/opgen/template/add/CMakeLists.txt) | 定义本目录源码、依赖、目标和安装/构建步骤。 | CMake 构建规则。 |

## `csrc/scripts/opgen/template/add/examples/`

examples 子模块；所属范围：算子代码生成、编译产物处理和安装打包脚本。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [test_aclnn_add_example.cpp](../../../csrc/scripts/opgen/template/add/examples/test_aclnn_add_example.cpp) | test_aclnn_add_example 的公共接口/实现，供所在算子工程复用。 | 函数：`GetShapeSize`, `PrintOutResult` |

## `csrc/scripts/opgen/template/add/op_host/`

该算子的主机侧定义、shape/参数校验、tiling 分块和注册。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [CMakeLists.txt](../../../csrc/scripts/opgen/template/add/op_host/CMakeLists.txt) | 定义本目录源码、依赖、目标和安装/构建步骤。 | CMake 构建规则。 |
| [add_example_def.cpp](../../../csrc/scripts/opgen/template/add/op_host/add_example_def.cpp) | opgen 的主机算子定义、支持属性和注册。 | 注册：`AddExample` |
| [add_example_infershape.cpp](../../../csrc/scripts/opgen/template/add/op_host/add_example_infershape.cpp) | opgen 的主机侧辅助接口或实现。 | 函数：`InferShapeAddExample` |
| [add_example_tiling.cpp](../../../csrc/scripts/opgen/template/add/op_host/add_example_tiling.cpp) | opgen 的主机侧 tiling 参数/分块策略计算实现。 | 类型：`AddExampleCompileInfo` |

## `csrc/scripts/opgen/template/add/op_kernel/`

该算子在设备上的 kernel、分块计算、搬运和流水实现。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [add_example.cpp](../../../csrc/scripts/opgen/template/add/op_kernel/add_example.cpp) | opgen 的设备执行源码。 | 类型：`AddExampleTilingKey` |
| [add_example.h](../../../csrc/scripts/opgen/template/add/op_kernel/add_example.h) | opgen 的类型/模板与设备实现头文件。 | 类型：`AddExample` |
| [add_example_tiling_data.h](../../../csrc/scripts/opgen/template/add/op_kernel/add_example_tiling_data.h) | opgen 的类型/模板与设备实现头文件。 | 类型：`AddExampleTilingData` |
| [add_example_tiling_key.h](../../../csrc/scripts/opgen/template/add/op_kernel/add_example_tiling_key.h) | opgen 的kernel 分派使用的 tiling key 定义。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |

## `csrc/scripts/opgen/template/add/tests/`

tests 子模块；所属范围：算子代码生成、编译产物处理和安装打包脚本。

| 子目录 | 职责 |
| --- | --- |
| [ut/](../../../csrc/scripts/opgen/template/add/tests/ut) | ut 子模块；所属范围：算子代码生成、编译产物处理和安装打包脚本。 |

## `csrc/scripts/opgen/template/add/tests/ut/`

ut 子模块；所属范围：算子代码生成、编译产物处理和安装打包脚本。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [.gitkeep](../../../csrc/scripts/opgen/template/add/tests/ut/.gitkeep) | 保留目录的占位文件，无运行逻辑。 | 依据文件格式及所在目录归类。 |

## `csrc/scripts/package/`

打包 子模块；所属范围：算子代码生成、编译产物处理和安装打包脚本。

| 子目录 | 职责 |
| --- | --- |
| [common/](../../../csrc/scripts/package/common) | 公共逻辑 子模块；所属范围：算子代码生成、编译产物处理和安装打包脚本。 |
| [latest_manager/](../../../csrc/scripts/package/latest_manager) | 管理器 子模块；所属范围：算子代码生成、编译产物处理和安装打包脚本。 |
| [module/](../../../csrc/scripts/package/module) | module 子模块；所属范围：算子代码生成、编译产物处理和安装打包脚本。 |
| [ops_transformer/](../../../csrc/scripts/package/ops_transformer) | ops_transformer 子模块；所属范围：算子代码生成、编译产物处理和安装打包脚本。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [package.py](../../../csrc/scripts/package/package.py) | 算子工程 打包 的生成、配置、编译或打包辅助；接口见右栏。 | 定义：`PackageOption`, `get_comments`, `get_compress_cmd`, `make_parse_option` |

## `csrc/scripts/package/common/`

公共逻辑 子模块；所属范围：算子代码生成、编译产物处理和安装打包脚本。

| 子目录 | 职责 |
| --- | --- |
| [cfg/](../../../csrc/scripts/package/common/cfg) | cfg 子模块；所属范围：算子代码生成、编译产物处理和安装打包脚本。 |
| [py/](../../../csrc/scripts/package/common/py) | py 子模块；所属范围：算子代码生成、编译产物处理和安装打包脚本。 |
| [sh/](../../../csrc/scripts/package/common/sh) | sh 子模块；所属范围：算子代码生成、编译产物处理和安装打包脚本。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../csrc/scripts/package/common/__init__.py) | 标记 Python 包边界；本文件没有顶层类或函数实现。 | 无顶层类/函数；包标记、常量或数据定义。 |

## `csrc/scripts/package/common/cfg/`

cfg 子模块；所属范围：算子代码生成、编译产物处理和安装打包脚本。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [path.cfg](../../../csrc/scripts/package/common/cfg/path.cfg) | 所在模块的结构化配置或参数模板。 | 配置条目与规则；按实际读取者生效。 |

## `csrc/scripts/package/common/py/`

py 子模块；所属范围：算子代码生成、编译产物处理和安装打包脚本。

| 子目录 | 职责 |
| --- | --- |
| [utils/](../../../csrc/scripts/package/common/py/utils) | 辅助函数 子模块；所属范围：算子代码生成、编译产物处理和安装打包脚本。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../csrc/scripts/package/common/py/__init__.py) | 标记 Python 包边界；本文件没有顶层类或函数实现。 | 无顶层类/函数；包标记、常量或数据定义。 |
| [filelist.py](../../../csrc/scripts/package/common/py/filelist.py) | 算子工程 filelist 的生成、配置、编译或打包辅助；接口见右栏。 | 注释：filelist相关类。；定义：`FileItem`, `FileItemRelation`, `create_file_item`, `soft_links_to_string`, `file_item_to_string` |
| [merge_binary_info_config.py](../../../csrc/scripts/package/common/py/merge_binary_info_config.py) | 算子工程 配置、合并 的生成、配置、编译或打包辅助；接口见右栏。 | 注释：合并算子binary_info_config.json。；定义：`load_json_file`, `save_json_file`, `update_config` |
| [packer.py](../../../csrc/scripts/package/common/py/packer.py) | 算子工程 打包 的生成、配置、编译或打包辅助；接口见右栏。 | 定义：`PackageName`, `MakeselfPkgParams`, `remove_ascend`, `get_func_name`, `get_compress_tool` |
| [pkg_parser.py](../../../csrc/scripts/package/common/py/pkg_parser.py) | 算子工程 pkg_parser 的生成、配置、编译或打包辅助；接口见右栏。 | 定义：`ParseOption`, `ParseEnv`, `parse_os_arch`, `replace_env`, `parse_package_info` |
| [version_info.py](../../../csrc/scripts/package/common/py/version_info.py) | 算子工程 version_info 的生成、配置、编译或打包辅助；接口见右栏。 | 定义：`VersionInfoError`, `VersionFormatNotMatch`, `is_version_number`, `get_version_dir`, `is_multi_version` |

## `csrc/scripts/package/common/py/utils/`

辅助函数 子模块；所属范围：算子代码生成、编译产物处理和安装打包脚本。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [comm_log.py](../../../csrc/scripts/package/common/py/utils/comm_log.py) | 算子工程 日志 的生成、配置、编译或打包辅助；接口见右栏。 | 定义：`CommLog` |
| [funcbase.py](../../../csrc/scripts/package/common/py/utils/funcbase.py) | 算子工程 funcbase 的生成、配置、编译或打包辅助；接口见右栏。 | 注释：函数基础库。；定义：`constant`, `dispatch`, `pipe` |
| [pkg_utils.py](../../../csrc/scripts/package/common/py/utils/pkg_utils.py) | 算子工程 辅助函数 的生成、配置、编译或打包辅助；接口见右栏。 | 注释：基础构件。；定义：`PackageError`, `PackageConfigError`, `flatten`, `merge_dict`, `star_pipe` |

## `csrc/scripts/package/common/sh/`

sh 子模块；所属范围：算子代码生成、编译产物处理和安装打包脚本。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [check_version_required.awk](../../../csrc/scripts/package/common/sh/check_version_required.awk) | 检查 附属文件；当前按文件格式及目录定位用途。 | 依据文件格式及所在目录归类。 |
| [cleanup.sh](../../../csrc/scripts/package/common/sh/cleanup.sh) | 算子工程 cleanup 的构建、打包或处理脚本；按命令内容确认执行效果。 | Shell 脚本；按工程阶段和文件名归类。 |
| [common_func.inc](../../../csrc/scripts/package/common/sh/common_func.inc) | 公共逻辑 附属文件；当前按文件格式及目录定位用途。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |
| [common_func_v2.inc](../../../csrc/scripts/package/common/sh/common_func_v2.inc) | 公共逻辑 附属文件；当前按文件格式及目录定位用途。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |
| [common_installer.inc](../../../csrc/scripts/package/common/sh/common_installer.inc) | 公共逻辑 附属文件；当前按文件格式及目录定位用途。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |
| [common_interface.csh](../../../csrc/scripts/package/common/sh/common_interface.csh) | 公共逻辑 附属文件；当前按文件格式及目录定位用途。 | 依据文件格式及所在目录归类。 |
| [common_interface.fish](../../../csrc/scripts/package/common/sh/common_interface.fish) | 公共逻辑 附属文件；当前按文件格式及目录定位用途。 | 依据文件格式及所在目录归类。 |
| [common_interface.sh](../../../csrc/scripts/package/common/sh/common_interface.sh) | 算子工程 公共逻辑 的构建、打包或处理脚本；按命令内容确认执行效果。 | 函数：`mk_custom_path`, `py_version_check` |
| [install_common_parser.sh](../../../csrc/scripts/package/common/sh/install_common_parser.sh) | 算子工程 公共逻辑 的构建、打包或处理脚本；按命令内容确认执行效果。 | 函数：`__pkg_in_pkgs`, `__block_in_blocks`, `__remove_pkg_in_pkgs` |
| [multi_version.inc](../../../csrc/scripts/package/common/sh/multi_version.inc) | multi_version.inc 附属文件；当前按文件格式及目录定位用途。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |
| [script_operator.inc](../../../csrc/scripts/package/common/sh/script_operator.inc) | script_operator.inc 附属文件；当前按文件格式及目录定位用途。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |
| [version_cfg.inc](../../../csrc/scripts/package/common/sh/version_cfg.inc) | version_cfg.inc 附属文件；当前按文件格式及目录定位用途。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |
| [version_compatiable.inc](../../../csrc/scripts/package/common/sh/version_compatiable.inc) | version_compatiable.inc 附属文件；当前按文件格式及目录定位用途。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |

## `csrc/scripts/package/latest_manager/`

管理器 子模块；所属范围：算子代码生成、编译产物处理和安装打包脚本。

| 子目录 | 职责 |
| --- | --- |
| [scripts/](../../../csrc/scripts/package/latest_manager/scripts) | scripts 子模块；所属范围：算子代码生成、编译产物处理和安装打包脚本。 |

## `csrc/scripts/package/latest_manager/scripts/`

scripts 子模块；所属范围：算子代码生成、编译产物处理和安装打包脚本。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [filelist.csv](../../../csrc/scripts/package/latest_manager/scripts/filelist.csv) | filelist.csv 附属文件；当前按文件格式及目录定位用途。 | 依据文件格式及所在目录归类。 |
| [install.sh](../../../csrc/scripts/package/latest_manager/scripts/install.sh) | 算子工程 install 的构建、打包或处理脚本；按命令内容确认执行效果。 | Shell 脚本；按工程阶段和文件名归类。 |
| [manager.sh](../../../csrc/scripts/package/latest_manager/scripts/manager.sh) | 算子工程 管理器 的构建、打包或处理脚本；按命令内容确认执行效果。 | Shell 脚本；按工程阶段和文件名归类。 |
| [manager_func.sh](../../../csrc/scripts/package/latest_manager/scripts/manager_func.sh) | 算子工程 管理器 的构建、打包或处理脚本；按命令内容确认执行效果。 | 函数：`get_manager_refs`, `manager_refs_exists`, `remove_manager_refs` |
| [uninstall.sh](../../../csrc/scripts/package/latest_manager/scripts/uninstall.sh) | 算子工程 uninstall 的构建、打包或处理脚本；按命令内容确认执行效果。 | Shell 脚本；按工程阶段和文件名归类。 |
| [version.info](../../../csrc/scripts/package/latest_manager/scripts/version.info) | version.info 附属文件；当前按文件格式及目录定位用途。 | 依据文件格式及所在目录归类。 |

## `csrc/scripts/package/module/`

module 子模块；所属范围：算子代码生成、编译产物处理和安装打包脚本。

| 子目录 | 职责 |
| --- | --- |
| [ascend/](../../../csrc/scripts/package/module/ascend) | 面向 ascend 的算子型号/架构配置或构建选择。 |

## `csrc/scripts/package/module/ascend/`

面向 ascend 的算子型号/架构配置或构建选择。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [EngineeringCommon.xml](../../../csrc/scripts/package/module/ascend/EngineeringCommon.xml) | 公共逻辑 附属文件；当前按文件格式及目录定位用途。 | 配置条目与规则；按实际读取者生效。 |
| [EngineeringFiles.xml](../../../csrc/scripts/package/module/ascend/EngineeringFiles.xml) | EngineeringFiles.xml 附属文件；当前按文件格式及目录定位用途。 | 配置条目与规则；按实际读取者生效。 |
| [OpsTransformer.xml](../../../csrc/scripts/package/module/ascend/OpsTransformer.xml) | OpsTransformer.xml 附属文件；当前按文件格式及目录定位用途。 | 配置条目与规则；按实际读取者生效。 |
| [OpsTransformerInc.xml](../../../csrc/scripts/package/module/ascend/OpsTransformerInc.xml) | OpsTransformerInc.xml 附属文件；当前按文件格式及目录定位用途。 | 配置条目与规则；按实际读取者生效。 |

## `csrc/scripts/package/ops_transformer/`

ops_transformer 子模块；所属范围：算子代码生成、编译产物处理和安装打包脚本。

| 子目录 | 职责 |
| --- | --- |
| [scripts/](../../../csrc/scripts/package/ops_transformer/scripts) | scripts 子模块；所属范围：算子代码生成、编译产物处理和安装打包脚本。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [ops_transformer.xml](../../../csrc/scripts/package/ops_transformer/ops_transformer.xml) | ops_transformer.xml 附属文件；当前按文件格式及目录定位用途。 | 配置条目与规则；按实际读取者生效。 |

## `csrc/scripts/package/ops_transformer/scripts/`

scripts 子模块；所属范围：算子代码生成、编译产物处理和安装打包脚本。

| 子目录 | 职责 |
| --- | --- |
| [empty_package_scripts/](../../../csrc/scripts/package/ops_transformer/scripts/empty_package_scripts) | 打包 子模块；所属范围：算子代码生成、编译产物处理和安装打包脚本。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [cleanup.sh](../../../csrc/scripts/package/ops_transformer/scripts/cleanup.sh) | 算子工程 cleanup 的构建、打包或处理脚本；按命令内容确认执行效果。 | Shell 脚本；按工程阶段和文件名归类。 |
| [help.info](../../../csrc/scripts/package/ops_transformer/scripts/help.info) | help.info 附属文件；当前按文件格式及目录定位用途。 | 依据文件格式及所在目录归类。 |
| [install.sh](../../../csrc/scripts/package/ops_transformer/scripts/install.sh) | 算子工程 install 的构建、打包或处理脚本；按命令内容确认执行效果。 | 函数：`startlog`, `exitlog`, `get_installed_info` |
| [opp_common.sh](../../../csrc/scripts/package/ops_transformer/scripts/opp_common.sh) | 算子工程 公共逻辑 的构建、打包或处理脚本；按命令内容确认执行效果。 | 函数：`getdate`, `logandprint`, `createrelativelysoftlink` |
| [opp_custom_install.sh](../../../csrc/scripts/package/ops_transformer/scripts/opp_custom_install.sh) | 算子工程 opp_custom_install 的构建、打包或处理脚本；按命令内容确认执行效果。 | Shell 脚本；按工程阶段和文件名归类。 |
| [opp_custom_uninstall.sh](../../../csrc/scripts/package/ops_transformer/scripts/opp_custom_uninstall.sh) | 算子工程 opp_custom_uninstall 的构建、打包或处理脚本；按命令内容确认执行效果。 | Shell 脚本；按工程阶段和文件名归类。 |
| [opp_install.sh](../../../csrc/scripts/package/ops_transformer/scripts/opp_install.sh) | 算子工程 opp_install 的构建、打包或处理脚本；按命令内容确认执行效果。 | 函数：`get_opts`, `init_install_env`, `log_with_errorlevel` |
| [opp_uninstall.sh](../../../csrc/scripts/package/ops_transformer/scripts/opp_uninstall.sh) | 算子工程 opp_uninstall 的构建、打包或处理脚本；按命令内容确认执行效果。 | 函数：`get_opts`, `get_docker_install_path`, `log_with_errorlevel` |
| [uninstall.sh](../../../csrc/scripts/package/ops_transformer/scripts/uninstall.sh) | 算子工程 uninstall 的构建、打包或处理脚本；按命令内容确认执行效果。 | 函数：`getdate`, `logandprint` |
| [ver_check.sh](../../../csrc/scripts/package/ops_transformer/scripts/ver_check.sh) | 算子工程 检查 的构建、打包或处理脚本；按命令内容确认执行效果。 | 函数：`getdate`, `logandprint`, `check_path_pre` |

## `csrc/scripts/package/ops_transformer/scripts/empty_package_scripts/`

打包 子模块；所属范围：算子代码生成、编译产物处理和安装打包脚本。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [cleanup.sh](../../../csrc/scripts/package/ops_transformer/scripts/empty_package_scripts/cleanup.sh) | 算子工程 cleanup 的构建、打包或处理脚本；按命令内容确认执行效果。 | Shell 脚本；按工程阶段和文件名归类。 |
| [install.sh](../../../csrc/scripts/package/ops_transformer/scripts/empty_package_scripts/install.sh) | 算子工程 install 的构建、打包或处理脚本；按命令内容确认执行效果。 | 函数：`getdate`, `logandprint` |

## `csrc/scripts/util/`

util 子模块；所属范围：算子代码生成、编译产物处理和安装打包脚本。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../csrc/scripts/util/__init__.py) | Python 包入口，导入或重新导出本模块接口；需检查导入副作用。 | 导入：`os`, `sys` |
| [build_opp_kernel_static.py](../../../csrc/scripts/util/build_opp_kernel_static.py) | 算子工程 build_opp_kernel_static 的生成、配置、编译或打包辅助；接口见右栏。 | 注释：build_opp_kernel_static.py；定义：`Const`, `CompileOpStaticLib`, `shell_exec`, `shell_checkout_key_func`, `to_upper_camel_case` |
| [const_var.py](../../../csrc/scripts/util/const_var.py) | 算子工程 const_var 的生成、配置、编译或打包辅助；接口见右栏。 | 注释：Function:；定义：`conv_soc_ver` |
| [insert_op_info.py](../../../csrc/scripts/util/insert_op_info.py) | 算子工程 insert_op_info 的生成、配置、编译或打包辅助；接口见右栏。 | 导入：`json`, `os` |
| [merge_aicpu_info_json.sh](../../../csrc/scripts/util/merge_aicpu_info_json.sh) | 算子工程 合并 的构建、打包或处理脚本；按命令内容确认执行效果。 | 命令类别：`python3` |
| [merge_proto.py](../../../csrc/scripts/util/merge_proto.py) | 算子工程 合并 的生成、配置、编译或打包辅助；接口见右栏。 | 定义：`match_op_proto`, `merge_op_proto`, `parse_args` |
| [modify_gen_aclnn.py](../../../csrc/scripts/util/modify_gen_aclnn.py) | 算子工程 modify_gen_aclnn 的生成、配置、编译或打包辅助；接口见右栏。 | 定义：`shell_exec`, `search_file`, `modify_gen_aclnn` |
