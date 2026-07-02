/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __INC_REGISTER_TUNING_BANK_KEY_REGISTRY_HEADER__
#define __INC_REGISTER_TUNING_BANK_KEY_REGISTRY_HEADER__
#include <memory>
#include <unordered_map>
#include <nlohmann/json.hpp>
#include <string>
#include "graph/ascend_string.h"
#include "register/register_types.h"
#include "exe_graph/runtime/tiling_context.h"

// v1 stub
#define REGISTER_OP_BANK_KEY_CONVERT_FUN(op, opfunc) REGISTER_OP_BANK_KEY_CONVERT_FUN_UNIQ_HELPER(op, (opfunc))

#define REGISTER_OP_BANK_KEY_CONVERT_FUN_UNIQ_HELPER(optype, opfunc) REGISTER_OP_BANK_KEY_UNIQ(optype, (opfunc))

#define REGISTER_OP_BANK_KEY_UNIQ(optype, opfunc) \
    static tuningtiling::OpBankKeyFuncRegistry g_##optype##BankKeyRegistryInterf(#optype, (opfunc))

#define REGISTER_OP_BANK_KEY_PARSE_FUN(op, parse_func, load_func) \
    REGISTER_OP_BANK_KEY_PARSE_FUN_UNIQ_HELPER(op, (parse_func), (load_func))

#define REGISTER_OP_BANK_KEY_PARSE_FUN_UNIQ_HELPER(optype, parse_func, load_func) \
    REGISTER_OP_BANK_KEY_PARSE_UNIQ(optype, (parse_func), (load_func))

#define REGISTER_OP_BANK_KEY_PARSE_UNIQ(optype, parse_func, load_func) \
    static tuningtiling::OpBankKeyFuncRegistry g_##optype##BankParseInterf(#optype, (parse_func), (load_func))

// v2
#define REGISTER_OP_BANK_KEY_CONVERT_FUN_V2(op, opfunc) REGISTER_OP_BANK_KEY_CONVERT_FUN_UNIQ_HELPER_V2(op, (opfunc))

#define REGISTER_OP_BANK_KEY_CONVERT_FUN_UNIQ_HELPER_V2(optype, opfunc) REGISTER_OP_BANK_KEY_UNIQ_V2(optype, (opfunc))

#define REGISTER_OP_BANK_KEY_UNIQ_V2(optype, opfunc) \
    static tuningtiling::OpBankKeyFuncRegistryV2 g_##optype##BankKeyRegistryInterf(#optype, (opfunc))

#define REGISTER_OP_BANK_KEY_PARSE_FUN_V2(op, parse_func, load_func) \
    REGISTER_OP_BANK_KEY_PARSE_FUN_UNIQ_HELPER_V2(op, (parse_func), (load_func))

#define REGISTER_OP_BANK_KEY_PARSE_FUN_UNIQ_HELPER_V2(optype, parse_func, load_func) \
    REGISTER_OP_BANK_KEY_PARSE_UNIQ_V2(optype, (parse_func), (load_func))

#define REGISTER_OP_BANK_KEY_PARSE_UNIQ_V2(optype, parse_func, load_func) \
    static tuningtiling::OpBankKeyFuncRegistryV2 g_##optype##BankParseInterf(#optype, (parse_func), (load_func))

#define TUNING_TILING_MAKE_SHARED(exec_expr0, exec_expr1) \
    do {                                                  \
        try {                                             \
            exec_expr0;                                   \
        } catch (...) {                                   \
            exec_expr1;                                   \
        }                                                 \
    } while (0)

// v1 stub
#define DECLARE_STRUCT_RELATE_WITH_OP(op, bank_key, ...)                                       \
    do {                                                                                       \
        NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(bank_key, __VA_ARGS__);                             \
        static bool ParseFunc##op##bank_key(                                                   \
            const std::shared_ptr<void>& in_args, size_t len, ge::AscendString& bank_key_str)  \
        {                                                                                      \
            if (sizeof(bank_key_str) != len || in_args == nullptr) {                           \
                return false;                                                                  \
            }                                                                                  \
            return false;                                                                      \
        }                                                                                      \
        static bool LoadFunc##op##bank_key(                                                    \
            std::shared_ptr<void>& in_args, size_t& len, const ge::AscendString& bank_key_str) \
        {                                                                                      \
            len = sizeof(bank_key_str);                                                        \
            TUNING_TILING_MAKE_SHARED(in_args = std::make_shared<bank_key>(), return false);   \
            auto op_ky = std::static_pointer_cast<bank_key>(in_args);                          \
            return false;                                                                      \
        }                                                                                      \
        REGISTER_OP_BANK_KEY_PARSE_FUN(op, ParseFunc##op##bank_key, LoadFunc##op##bank_key)    \
    } while (0)                                                                                 


// v2
#define DECLARE_STRUCT_RELATE_WITH_OP_V2(op, bank_key, ...)                                         \
    do {                                                                                            \
        NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(bank_key, __VA_ARGS__);                                  \
        static bool ParseFuncV2##op##bank_key(                                                      \
            const std::shared_ptr<void>& in_args, size_t len, ge::AscendString& bank_key_json_str)  \
        {                                                                                           \
            if (sizeof(bank_key) != len || in_args == nullptr) {                                    \
                return false;                                                                       \
            }                                                                                       \
            nlohmann::json bank_key_json;                                                           \
            bank_key_json = *(std::static_pointer_cast<bank_key>(in_args));                         \
            try {                                                                                   \
                std::string json_dump_str = bank_key_json.dump();                                   \
                bank_key_json_str = ge::AscendString(json_dump_str.c_str());                        \
            } catch (std::exception & e) {                                                          \
                return false;                                                                       \
            }                                                                                       \
            return true;                                                                            \
        }                                                                                           \
        static bool LoadFuncV2##op##bank_key(                                                       \
            std::shared_ptr<void>& in_args, size_t& len, const ge::AscendString& bank_key_json_str) \
        {                                                                                           \
            len = sizeof(bank_key);                                                                 \
            TUNING_TILING_MAKE_SHARED(in_args = std::make_shared<bank_key>(), return false);        \
            nlohmann::json bank_key_json;                                                           \
            try {                                                                                   \
                bank_key_json = nlohmann::json::parse(bank_key_json_str.GetString());               \
                auto op_ky = std::static_pointer_cast<bank_key>(in_args);                           \
                *op_ky = bank_key_json.get<bank_key>();                                             \
            } catch (std::exception & e) {                                                          \
                return false;                                                                       \
            }                                                                                       \
            return true;                                                                            \
        }                                                                                           \
        REGISTER_OP_BANK_KEY_PARSE_FUN_V2(op, ParseFuncV2##op##bank_key, LoadFuncV2##op##bank_key)  \
    } while (0)


namespace tuningtiling {
// v1兼容老版本om
using OpBankKeyConvertFun = std::function<bool(const gert::TilingContext*, std::shared_ptr<void>&, size_t&)>;
using OpBankParseFun = std::function<bool(const std::shared_ptr<void>&, size_t, ge::AscendString&)>;
using OpBankLoadFun = std::function<bool(std::shared_ptr<void>&, size_t&, const ge::AscendString&)>;

// v2
using OpBankKeyConvertFunV2 = std::function<bool(const gert::TilingContext*, std::shared_ptr<void>&, size_t&)>;
using OpBankParseFunV2 = std::function<bool(const std::shared_ptr<void>&, size_t, ge::AscendString&)>;
using OpBankLoadFunV2 = std::function<bool(std::shared_ptr<void>&, size_t&, const ge::AscendString&)>;
// v1兼容老版本om
class FMK_FUNC_HOST_VISIBILITY OpBankKeyFuncInfo
{
public:
    explicit OpBankKeyFuncInfo(const ge::AscendString& optype);
    OpBankKeyFuncInfo() = default;
    ~OpBankKeyFuncInfo() = default;
    void SetOpConvertFunc(const OpBankKeyConvertFun& convert_func);
    void SetOpParseFunc(const OpBankParseFun& parse_func);
    void SetOpLoadFunc(const OpBankLoadFun& load_func);
    const OpBankKeyConvertFun& GetBankKeyConvertFunc() const;
    const OpBankParseFun& GetBankKeyParseFunc() const;
    const OpBankLoadFun& GetBankKeyLoadFunc() const;
    const ge::AscendString& GetOpType() const
    {
        return optype_;
    }

private:
    ge::AscendString optype_;
    OpBankKeyConvertFun convert_func_;
    OpBankParseFun parse_func_;
    OpBankLoadFun load_func_;
};

// v2
class FMK_FUNC_HOST_VISIBILITY OpBankKeyFuncInfoV2
{
public:
    explicit OpBankKeyFuncInfoV2(const ge::AscendString& optypeV2);
    OpBankKeyFuncInfoV2() = default;
    ~OpBankKeyFuncInfoV2() = default;
    void SetOpConvertFuncV2(const OpBankKeyConvertFunV2& convert_funcV2);
    void SetOpParseFuncV2(const OpBankParseFunV2& parse_funcV2);
    void SetOpLoadFuncV2(const OpBankLoadFunV2& load_funcV2);
    const OpBankKeyConvertFunV2& GetBankKeyConvertFuncV2() const;
    const OpBankParseFunV2& GetBankKeyParseFuncV2() const;
    const OpBankLoadFunV2& GetBankKeyLoadFuncV2() const;
    const ge::AscendString& GetOpTypeV2() const
    {
        return optypeV2_;
    }

private:
    ge::AscendString optypeV2_;
    OpBankKeyConvertFunV2 convert_funcV2_;
    OpBankParseFunV2 parse_funcV2_;
    OpBankLoadFunV2 load_funcV2_;
};

// v1兼容老版本om
class FMK_FUNC_HOST_VISIBILITY OpBankKeyFuncRegistry
{
public:
    OpBankKeyFuncRegistry(const ge::AscendString& optype, const OpBankKeyConvertFun& convert_func);
    OpBankKeyFuncRegistry(
        const ge::AscendString& optype, const OpBankParseFun& parse_func, const OpBankLoadFun& load_func);
    ~OpBankKeyFuncRegistry() = default;
    static std::unordered_map<ge::AscendString, OpBankKeyFuncInfo>& RegisteredOpFuncInfo();
};

// v2
class FMK_FUNC_HOST_VISIBILITY OpBankKeyFuncRegistryV2
{
public:
    OpBankKeyFuncRegistryV2(const ge::AscendString& optype, const OpBankKeyConvertFunV2& convert_funcV2);
    OpBankKeyFuncRegistryV2(
        const ge::AscendString& optype, const OpBankParseFunV2& parse_funcV2, const OpBankLoadFunV2& load_funcV2);
    ~OpBankKeyFuncRegistryV2() = default;
    static std::unordered_map<ge::AscendString, OpBankKeyFuncInfoV2>& RegisteredOpFuncInfoV2();
};
} // namespace tuningtiling
#endif