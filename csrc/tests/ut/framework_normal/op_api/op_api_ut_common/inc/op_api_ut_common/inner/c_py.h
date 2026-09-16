/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef _OP_API_UT_COMMON_C_PY_H_
#define _OP_API_UT_COMMON_C_PY_H_

#include <string>
#include <Python.h>

using namespace std;

using UtPyRunSimpleString = PyObject *(*)(const char *);
using UtPyInitialize = void (*)();
using UtPyIsInitialized = int (*)();
using UtPyFinalize = void (*)();
using UtPyImportImportModule = PyObject *(*)(const char *);
using UtPyObjectCallObject = PyObject *(*)(PyObject *, PyObject *);
using UtPyTupleNew = PyObject *(*)(Py_ssize_t);
using UtPyTupleSetItem = int (*)(PyObject *, Py_ssize_t, PyObject *);
using UtPyObjectGetAttrString = PyObject *(*)(PyObject *, const char *);
using UtPyErrPrint = void (*)();
using UtPyDealloc = void (*)(PyObject *);
using UtPyCallableCheck = int (*)(PyObject *);
using UtPyLongAsLong = int (*)(PyObject *);
using UtPyErrFetch = void (*)(PyObject **, PyObject **, PyObject **);
using UtPyErrNormalizeException = void (*)(PyObject **, PyObject **, PyObject **);
using UtPyBuildValue = PyObject *(*)(const char *, ...);
using UtPyGILStateCheck = int (*)();
using UtPyEvalSaveThread = PyThreadState *(*)();
using UtPyEvalRestoreThread = void (*)(PyThreadState *);
using UtPyDECREF = void (*)(PyObject *);
using UtPyImportCleanup = void (*)();

class PyHolder {
public:
    static PyHolder &GetInstance()
    {
        static PyHolder py_holder;
        return py_holder;
    }
    void Clean();
    int Initialize();
    PyObject *ImportModule(const string &moduleName) const;
    PyObject *ImportFunction(PyObject *module, const string &funcName) const;

private:
    PyHolder() = default;
    ~PyHolder();

    int LoadDynamicLib();
    int LoadFuncs();

public:
    UtPyRunSimpleString Ut_PyRun_SimpleString;
    UtPyInitialize Ut_Py_Initialize;
    UtPyIsInitialized Ut_Py_IsInitialized;
    UtPyFinalize Ut_Py_Finalize;
    UtPyImportImportModule Ut_PyImport_ImportModule;
    UtPyObjectCallObject Ut_PyObject_CallObject;
    UtPyTupleNew Ut_PyTuple_New;
    UtPyTupleSetItem Ut_PyTuple_SetItem;
    UtPyObjectGetAttrString Ut_PyObject_GetAttrString;
    UtPyErrPrint Ut_PyErr_Print;
    UtPyDealloc Ut_Py_Dealloc;
    UtPyCallableCheck Ut_PyCallable_Check;
    UtPyLongAsLong Ut_PyLong_AsLong;
    UtPyErrFetch Ut_PyErr_Fetch;
    UtPyErrNormalizeException Ut_PyErr_NormalizeException;
    UtPyBuildValue Ut_PyBuildValue;
    UtPyGILStateCheck Ut_PyGIL_StateCheck;
    UtPyEvalSaveThread Ut_PyEval_SaveThread;
    UtPyEvalRestoreThread Ut_PyEval_RestoreThread;
    UtPyDECREF Ut_Py_DECREF;
    UtPyImportCleanup Ut_PyImport_Cleanup;

private:
    void *py_hdl = nullptr;
    PyThreadState *PyThreadState_ = nullptr;
};

class PyScripts {
public:
    static PyScripts &GetInstance()
    {
        static PyScripts py_scripts;
        return py_scripts;
    }
    int Initialize(const string &full_path);
    int GenerateInput(const string &case_json) const;
    int GenerateGolden(const string &case_json, const string &py_script, const string &golden_func) const;
    int CompareGolden(const string &case_json) const;
    void Clean();

private:
    PyScripts() = default;
    ~PyScripts();
    int CallFunction(PyObject *pyFunc, PyObject *pyArgs) const;

private:
    PyObject *PyInputModule = nullptr;
    PyObject *PyGoldenModule = nullptr;
    PyObject *PyCompareModule = nullptr;
    PyObject *PyInputFunc = nullptr;
    PyObject *PyGoldenFunc = nullptr;
    PyObject *PyCompareFunc = nullptr;
};

#endif
