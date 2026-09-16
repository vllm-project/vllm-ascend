/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <dlfcn.h>
#include <iostream>
#include <memory>
#include <stdio.h>

#include "op_api_ut_common/inner/c_py.h"
#include "op_api_ut_common/inner/file_io.h"

using namespace std;


void PyHolder::Clean()
{
    if (Ut_Py_IsInitialized()) {
        // if (PyThreadState_) {
        //     Ut_PyEval_RestoreThread(PyThreadState_);
        // }
        Ut_Py_Finalize();
    }

    if (py_hdl != nullptr) {
        (void)dlclose(py_hdl);
    }
}

PyHolder::~PyHolder()
{
}

int PyHolder::Initialize()
{
    void *py_init_func = dlsym(RTLD_DEFAULT, "Py_Initialize");
    if (py_init_func == nullptr) {
        if (LoadDynamicLib() != 0) {
            return 1;
        }
    }

    if (LoadFuncs() != 0) {
        return 1;
    }

    Ut_Py_Initialize();

    // // if (Ut_PyGIL_StateCheck() != 0) {
    // //     PyThreadState_ = Ut_PyEval_SaveThread();
    // // }

    // Restore SIGINT handler to system default, avoid delayed quit in mixed python/c environment
    (void)Ut_PyRun_SimpleString("try:\n"
                                "  import signal\n"
                                "  signal.signal(signal.SIGINT, signal.SIG_DFL)\n"
                                "except:\n"
                                "  pass");
    return 0;
}

int PyHolder::LoadDynamicLib()
{
    constexpr uint32_t CMD_MAX_SIZE = 1024;
    char line[CMD_MAX_SIZE] = {0};
    int x, y, z;
    const int X_STD = 3;
    const int Y_STD = 7;
    string cmd = "python3 -V";
    string cmd_backup = "python -V";
    string cmd_backup2 = "python3.7 -V";
    string cmd_path = "python3-config --prefix";
    string PY = "Python";
    string pythonLib;

    FILE *fp = popen(cmd.c_str(), "r");
    if (fp == nullptr) {
        cout << "popen failed to run " << cmd << endl;
        return 1;
    }

    auto ret1 = fgets(line, CMD_MAX_SIZE, fp);
    pclose(fp);
    if (!ret1) {
        cout << "Failed to run cmd [" << cmd << "]. Try to run cmd [" << cmd_backup << "]." << endl;
        fp = popen(cmd_backup.c_str(), "r");
        if (fp == nullptr) {
            cout << "popen failed to run " << cmd_backup << endl;
            return 1;
        }
        ret1 = fgets(line, CMD_MAX_SIZE, fp);
        pclose(fp);
        if (!ret1) {
            cout << "Failed to run cmd [" << cmd_backup << "]. Try to run cmd [" << cmd_backup2 << "]." << endl;
            fp = popen(cmd_backup2.c_str(), "r");
            if (fp == nullptr) {
                cout << "popen failed to run " << cmd_backup2 << endl;
                return 1;
            }
            ret1 = fgets(line, CMD_MAX_SIZE, fp);
            pclose(fp);
            if (!ret1) {
                cout << "Failed to run cmd [" << cmd_backup2 << "]." << endl;
                return 1;
            }
        }
    }

    char buffer[CMD_MAX_SIZE] = {0};
    (void)sscanf(line, "%s %d.%d.%d", buffer, &x, &y, &z);
    if (strcmp(PY.c_str(), buffer) != 0 || x != X_STD || y < Y_STD) {
        cout << "Python version must be at least 3.7. But got " << line << endl;
        return 1;
    }

    string temp_X = to_string(x);
    string temp_Y = to_string(y);
    pythonLib = "libpython" + temp_X + "." + temp_Y + string(y > Y_STD ? "" : "m") + ".so.1.0";

    void *handle = dlopen(pythonLib.c_str(), RTLD_NOW | RTLD_GLOBAL);
    if (handle == nullptr) {
        char line_path[CMD_MAX_SIZE] = {0};
        string pythonLibUpdate;

        FILE *fptr = popen(cmd_path.c_str(), "r");
        if (fptr == nullptr) {
            cout << "popen failed to run " << cmd_path << endl;
            return 1;
        }

        auto ret2 = fgets(line_path, CMD_MAX_SIZE, fptr);
        pclose(fptr);
        if (ret2) {
            if (strlen(line_path) - 1 >= CMD_MAX_SIZE) {
                return 1;
            }
            string line_path_temp = line_path;
            cout << "Get python lib path [" << line_path_temp << "]." << endl;
            pythonLibUpdate = line_path_temp + "/lib/" + pythonLib;
            handle = dlopen(pythonLibUpdate.c_str(), RTLD_NOW | RTLD_GLOBAL);
            if (handle == nullptr) {
                cout << "Failed to load python library [" << pythonLib << "]. Please add its path to LD_LIBRARY_PATH."
                     << endl;
                return 1;
            }
            pythonLib = pythonLibUpdate;
        } else {
            cout << "Cannot fetch the return value from cmd [" << cmd_path << "]." << endl;
            return 1;
        }
    }

    cout << "Dynamic load python library [" << pythonLib << "] success." << endl;
    py_hdl = handle;
    return 0;
}

int PyHolder::LoadFuncs()
{
    Ut_PyRun_SimpleString = (UtPyRunSimpleString)dlsym(py_hdl, "PyRun_SimpleString");
    if (Ut_PyRun_SimpleString == nullptr) {
        cout << "Failed to dlsym function PyRun_SimpleString. Error: " << dlerror() << endl;
        return 1;
    }

    Ut_Py_Initialize = (UtPyInitialize)dlsym(py_hdl, "Py_Initialize");
    if (Ut_Py_Initialize == nullptr) {
        cout << "Failed to dlsym function Py_Initialize. Error: " << dlerror() << endl;
        return 1;
    }

    Ut_Py_IsInitialized = (UtPyIsInitialized)dlsym(py_hdl, "Py_IsInitialized");
    if (Ut_Py_IsInitialized == nullptr) {
        cout << "Failed to dlsym function Py_IsInitialized. Error: " << dlerror() << endl;
        return 1;
    }

    Ut_Py_Finalize = (UtPyFinalize)dlsym(py_hdl, "Py_Finalize");
    if (Ut_Py_Finalize == nullptr) {
        cout << "Failed to dlsym function Py_Finalize. Error: " << dlerror() << endl;
        return 1;
    }

    Ut_PyImport_ImportModule = (UtPyImportImportModule)dlsym(py_hdl, "PyImport_ImportModule");
    if (Ut_PyImport_ImportModule == nullptr) {
        cout << "Failed to dlsym function PyImport_ImportModule. Error: " << dlerror() << endl;
        return 1;
    }

    Ut_PyObject_CallObject = (UtPyObjectCallObject)dlsym(py_hdl, "PyObject_CallObject");
    if (Ut_PyObject_CallObject == nullptr) {
        cout << "Failed to dlsym function PyObject_CallObject. Error: " << dlerror() << endl;
        return 1;
    }

    Ut_PyTuple_New = (UtPyTupleNew)dlsym(py_hdl, "PyTuple_New");
    if (Ut_PyTuple_New == nullptr) {
        cout << "Failed to dlsym function PyTuple_New. Error: " << dlerror() << endl;
        return 1;
    }

    Ut_PyTuple_SetItem = (UtPyTupleSetItem)dlsym(py_hdl, "PyTuple_SetItem");
    if (Ut_PyTuple_SetItem == nullptr) {
        cout << "Failed to dlsym function PyTuple_SetItem. Error: " << dlerror() << endl;
        return 1;
    }

    Ut_PyObject_GetAttrString = (UtPyObjectGetAttrString)dlsym(py_hdl, "PyObject_GetAttrString");
    if (Ut_PyObject_GetAttrString == nullptr) {
        cout << "Failed to dlsym function PyObject_GetAttrString. Error: " << dlerror() << endl;
        return 1;
    }

    Ut_PyErr_Print = (UtPyErrPrint)dlsym(py_hdl, "PyErr_Print");
    if (Ut_PyErr_Print == nullptr) {
        cout << "Failed to dlsym function PyErr_Print. Error: " << dlerror() << endl;
        return 1;
    }

    Ut_Py_Dealloc = (UtPyDealloc)dlsym(py_hdl, "_Py_Dealloc");
    if (Ut_Py_Dealloc == nullptr) {
        cout << "Failed to dlsym function _Py_Dealloc. Error: " << dlerror() << endl;
        return 1;
    }

    Ut_PyCallable_Check = (UtPyCallableCheck)dlsym(py_hdl, "PyCallable_Check");
    if (Ut_PyCallable_Check == nullptr) {
        cout << "Failed to dlsym function PyCallable_Check. Error: " << dlerror() << endl;
        return 1;
    }

    Ut_PyLong_AsLong = (UtPyLongAsLong)dlsym(py_hdl, "PyLong_AsLong");
    if (Ut_PyLong_AsLong == nullptr) {
        cout << "Failed to dlsym function PyLong_AsLong. Error: " << dlerror() << endl;
        return 1;
    }

    Ut_PyErr_Fetch = (UtPyErrFetch)dlsym(py_hdl, "PyErr_Fetch");
    if (Ut_PyErr_Fetch == nullptr) {
        cout << "Failed to dlsym function PyErr_Fetch. Error: " << dlerror() << endl;
        return 1;
    }

    Ut_PyErr_NormalizeException = (UtPyErrNormalizeException)dlsym(py_hdl, "PyErr_NormalizeException");
    if (Ut_PyErr_NormalizeException == nullptr) {
        cout << "Failed to dlsym function PyErr_NormalizeException. Error: " << dlerror() << endl;
        return 1;
    }

    Ut_PyBuildValue = (UtPyBuildValue)dlsym(py_hdl, "Py_BuildValue");
    if (Ut_PyBuildValue == nullptr) {
        cout << "Failed to dlsym function Py_BuildValue. Error: " << dlerror() << endl;
        return 1;
    }

    Ut_PyGIL_StateCheck = (UtPyGILStateCheck)dlsym(py_hdl, "PyGILState_Check");
    if (Ut_PyGIL_StateCheck == nullptr) {
        cout << "Failed to dlsym function PyGILState_Check. Error: " << dlerror() << endl;
        return 1;
    }

    Ut_PyEval_SaveThread = (UtPyEvalSaveThread)dlsym(py_hdl, "PyEval_SaveThread");
    if (Ut_PyEval_SaveThread == nullptr) {
        cout << "Failed to dlsym function PyEval_SaveThread. Error: " << dlerror() << endl;
        return 1;
    }

    Ut_PyEval_RestoreThread = (UtPyEvalRestoreThread)dlsym(py_hdl, "PyEval_RestoreThread");
    if (Ut_PyEval_RestoreThread == nullptr) {
        cout << "Failed to dlsym function PyEval_RestoreThread. Error: " << dlerror() << endl;
        return 1;
    }

    Ut_Py_DECREF = (UtPyDECREF)dlsym(py_hdl, "Py_DecRef");
    if (Ut_Py_DECREF == nullptr) {
        cout << "Failed to dlsym function Py_DecRef. Error: " << dlerror() << endl;
        return 1;
    }

    Ut_PyImport_Cleanup = (UtPyImportCleanup)dlsym(py_hdl, "PyImport_Cleanup");
    if (Ut_PyImport_Cleanup == nullptr) {
        cout << "Failed to dlsym function PyImport_Cleanup. Error: " << dlerror() << endl;
        return 1;
    }


    return 0;
}

PyObject *PyHolder::ImportModule(const string &moduleName) const
{
    if (moduleName.empty()) {
        return nullptr;
    }

    PyObject *pyModule = Ut_PyImport_ImportModule(moduleName.c_str());
    if (pyModule == nullptr) {
        // python exception occur
        cout << "Import module [" << moduleName << "] failed." << endl;
        Ut_PyErr_Print();
        return nullptr;
    }
    return pyModule;
}

PyObject *PyHolder::ImportFunction(PyObject *module, const string &funcName) const
{
    if (module == nullptr || funcName.empty()) {
        return nullptr;
    }

    PyObject *pyFunc = Ut_PyObject_GetAttrString(module, funcName.c_str());
    if (pyFunc == nullptr || !Ut_PyCallable_Check(pyFunc)) {
        cout << "Failed to import function [" << funcName << endl;
        Ut_PyErr_Print();
        return nullptr;
    }
    return pyFunc;
}

static void PyObjectDecRef(PyObject *pyObj)
{
    PyObject *py_decref_tmp = reinterpret_cast<PyObject *>(pyObj);
    if (py_decref_tmp != nullptr) {
        PyHolder::GetInstance().Ut_Py_Dealloc(py_decref_tmp);
    }
}

int PyScripts::Initialize(const string &full_path)
{
    string full_path_ = RealPath(full_path);
    string sys_append_cmd = "sys.path.append('" + full_path_ + "') \n";
    string input_module = "generate_input";
    string golden_module = "generate_golden";
    string compare_module = "compare_golden";

    PyHolder::GetInstance().Ut_PyRun_SimpleString("import sys \n");
    PyHolder::GetInstance().Ut_PyRun_SimpleString(sys_append_cmd.c_str());

    PyInputModule = PyHolder::GetInstance().ImportModule(input_module.c_str());
    if (PyInputModule == nullptr) {
        return 1;
    }

    PyGoldenModule = PyHolder::GetInstance().ImportModule(golden_module.c_str());
    if (PyGoldenModule == nullptr) {
        return 1;
    }

    PyCompareModule = PyHolder::GetInstance().ImportModule(compare_module.c_str());
    if (PyCompareModule == nullptr) {
        return 1;
    }

    PyInputFunc = PyHolder::GetInstance().ImportFunction(PyInputModule, "main");
    if (PyInputFunc == nullptr) {
        return 1;
    }

    PyGoldenFunc = PyHolder::GetInstance().ImportFunction(PyGoldenModule, "main");
    if (PyGoldenFunc == nullptr) {
        return 1;
    }

    PyCompareFunc = PyHolder::GetInstance().ImportFunction(PyCompareModule, "main");
    if (PyCompareFunc == nullptr) {
        return 1;
    }
    return 0;
}

void PyScripts::Clean()
{
    if (PyInputFunc != nullptr) {
        PyHolder::GetInstance().Ut_Py_Dealloc(PyInputFunc);
    }
    if (PyGoldenFunc != nullptr) {
        PyHolder::GetInstance().Ut_Py_Dealloc(PyGoldenFunc);
    }
    if (PyCompareFunc != nullptr) {
        PyHolder::GetInstance().Ut_Py_Dealloc(PyCompareFunc);
    }
    if (PyInputModule != nullptr) {
        PyHolder::GetInstance().Ut_Py_Dealloc(PyInputModule);
    }
    if (PyGoldenModule != nullptr) {
        PyHolder::GetInstance().Ut_Py_Dealloc(PyGoldenModule);
    }
    if (PyCompareModule != nullptr) {
        PyHolder::GetInstance().Ut_Py_Dealloc(PyCompareModule);
    }
}

PyScripts::~PyScripts()
{
}

int PyScripts::CallFunction(PyObject *pyFunc, PyObject *pyArgs) const
{
    int ret;
    PyObject *pyRes = PyHolder::GetInstance().Ut_PyObject_CallObject(pyFunc, pyArgs);
    if (!pyRes) {
        PyHolder::GetInstance().Ut_PyErr_Print();
        ret = 1;
    } else {
        ret = PyHolder::GetInstance().Ut_PyLong_AsLong(pyRes);
    }
    return ret;
}

int PyScripts::GenerateInput(const string &case_json) const
{
    unique_ptr<PyObject, decltype(&PyObjectDecRef)> pArgs(PyHolder::GetInstance().Ut_PyTuple_New(1), PyObjectDecRef);
    PyHolder::GetInstance().Ut_PyTuple_SetItem(pArgs.get(), 0,
                                               PyHolder::GetInstance().Ut_PyBuildValue("s", case_json.c_str()));
    return CallFunction(PyInputFunc, pArgs.get());
}

int PyScripts::GenerateGolden(const string &case_json, const string &py_script, const string &golden_func) const
{
    unique_ptr<PyObject, decltype(&PyObjectDecRef)> pArgs(PyHolder::GetInstance().Ut_PyTuple_New(3), PyObjectDecRef);
    PyHolder::GetInstance().Ut_PyTuple_SetItem(pArgs.get(), 0,
                                               PyHolder::GetInstance().Ut_PyBuildValue("s", case_json.c_str()));
    PyHolder::GetInstance().Ut_PyTuple_SetItem(pArgs.get(), 1,
                                               PyHolder::GetInstance().Ut_PyBuildValue("s", py_script.c_str()));
    PyHolder::GetInstance().Ut_PyTuple_SetItem(pArgs.get(), 2,
                                               PyHolder::GetInstance().Ut_PyBuildValue("s", golden_func.c_str()));
    return CallFunction(PyGoldenFunc, pArgs.get());
}

int PyScripts::CompareGolden(const string &case_json) const
{
    unique_ptr<PyObject, decltype(&PyObjectDecRef)> pArgs(PyHolder::GetInstance().Ut_PyTuple_New(1), PyObjectDecRef);
    PyHolder::GetInstance().Ut_PyTuple_SetItem(pArgs.get(), 0,
                                               PyHolder::GetInstance().Ut_PyBuildValue("s", case_json.c_str()));
    return CallFunction(PyCompareFunc, pArgs.get());
}
