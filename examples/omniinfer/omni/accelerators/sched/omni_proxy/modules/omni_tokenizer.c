// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

#include "omni_tokenizer.h"
#include <Python.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static PyObject *pInitTokenizerFunc = NULL;
static PyObject *pBatchChatEncodeFunc = NULL;
static int is_initialized = 0;

int omni_tokenizer_init()
{
    if (is_initialized)
    {
        return 0;
    }

    Py_Initialize();
    if (!Py_IsInitialized())
    {
        return -1;
    }

    PyObject *sys_path = PySys_GetObject("path");
    PyObject *path = PyUnicode_FromString(".");
    PyList_Insert(sys_path, 0, path);
    Py_DECREF(path);

    PyObject *pModule = PyImport_ImportModule("omni_tokenizer");
    if (!pModule)
    {
        PyErr_Print();
        Py_Finalize();
        return -1;
    }

    pInitTokenizerFunc = PyObject_GetAttrString(pModule, "c_init_tokenizer");
    pBatchChatEncodeFunc = PyObject_GetAttrString(pModule, "c_batch_chat_encode_bytes");

    if (!pBatchChatEncodeFunc)
    {
        PyErr_Print();
        Py_XDECREF(pInitTokenizerFunc);
        Py_DECREF(pModule);
        Py_Finalize();
        return -1;
    }

    Py_DECREF(pModule);
    is_initialized = 1;
    return 0;
}

void omni_tokenizer_cleanup()
{
    Py_XDECREF(pInitTokenizerFunc);
    Py_XDECREF(pBatchChatEncodeFunc);
    pInitTokenizerFunc = NULL;
    pBatchChatEncodeFunc = NULL;

    if (Py_IsInitialized())
    {
        Py_Finalize();
    }

    is_initialized = 0;
}

int omni_init_tokenizer(const char *model_path)
{
    if (!is_initialized)
    {
        return -1;
    }

    PyObject *pArgs = PyTuple_New(1);
    PyObject *pModelPath = PyUnicode_FromString(model_path);
    PyTuple_SetItem(pArgs, 0, pModelPath);

    PyObject *pResult = PyObject_CallObject(pInitTokenizerFunc, pArgs);
    Py_DECREF(pArgs);

    if (!pResult)
    {
        PyErr_Print();
        return -1;
    }

    int result = PyLong_AsLong(pResult);
    Py_DECREF(pResult);

    return result;
}

int omni_batch_chat_encode(omni_tokenizer_request **requests, size_t num_reqs)
{
    if (!is_initialized)
    {
        return -1;
    }

    if (num_reqs == 0 || requests == NULL)
    {
        return -1;
    }

    for (size_t i = 0; i < num_reqs; i++)
    {
        if (requests[i]->input_data == NULL || requests[i]->input_len == 0 ||
            requests[i]->prompt == NULL || requests[i]->prompt_buf_size == 0 ||
            requests[i]->input_ids == NULL || requests[i]->input_ids_buf_size == 0 ||
            requests[i]->block_hashes == NULL || requests[i]->block_hashes_buf_size == 0)
        {
            return -1;
        }
    }

    PyObject *pTextsList = PyList_New(num_reqs);
    if (!pTextsList)
    {
        PyErr_Print();
        return -1;
    }

    for (size_t i = 0; i < num_reqs; i++)
    {
        PyObject *pTextBytes = PyBytes_FromStringAndSize(requests[i]->input_data, requests[i]->input_len);
        if (!pTextBytes)
        {
            Py_DECREF(pTextsList);
            PyErr_Print();
            return -1;
        }
        PyList_SetItem(pTextsList, i, pTextBytes);
    }

    PyObject *pArgs = PyTuple_New(1);
    PyTuple_SetItem(pArgs, 0, pTextsList);

    PyObject *pResult = PyObject_CallObject(pBatchChatEncodeFunc, pArgs);
    Py_DECREF(pArgs);

    if (!pResult)
    {
        PyErr_Print();
        return -1;
    }

    if (!PyTuple_Check(pResult) || PyTuple_Size(pResult) != 4)
    {
        Py_DECREF(pResult);
        return -1;
    }

    PyObject *pPrompts = PyTuple_GetItem(pResult, 0);
    PyObject *pInputIds = PyTuple_GetItem(pResult, 1);
    PyObject *pBlockIds = PyTuple_GetItem(pResult, 2);
    PyObject *pMultiModalSizes = PyTuple_GetItem(pResult, 3);

    if (!PyList_Check(pPrompts) || !PyList_Check(pInputIds) || !PyList_Check(pMultiModalSizes))
    {
        Py_DECREF(pResult);
        return -1;
    }

    size_t result_count = PyList_Size(pPrompts);
    if (result_count != num_reqs)
    {
        Py_DECREF(pResult);
        return -1;
    }

    for (size_t i = 0; i < result_count; i++)
    {
        PyObject *pPrompt = PyList_GetItem(pPrompts, i);
        if (PyBytes_Check(pPrompt))
        {
            char *prompt_data;
            Py_ssize_t prompt_size;
            if (PyBytes_AsStringAndSize(pPrompt, &prompt_data, &prompt_size) != -1)
            {
                if ((size_t)prompt_size < requests[i]->prompt_buf_size)
                {
                    memcpy(requests[i]->prompt, prompt_data, prompt_size);
                    requests[i]->prompt[prompt_size] = '\0';
                    requests[i]->prompt_len = prompt_size;
                }
            }
        }

        PyObject *pInputIdList = PyList_GetItem(pInputIds, i);
        if (PyList_Check(pInputIdList))
        {
            size_t id_count = PyList_Size(pInputIdList);
            if (id_count <= requests[i]->input_ids_buf_size)
            {
                for (size_t j = 0; j < id_count; j++)
                {
                    PyObject *pId = PyList_GetItem(pInputIdList, j);
                    if (PyLong_Check(pId))
                    {
                        requests[i]->input_ids[j] = PyLong_AsLong(pId);
                    }
                }
                requests[i]->input_ids_len = id_count;
            }
        }

        PyObject *pBlockIdList = PyList_GetItem(pBlockIds, i);
        if (PyList_Check(pBlockIdList))
        {
            size_t id_count = PyList_Size(pBlockIdList);
            if (id_count <= requests[i]->block_hashes_buf_size)
            {
                for (size_t j = 0; j < id_count; j++)
                {
                    PyObject *pId = PyList_GetItem(pBlockIdList, j);
                    if (PyLong_Check(pId))
                    {
                        requests[i]->block_hashes[j] = PyLong_AsLong(pId);
                    }
                }
                requests[i]->block_hashes_len = id_count;
            }
        }

        PyObject *pSize = PyList_GetItem(pMultiModalSizes, i);
        if (PyLong_Check(pSize))
        {
            requests[i]->multi_modal_size = PyLong_AsLong(pSize);
        }
    }

    Py_DECREF(pResult);
    return 0;
}

void print_tokenize_result(omni_tokenizer_request *requests)
{
    if (requests == NULL)
    {
        return;
    }

    printf("Prompt: %.*s\n", (int)requests->prompt_len, requests->prompt);
    printf("Input IDs (%zu): ", requests->input_ids_len);
    for (size_t i = 0; i < requests->input_ids_len; i++)
    {
        printf("%ld ", requests->input_ids[i]);
    }
    printf("\n");

    printf("Block Hashes (%zu): ", requests->block_hashes_len);
    for (size_t i = 0; i < requests->block_hashes_len; i++)
    {
        printf("%ld ", requests->block_hashes[i]);
    }
    printf("\n");

    printf("Multi Modal Size: %d\n", requests->multi_modal_size);
}