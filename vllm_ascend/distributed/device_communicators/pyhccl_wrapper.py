#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import ctypes
import platform
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from torch.distributed import ReduceOp
from vllm.logger import logger

from vllm_ascend.utils import find_hccl_library

# export types and functions from hccl to Python ===
# for the original hccl definition, please check
# https://github.com/EternalLied/cann-hccl-new/blob/64ec6ce2923319caa5df8c3c531e06bdc148ce9c/inc/hccl/hccl.h#L90
# https://github.com/EternalLied/cann-hccl-new/blob/64ec6ce2923319caa5df8c3c531e06bdc148ce9c/inc/hccl/hccl_types.h#L48

aclrtContext_t = ctypes.c_void_p
hcclResult_t = ctypes.c_int
hcclComm_t = ctypes.c_void_p
hcclRootInfo_t = ctypes.c_void_p


class hcclUniqueId(ctypes.Structure):
    _fields_ = [("internal", ctypes.c_byte * 4108)]


aclrtStream_t = ctypes.c_void_p
buffer_type = ctypes.c_void_p

hcclDataType_t = ctypes.c_int


class hcclDataTypeEnum:
    hcclInt8 = 0
    hcclInt16 = 1
    hcclInt32 = 2
    hcclFloat16 = 3
    hcclFloat32 = 4
    hcclInt64 = 5
    hcclUint64 = 6
    hcclUint8 = 7
    hcclUint16 = 8
    hcclUint32 = 9
    hcclFloat64 = 10
    hcclBfloat16 = 11
    hcclInt128 = 12

    @classmethod
    def from_torch(cls, dtype: torch.dtype) -> int:
        if dtype == torch.int8:
            return cls.hcclInt8
        if dtype == torch.uint8:
            return cls.hcclUint8
        if dtype == torch.int32:
            return cls.hcclInt32
        if dtype == torch.int64:
            return cls.hcclInt64
        if dtype == torch.float16:
            return cls.hcclFloat16
        if dtype == torch.float32:
            return cls.hcclFloat32
        if dtype == torch.float64:
            return cls.hcclFloat64
        if dtype == torch.bfloat16:
            return cls.hcclBfloat16
        raise ValueError(f"Unsupported dtype: {dtype}")


hcclRedOp_t = ctypes.c_int


class hcclRedOpTypeEnum:
    hcclSum = 0
    hcclProd = 1
    hcclMax = 2
    hcclMin = 3

    @classmethod
    def from_torch(cls, op: ReduceOp) -> int:
        if op == ReduceOp.SUM:
            return cls.hcclSum
        if op == ReduceOp.PRODUCT:
            return cls.hcclProd
        if op == ReduceOp.MAX:
            return cls.hcclMax
        if op == ReduceOp.MIN:
            return cls.hcclMin
        raise ValueError(f"Unsupported op: {op}")


@dataclass
class Function:
    name: str
    restype: Any
    argtypes: List[Any]


class HCCLLibrary:
    exported_functions = [
        # const char* HcclGetErrorString(HcclResult code);
        Function("HcclGetErrorString", ctypes.c_char_p, [hcclResult_t]),

        # HcclResult HcclGetRootInfo(HcclRootInfo *rootInfo);
        Function("HcclGetRootInfo", hcclResult_t,
                 [ctypes.POINTER(hcclUniqueId)]),

        # HcclResult HcclCommInitRootInfo(
        #   uint32_t nRanks, const HcclRootInfo *rootInfo, uint32_t rank, HcclComm *comm);
        # note that HcclComm is a pointer type, so the last argument is a pointer to a pointer
        Function("HcclCommInitRootInfo", hcclResult_t, [
            ctypes.c_int,
            ctypes.POINTER(hcclUniqueId),
            ctypes.c_int,
            ctypes.POINTER(hcclComm_t),
        ]),

        # HcclResult HcclAllReduce(
        #   void *sendBuf, void *recvBuf, uint64_t count,
        #   HcclDataType dataType, HcclReduceOp op, HcclComm comm,
        #   aclrtStream stream);
        Function("HcclAllReduce", hcclResult_t, [
            buffer_type,
            buffer_type,
            ctypes.c_size_t,
            hcclDataType_t,
            hcclRedOp_t,
            hcclComm_t,
            aclrtStream_t,
        ]),

        # HcclResult HcclBroadcast(
        #   void *buf, uint64_t count,
        #   HcclDataType dataType, uint32_t root,
        #   HcclComm comm, aclrtStream stream);
        Function("HcclBroadcast", hcclResult_t, [
            buffer_type,
            ctypes.c_size_t,
            hcclDataType_t,
            ctypes.c_int,
            hcclComm_t,
            aclrtStream_t,
        ]),

        # HcclResult HcclCommDestroy(HcclComm comm);
        Function("HcclCommDestroy", hcclResult_t, [hcclComm_t]),

        # hcclResult_t  hcclSend(
        #   const void* sendbuff, size_t count, hcclDataType_t datatype,
        #   int dest, hcclComm_t comm, aclrtStream_t stream);
        Function("HcclSend", hcclResult_t, [
            buffer_type, ctypes.c_size_t, hcclDataType_t, ctypes.c_int,
            hcclComm_t, aclrtStream_t
        ]),

        # hcclResult_t  hcclRecv(
        #   void* recvbuff, size_t count, ncclDataType_t datatype,
        #   int src, ncclComm_t comm, cudaStream_t stream);
        Function("HcclRecv", hcclResult_t, [
            buffer_type, ctypes.c_size_t, hcclDataType_t, ctypes.c_int,
            hcclComm_t, aclrtStream_t
        ]),
    ]

    acl_exported_functions = [
        Function("aclrtGetCurrentContext", None,
                 [ctypes.POINTER(aclrtContext_t)]),
        Function("aclrtSetCurrentContext", None, [aclrtContext_t]),
    ]

    # class attribute to store the mapping from the path to the library
    # to avoid loading the same library multiple times
    path_to_library_cache: Dict[str, Any] = {}

    # class attribute to store the mapping from library path
    # to the correspongding directory
    path_to_dict_mapping: Dict[str, Dict[str, Any]] = {}

    def __init__(self, so_file: Optional[str] = None):

        so_file = so_file or find_hccl_library()

        try:
            if so_file not in HCCLLibrary.path_to_dict_mapping:
                lib = ctypes.CDLL(so_file)
                HCCLLibrary.path_to_library_cache[so_file] = lib
            self.lib = HCCLLibrary.path_to_library_cache[so_file]
        except Exception as e:
            logger.error(
                "Failed to load HCCL library from %s. "
                "It is expected if you are not running on Ascend NPUs."
                "Otherwise, the hccl library might not exist, be corrupted "
                "or it does not support the current platform %s. "
                "If you already have the library, please set the "
                "environment variable HCCL_SO_PATH"
                " to point to the correct hccl library path.", so_file,
                platform.platform())
            raise e

        if so_file not in HCCLLibrary.path_to_dict_mapping:
            _funcs: Dict[str, Any] = {}
            for func in HCCLLibrary.exported_functions:
                f = getattr(self.lib, func.name)
                f.restype = func.restype
                f.argtypes = func.argtypes
                _funcs[func.name] = f
            HCCLLibrary.path_to_dict_mapping[so_file] = _funcs
        self._funcs = HCCLLibrary.path_to_dict_mapping[so_file]

        cann_version = ""
        cann_version_file = "/usr/local/Ascend/ascend-toolkit/latest/aarch64-linux/include/version/cann_version.h"
        try:
            content = ""
            with open(cann_version_file, "r") as f:
                content = '\n'.join(f.readlines())
            if m := re.search(r'CANN_VERSION_STR "([^"]+)"', content):
                cann_version = m.group(1)
        except Exception as e:
            logger.info("cannot find cann version file %s: %s",
                        cann_version_file, e)

        acl_so = ""
        try:
            if cann_version == "8.1.RC1":
                acl_so = "libascendcl.so"
            else:
                acl_so = "libgert.so"
            aclLib = ctypes.CDLL(acl_so)
        except Exception as e:
            logger.error("Failed to load %s", acl_so)
            raise e
        for func in HCCLLibrary.acl_exported_functions:
            f = getattr(aclLib, func.name)
            f.argtypes = func.argtypes
            self._funcs[func.name] = f

    def aclrtGetCurrentContext(self) -> aclrtContext_t:
        context = aclrtContext_t()
        self._funcs["aclrtGetCurrentContext"](ctypes.byref(context))
        return context

    def aclrtSetCurrentContext(self, context: aclrtContext_t) -> None:
        self._funcs["aclrtSetCurrentContext"](context)

    def hcclGetErrorString(self, result: hcclResult_t) -> str:
        return self._funcs["HcclGetErrorString"](result).decode("utf-8")

    def HCCL_CHECK(self, result: hcclResult_t) -> None:
        if result != 0:
            error_str = self.hcclGetErrorString(result)
            raise RuntimeError(f"HCCL error: {error_str}")

    def hcclGetUniqueId(self) -> hcclUniqueId:
        unique_id = hcclUniqueId()
        self.HCCL_CHECK(self._funcs["HcclGetRootInfo"](
            ctypes.byref(unique_id)))
        return unique_id

    def unique_id_from_bytes(self, data: bytes) -> hcclUniqueId:
        if len(data) != 4108:
            raise ValueError(
                f"Expected 4108 bytes for hcclUniqueId, got {len(data)} bytes")
        unique_id = hcclUniqueId()
        ctypes.memmove(ctypes.addressof(unique_id.internal), data, 4108)
        return unique_id

    def hcclGetRootInfo(self) -> hcclUniqueId:
        rootInfo = hcclUniqueId()
        self.HCCL_CHECK(self._funcs["HcclGetRootInfo"](ctypes.byref(rootInfo)))
        return rootInfo

    def hcclCommInitRank(self, world_size: int, unique_id: hcclUniqueId,
                         rank: int) -> hcclComm_t:
        comm = hcclComm_t()
        self.HCCL_CHECK(self._funcs["HcclCommInitRootInfo"](
            world_size, ctypes.byref(unique_id), rank, ctypes.byref(comm)))
        return comm

    def hcclAllReduce(self, sendbuff: buffer_type, recvbuff: buffer_type,
                      count: int, datatype: int, op: int, comm: hcclComm_t,
                      stream: aclrtStream_t) -> None:
        # `datatype` actually should be `hcclDataType_t`
        # and `op` should be `hcclRedOp_t`
        # both are aliases of `ctypes.c_int`
        # when we pass int to a function, it will be converted to `ctypes.c_int`
        # by ctypes automatically
        self.HCCL_CHECK(self._funcs["HcclAllReduce"](sendbuff, recvbuff, count,
                                                     datatype, op, comm,
                                                     stream))

    def hcclBroadcast(self, buf: buffer_type, count: int, datatype: int,
                      root: int, comm: hcclComm_t,
                      stream: aclrtStream_t) -> None:
        self.HCCL_CHECK(self._funcs["HcclBroadcast"](buf, count, datatype,
                                                     root, comm, stream))

    def hcclCommDestroy(self, comm: hcclComm_t) -> None:
        self.HCCL_CHECK(self._funcs["HcclCommDestroy"](comm))

    def hcclSend(self, sendbuff: buffer_type, count: int, datatype: int,
                 dest: int, comm: hcclComm_t, stream: aclrtStream_t) -> None:
        self.HCCL_CHECK(self._funcs["HcclSend"](sendbuff, count, datatype,
                                                dest, comm, stream))

    def hcclRecv(self, recvbuff: buffer_type, count: int, datatype: int,
                 src: int, comm: hcclComm_t, stream: aclrtStream_t) -> None:
        self.HCCL_CHECK(self._funcs["HcclRecv"](recvbuff, count, datatype, src,
                                                comm, stream))


__all__ = [
    "HCCLLibrary",
    "hcclDataTypeEnum",
    "hcclRedOpTypeEnum",
    "hcclUniqueId",
    "hcclComm_t",
    "aclrtStream_t",
    "buffer_type",
]
