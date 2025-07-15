import ctypes
import os
import tempfile
from unittest import mock
from unittest.mock import MagicMock, patch

import torch
from torch.distributed import ReduceOp

from tests.ut.base import TestBase
from vllm_ascend.distributed.device_communicators.pyhccl_wrapper import (
    Function, HCCLLibrary, aclrtStream_t, buffer_type, hcclComm_t,
    hcclDataType_t, hcclDataTypeEnum, hcclRedOp_t, hcclRedOpTypeEnum,
    hcclResult_t, hcclUniqueId)


class TestHcclUniqueId(TestBase):
    def test_construct(self):
        uid = hcclUniqueId()  # 构造
        uid.internal[0] = 0xAB
        self.assertEqual(len(uid.internal), 4108)
        self.assertEqual(uid.internal[0], 0xAB)


class TestHcclDataTypeEnum(TestBase):
    def test_torch_dtype_mapping(self):
        expected = {
            torch.int8: hcclDataTypeEnum.hcclInt8,
            torch.uint8: hcclDataTypeEnum.hcclUint8,
            torch.int32: hcclDataTypeEnum.hcclInt32,
            torch.int64: hcclDataTypeEnum.hcclInt64,
            torch.float16: hcclDataTypeEnum.hcclFloat16,
            torch.float32: hcclDataTypeEnum.hcclFloat32,
            torch.float64: hcclDataTypeEnum.hcclFloat64,
            torch.bfloat16: hcclDataTypeEnum.hcclBfloat16,
        }

        for torch_dtype, expected_enum in expected.items():
            with self.subTest(torch_dtype=torch_dtype):
                self.assertEqual(hcclDataTypeEnum.from_torch(torch_dtype),
                                 expected_enum)

    def test_unsupported_dtype_raises(self):
        with self.assertRaises(ValueError):
            hcclDataTypeEnum.from_torch(torch.complex64)  # 举例一个目前不支持的 dtype


class TestHcclRedOpTypeEnum(TestBase):
    def test_torch_reduce_op_mapping(self):
        expected = {
            ReduceOp.SUM: hcclRedOpTypeEnum.hcclSum,
            ReduceOp.PRODUCT: hcclRedOpTypeEnum.hcclProd,
            ReduceOp.MAX: hcclRedOpTypeEnum.hcclMax,
            ReduceOp.MIN: hcclRedOpTypeEnum.hcclMin,
        }

        for torch_op, expected_enum in expected.items():
            with self.subTest(torch_op=torch_op):
                self.assertEqual(hcclRedOpTypeEnum.from_torch(torch_op),
                                 expected_enum)

    def test_unsupported_op_raises(self):
        unsupported_op = "NOT_EXIST"
        with self.assertRaises(ValueError):
            hcclRedOpTypeEnum.from_torch(unsupported_op)


class TestFunction(TestBase):
    def test_construct_with_valid_args(self):
        func = Function(name="foo", restype=int, argtypes=[int, str, float])
        self.assertEqual(func.name, "foo")
        self.assertIs(func.restype, int)
        self.assertEqual(func.argtypes, [int, str, float])


class TestHCLLLibrary(TestBase):
    def test_init_with_nonexistent_so(self):
        fake_path = "/definitely/not/exist/libhccl.so"
        with self.assertRaises(OSError):
            HCCLLibrary(fake_path)

    def test_init_load_and_cache(self):
        with tempfile.NamedTemporaryFile(suffix=".so", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            lib1 = HCCLLibrary(tmp_path)
            with mock.patch("ctypes.CDLL") as mock_cdll:
                lib2 = HCCLLibrary(tmp_path)
                mock_cdll.assert_not_called()
            self.assertIs(lib1.lib, lib2.lib)
            self.assertIs(lib1._funcs, lib2._funcs)
        finally:
            os.unlink(tmp_path)

    def test_hccl_get_error_string(self):
        lib = MagicMock(sepc=HCCLLibrary)
        lib._funcs = {}
        mock_fn = MagicMock
        mock_fn.return_value = "HCCL internal error"
        lib._funcs["HcclGetErrorString"] = mock_fn

        result = hcclResult_t(1)
        msg = lib.hcclGetErrorString(result)
        self.assertEqual(msg, "HCCL internal error")
        mock_fn.assert_called_once()

    def test_hccl_check(self):
        lib = MagicMock(sepc=HCCLLibrary)
        lib.hcclGetErrorString.return_value = "fake error"
        result = hcclResult_t(123)
        with self.assertRaises(RuntimeError) as cm:
            lib.HCCL_CHECK(result)

        self.assertEqual(str(cm.exception), "HCCL error: fake error")

    @patch.object(HCCLLibrary, "HCCL_CHECK")
    def test_hccl_get_uniqueId(self, mock_HCCL_CHECK):
        lib = HCCLLibrary.__new__(HCCLLibrary)
        lib._funcs = {"HcclGetRootInfo": MagicMock(return_value=0)}
        unique_id = lib.hcclGetUniqueId()
        self.assertIsInstance(unique_id, hcclUniqueId)
        lib._funcs["HcclGetRootInfo"].assert_called_once()
        mock_HCCL_CHECK.assert_called_once_with(0)

    @patch.object(HCCLLibrary, "HCCL_CHECK")
    def test_hccl_comm_initRank(self, mock_hccl_check):
        """测试：HcclCommInitRootInfo 成功返回"""
        # 创建一个 HCCLLibrary 实例
        lib = HCCLLibrary.__new__(HCCLLibrary)
        lib._funcs = {"HcclCommInitRootInfo": MagicMock(return_value=0)}

        # 创建输入参数
        world_size = 4
        unique_id = hcclUniqueId()
        rank = 1

        # 调用 hcclCommInitRank
        comm = lib.hcclCommInitRank(world_size, unique_id, rank)

        # 验证
        self.assertIsInstance(comm, hcclComm_t)
        lib._funcs["HcclCommInitRootInfo"].assert_called_once_with(
            world_size, ctypes.byref(unique_id), rank, ctypes.byref(comm))
        mock_hccl_check.assert_called_once_with(0)

    def test_hccl_all_reduce(self, mock_hccl_check):
        """测试：hcclAllReduce 成功执行"""
        # 创建输入参数

        lib = HCCLLibrary.__new__(HCCLLibrary)
        lib._funcs = {"HcclAllReduce": MagicMock(return_value=0)}
        sendbuff = buffer_type()  # 假设 buffer_type 是 ctypes.c_void_p 或类似
        recvbuff = buffer_type()
        count = 10
        datatype = hcclDataType_t(1)  # 假设 hcclDataType_t 是 ctypes.c_int
        op = hcclRedOp_t(0)  # 假设 hcclRedOp_t 是 ctypes.c_int
        comm = hcclComm_t()
        stream = aclrtStream_t()

        # 调用 hcclAllReduce
        lib.hcclAllReduce(sendbuff, recvbuff, count, datatype, op, comm,
                          stream)

        # 验证
        lib._funcs["HcclAllReduce"].assert_called_once_with(
            sendbuff, recvbuff, count, datatype, op, comm, stream)
        mock_hccl_check.assert_called_once_with(0)

    def test_hccl_broad_cast(self, mock_hccl_check):
        """测试：hcclAllReduce 成功执行"""
        # 创建输入参数

        lib = HCCLLibrary.__new__(HCCLLibrary)
        lib._funcs = {"HcclAllReduce": MagicMock(return_value=0)}
        buff = buffer_type()  # 假设 buffer_type 是 ctypes.c_void_p 或类似
        count = 10
        datatype = 1  # 假设 hcclDataType_t 是 ctypes.c_int
        root = 0  # 假设 hcclRedOp_t 是 ctypes.c_int
        comm = hcclComm_t()
        stream = aclrtStream_t()

        # 调用 hcclAllReduce
        lib.hcclAllReduce(buff, count, datatype, root, comm, stream)

        # 验证
        lib._funcs["HcclBroadcast"].assert_called_once_with(
            buff, count, datatype, root, comm, stream)
        mock_hccl_check.assert_called_once_with(0)

    @patch.object(HCCLLibrary, "HCCL_CHECK")
    def test_hcclCommDestroy_success(self, mock_hccl_check):
        """测试：hcclCommDestroy 成功执行"""
        # 创建输入参数
        lib = HCCLLibrary.__new__(HCCLLibrary)
        lib._funcs = {"HcclCommDestroy": MagicMock(return_value=0)}
        comm = hcclComm_t()
        # 调用 hcclCommDestroy
        lib.hcclCommDestroy(comm)
        lib._funcs["HcclCommDestroy"].assert_called_once_with(comm)
        mock_hccl_check.assert_called_once_with(0)
