# SPDX-License-Identifier: Apache-2.0
"""SDK transport for one mapped FSA planner payload shared by a local TP group."""

import ctypes as ct

import torch.distributed as dist

# The A3 STARS Notify task takes seconds, as verified by the device timeout log.
_A3_NOTIFY_TIMEOUT_SECONDS = 60
_ACL_STREAM_ATTR_FAILURE_MODE = 1
_ACL_STOP_ON_FAILURE = 1


class _StreamAttribute(ct.Union):
    # Match aclrtStreamAttrValue, including its reserved tail.
    _fields_ = [("failure_mode", ct.c_uint64), ("reserved", ct.c_uint32 * 4)]


class FsaPlanIpc:
    def __init__(self, group, allocate_owner, plan_stream):
        self._failed = False
        self.rank = dist.get_rank(group)
        self.size = dist.get_world_size(group)
        self.owner_payload = None
        self.owned = {}
        self.imported = {}

        def exchange(operation):
            try:
                local = (operation(), None)
            except Exception as exc:
                local = (None, f"{type(exc).__name__}: {exc}")
            results = [None] * self.size
            dist.all_gather_object(results, local, group=group)
            failed = [f"TP{rank}: {failure}" for rank, (_, failure) in enumerate(results) if failure]
            if failed:
                raise RuntimeError("FSA planner IPC setup: " + "; ".join(failed))
            return [value for value, _ in results]

        def prepare():
            lib = ct.CDLL("libascendcl.so")

            def bind(name, *types):
                f = getattr(lib, name)
                f.argtypes = list(types)
                f.restype = ct.c_int
                return f

            p = ct.c_void_p
            self._create = bind("aclrtCreateNotify", ct.POINTER(p), ct.c_uint64)
            self._destroy = bind("aclrtDestroyNotify", p)
            self._export = bind("aclrtNotifyGetExportKey", p, ct.POINTER(ct.c_char), ct.c_size_t, ct.c_uint64)
            self._permit = bind("aclrtNotifySetImportPid", p, ct.POINTER(ct.c_int32), ct.c_size_t)
            self._import = bind("aclrtNotifyImportByKey", ct.POINTER(p), ct.c_char_p, ct.c_uint64)
            self._record = bind("aclrtRecordNotify", p, p)
            self._wait = bind("aclrtWaitAndResetNotify", p, p, ct.c_uint32)
            get_attribute = bind("aclrtGetStreamAttribute", p, ct.c_int, ct.POINTER(_StreamAttribute))
            attribute = _StreamAttribute()
            self._check(
                get_attribute(plan_stream, _ACL_STREAM_ATTR_FAILURE_MODE, ct.byref(attribute)),
                "read planning stream failure mode",
            )
            if attribute.failure_mode != _ACL_STOP_ON_FAILURE:
                raise RuntimeError("FSA planner IPC requires ACL_STOP_ON_FAILURE on the planning stream")

            get_pid = bind("aclrtDeviceGetBareTgid", ct.POINTER(ct.c_int32))
            pid = ct.c_int32()
            self._check(get_pid(ct.byref(pid)), "read bare TGID")
            if pid.value <= 0:
                raise RuntimeError("invalid bare TGID")
            return pid.value

        pids = exchange(prepare)
        if len(set(pids)) != len(pids):
            raise RuntimeError("FSA planner IPC needs one process per TP rank")
        peers = range(1, self.size) if self.rank == 0 else (0,)

        def export():
            if self.rank == 0:
                (self.owner_payload,) = allocate_owner()
            keys = {}
            for peer in peers:
                handle = ct.c_void_p()
                self._check(self._create(ct.byref(handle), 0), "create owner Notify")
                self.owned[peer] = handle
                key = ct.create_string_buffer(65)
                # Default export mode enables the TGID whitelist.
                self._check(self._export(handle, key, len(key), 0), "export owner Notify")
                pid = ct.c_int32(pids[peer])
                self._check(self._permit(handle, ct.byref(pid), 1), "set peer TGID")
                keys[peer] = key.value
            return keys, self.owner_payload.data_ptr() if self.rank == 0 else None

        try:
            exports = exchange(export)
            self.payload_ptr = exports[0][1]
            if not self.payload_ptr or self.payload_ptr % 32:
                raise RuntimeError("FSA planner IPC payload is not DMA-aligned")

            def import_peers():
                for peer in peers:
                    handle = ct.c_void_p()
                    # 0x02 requests peer access when importing a Notify.
                    self._check(self._import(ct.byref(handle), exports[peer][0][self.rank], 2), "import peer Notify")
                    self.imported[peer] = handle

            exchange(import_peers)
        except Exception as exc:
            try:
                self.close()
            except Exception as cleanup_error:
                raise exc from cleanup_error
            raise

    @staticmethod
    def _check(status, action):
        if status:
            raise RuntimeError(f"FSA planner IPC {action} failed with ACL status {status}")

    def ensure_usable(self):
        if self._failed:
            raise RuntimeError("FSA planner IPC cannot be used after a submission failure or shutdown")

    def mark_failed(self):
        self._failed = True

    def _submit(self, operation, action, *args):
        try:
            self._check(operation(*args), action)
        except Exception:
            self.mark_failed()
            raise

    def before_publish(self, stream):
        self.ensure_usable()
        if self.rank == 0:
            for handle in self.imported.values():
                self._submit(self._record, "submit READY", handle, stream)
        else:
            self._submit(self._wait, "submit READY wait", self.owned[0], stream, _A3_NOTIFY_TIMEOUT_SECONDS)

    def after_publish(self, stream):
        self.ensure_usable()
        if self.rank == 0:
            for handle in self.owned.values():
                self._submit(self._wait, "submit ACK wait", handle, stream, _A3_NOTIFY_TIMEOUT_SECONDS)
        else:
            self._submit(self._record, "submit ACK", self.imported[0], stream)

    def close(self):
        """Release after all local tasks using the handles have completed."""
        self.mark_failed()
        failures = []
        for role, handles in (("imported", self.imported), ("owned", self.owned)):
            for peer, handle in list(handles.items()):
                status = self._destroy(handle)
                if status == 0:
                    del handles[peer]
                else:
                    failures.append(f"{role} TP{peer}: ACL status {status}")
        if failures:
            raise RuntimeError("FSA planner IPC Notify release failed: " + ", ".join(failures))
        self.owner_payload = None
