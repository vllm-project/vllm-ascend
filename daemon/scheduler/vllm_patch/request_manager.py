# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio


class RequestManager:
    _instance = None
    _lock = asyncio.Lock() # 用于单例创建的锁

    def __init__(self):
        if hasattr(self, '_initialized') and self._initialized:
            return
        self.blocked_requests_num = 0
        self._count = 0
        self.is_running = True

        self._value_lock = asyncio.Lock()
        self.condition = asyncio.Condition()
        self._initialized = True

    @classmethod
    async def get_instance(cls):
        async with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
        return cls._instance

    async def increment(self):
        async with self._value_lock:
            self.blocked_requests_num += 1
            self._count +=1

    async def decrement(self):
        async with self._value_lock:
            self.blocked_requests_num -= 1

    async def get_num(self):
        async with self._value_lock:
            return self.blocked_requests_num

    async def get_count_id(self):
        async with self._value_lock:
            return self._count

    async def set_running_status(self, status: bool):
        need_notify = False
        async with self._value_lock:
            if self.is_running is not True and status is True:
                self.is_running = status
                need_notify = True
            else:
                self.is_running = status

        if need_notify:
            await self.broadcast_notify()

    async def get_running_status(self):
        async with self._value_lock:
            return self.is_running

    async def wait_for_notify(self):
        async with self.condition:
            await self.condition.wait()

    async def broadcast_notify(self):
        async with self.condition:
            self.condition.notify_all()