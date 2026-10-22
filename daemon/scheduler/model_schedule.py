import time
import os
import json
import asyncio
import enum
from dataclasses import dataclass
from urllib.parse import urlparse

import aiohttp
import websockets


class ProcessStatus(enum.IntEnum):
    RUNNING = 1
    SLEEPING = 2
    SWAPPING = 3

@dataclass
class ModelStats:
    model_instance: str                # 实例名称
    last_switch_time: float            # 上次切换时间
    status: ProcessStatus              # 运行状态
    model_size: int                    # 模型权重大小

    pending_requests: int = 0          # 积压请求数
    idle_times: int = 0                # 连续空闲次数
    priority: float = 1                # 换入优先级

@dataclass
class LoadUpdateContext:
    load_stats: list[ModelStats]       # 待更新的模型状态
    load_results: list                 # 负载查询结果
    backlog_stats: list[ModelStats]    # 待更新的模型状态   
    backlog_results: list              # 堆积查询结果


class WebSocketClient:
    """
    Establish a persistent connection with the master scheduler and upload load information.
    """
    def __init__(self, uri):
        self.uri = uri
        self.websocket = None
        self.is_connected = False

    async def build_connect(self) -> bool:
        """建立WebSocket连接"""
        print("WebSocketClient build")
        try:
            self.websocket = await asyncio.wait_for(
                websockets.connect(
                    self.uri,
                    ping_interval=60,
                    ping_timeout=30
                ),
                timeout=10
            )
            self.is_connected = True
            print("WebSocket connection established successfully.")            
        except asyncio.TimeoutError:
            print("error: Connection timed out during handshake.")
        except Exception as e:
            print(f"error: Unexpected error during connection: {e}")

        return self.is_connected

    async def close_connect(self) -> None:
        """安全关闭连接"""
        if self.websocket:
            print("Closing WebSocket connection...")
            await self.websocket.close()
            self.is_connected = False

    async def send_message(self, data: dict) -> None:
        """发送消息到服务端"""
        if self.websocket:
            try:
                message = json.dumps(data, default=str, ensure_ascii=False)
                await self.websocket.send(message)
            except TypeError as e:
                raise TypeError(f"Data is not JSON serializable: {data}")
            except Exception as e:
                raise RuntimeError(f"Failed to send : {e}")
        else:
            raise RuntimeError("Connection is not open, cannot send message")

async def send_control_request(session: aiohttp.ClientSession, url: str, order: str) -> None:
    service_url = f"http://{url}/{order}"
    try:
        async with session.post(service_url) as response:
            response.raise_for_status()
            await response.release()
    except Exception as e:
        raise RuntimeError(f"Failed to send order: {service_url}") from e

async def get_model_size(session: aiohttp.ClientSession, url: str) -> int:
    service_url = f"http://{url}/get_model_size"
    try:
        async with session.get(service_url) as response:
            response.raise_for_status()
            data = await response.json()
            return data.get("model_size")
    except Exception as e:
        raise RuntimeError(f"Failed to get model size by {service_url}") from e

async def get_request_counts(session: aiohttp.ClientSession, url: str) -> list:
    service_url = f"http://{url}/get_request_counts"
    try:
        async with session.get(service_url) as response:
            response.raise_for_status()
            data = await response.json()
            return data.get("request_counts")
    except Exception as e:
        raise RuntimeError(f"Failed to get request counts by {service_url}") from e

async def get_model_running_stats(session: aiohttp.ClientSession, url: str) -> bool:
    service_url = f"http://{url}/is_sleeping"
    try:
        async with session.get(service_url) as response:
            response.raise_for_status()
            data = await response.json()
            return not data.get("is_sleeping")
    except Exception as e:
        raise RuntimeError(f"Failed to get model stats by {service_url}") from e

async def get_backlog_count(session: aiohttp.ClientSession, url: str) -> int:
    service_url = f"http://{url}/get_blocked_num"
    try:
        async with session.get(service_url) as response:
            response.raise_for_status()
            data = await response.json()
            return data.get("blocked_requests_num")
    except Exception as e:
        raise RuntimeError(f"Failed to get blocked requests num by {service_url}") from e


class ModelScheduler:
    def __init__(self, config_path, weight_space):
        # init config
        self.pod_name = ""
        self.instance_infos = None
        self.instance_infos_map = {}
        self.websocket_url = ""
        self._init_config(config_path)
        self.free_space = weight_space    # 权重空间大小，单位 GB

        self.websocket = None
        self.running_models = {}
        self.sleeping_models = {}

        self.is_ready = False
        self.session = None
        self._stop_event = asyncio.Event()

        # 核心参数配置
        self.IDLE_TIMES = 3           # 空闲超时次数：决定是否切换去处理别人
        self.MAX_STAY_TIME = 30.0     # 最大驻留
        self.ALPHA = 0.5
        self.POLL_INTERVAL = 0.1          # 轮询间隔
        self.loaded_models = []       # 可换出模型
        self.pending_queue = []       # 待换入模型
        self.model_load_status = {}   # 保存模型状态信息
    
        self.lock = asyncio.Lock()
        self.space_available = asyncio.Condition(self.lock)

        self.update_task = None
        self.swap_out_task = None
        self.swap_in_task = None

    async def send_message(self, data: dict[str, any]) -> None:
        """通过 websocket 发送信息"""
        if self.websocket:
            await self.websocket.send_message(data)
        else:
            raise RuntimeError("WebSocket connection is not available")

    async def get_session(self) -> aiohttp.ClientSession:
        """获取或创建全局 ClientSession"""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession(
                connector=aiohttp.TCPConnector(
                    limit=100,
                    limit_per_host=20,
                    keepalive_timeout=30
                ),
                timeout=aiohttp.ClientTimeout(total=300)
            )
        return self.session

    async def close_session(self) -> None:
        """关闭全局 ClientSession"""
        if self.session and not self.session.closed:
            await self.session.close()

    def calculate_priority(self, pending_requests, max_pending, stay_time, max_stay) -> float:
        """计算等待换入模型优先度"""
        """优先度: priority = a * stats.pending_requests / max_pending + (1 - a) * stay_time / max_stay """
        priority = self.ALPHA * pending_requests / max_pending + (1 - self.ALPHA) * stay_time / max_stay
        return priority

    def update_status_infos(self, context: LoadUpdateContext) -> None:
        """统一更新模型状态信息"""
        load_stats = context.load_stats
        load_results = context.load_results
        for index, stats in enumerate(load_stats):
            running_load, waiting_load = load_results[index]
            self.model_load_status[stats.model_instance] = running_load + waiting_load

            if stats.status != ProcessStatus.RUNNING:
                continue
            is_free = running_load == waiting_load == 0
            stats.idle_times = stats.idle_times + 1 if is_free else 0
            stay_time = time.time() - stats.last_switch_time
            if stats.idle_times >= self.IDLE_TIMES or stay_time >= self.MAX_STAY_TIME:
                if stats not in self.loaded_models:
                    self.loaded_models.append(stats)
                    print(f"--- [可换出队列更新] 实例 {stats.model_instance} 可换出 ---")

        backlog_stats = context.backlog_stats
        backlog_results = context.backlog_results
        max_pending = 0
        max_stay = 0
        for index, stats in enumerate(backlog_stats):
            stats.pending_requests = backlog_results[index]
            if stats.status != ProcessStatus.SLEEPING or stats.pending_requests <= 0:
                continue
            stay_time = time.time() - stats.last_switch_time
            max_pending = stats.pending_requests if stats.pending_requests > max_pending else max_pending
            max_stay = stay_time if stay_time > max_stay else max_stay
            priority = self.calculate_priority(stats.pending_requests, max_pending, stay_time, max_stay)
            stats.priority = priority

            if stats not in self.pending_queue:
                self.pending_queue.append(stats)
                print(f"--- [待换入队列更新] 实例 {stats.model_instance} 待换入 ---")
                
    async def update_monitor(self) -> None:
        """监听负载，更新模型状态"""
        try:
            session = await self.get_session()
            while True:
                await asyncio.sleep(self.POLL_INTERVAL) 
                load_stats = []
                backlog_stats = []
                async with self.lock:
                    if not self.is_ready:
                        continue
                    load_tasks = []
                    backlog_tasks = []

                    for stats in self.running_models.values():
                        url = stats.model_instance
                        load_task = asyncio.create_task(get_request_counts(session, url))
                        load_tasks.append(load_task)
                        load_stats.append(stats)

                        backlog_task = asyncio.create_task(get_backlog_count(session, url))
                        backlog_tasks.append(backlog_task)
                        backlog_stats.append(stats)

                    for stats in self.sleeping_models.values():
                        url = stats.model_instance
                        backlog_task = asyncio.create_task(get_backlog_count(session, url))
                        backlog_tasks.append(backlog_task)
                        backlog_stats.append(stats)

                        self.model_load_status[stats.model_instance] = 0

                load_results = await asyncio.gather(*load_tasks)
                backlog_results = await asyncio.gather(*backlog_tasks)
                async with self.lock:
                    context = LoadUpdateContext(load_stats, load_results, backlog_stats, backlog_results)
                    self.update_status_infos(context)
        except asyncio.CancelledError:
            pass

    async def swap_out_monitor(self) -> None:
        """模型换出 loop"""
        try:
            session = await self.get_session()
            while True:
                await asyncio.sleep(self.POLL_INTERVAL)
    
                async with self.lock:
                    if not self.is_ready:
                        continue
                    demand_needed = 0
                    for pending_stats in self.pending_queue:
                        demand_needed += pending_stats.model_size

                    accumulated_size = 0
                    sleep_models = []
                    tasks = []
                    while demand_needed > 0 and self.loaded_models:
                        largest_model = max(self.loaded_models, key=lambda m: m.model_size)
                        sleep_models.append(largest_model)
                        largest_model.status = ProcessStatus.SWAPPING
                        self.loaded_models.remove(largest_model)
                        print(f"--- [换出管理] 决定换出模型 {largest_model.model_instance} ({largest_model.model_size}GB) 以填补缺口 ---")
                        
                        url = self.instance_infos_map[largest_model.model_instance].get("url", None)
                        task = asyncio.create_task(send_control_request(session, url, "sleep"))
                        tasks.append(task)

                        accumulated_size += largest_model.model_size
                        demand_needed -= largest_model.model_size
        
                    if accumulated_size > 0:
                        print(f"--- [换出管理] 本轮计划释放 {accumulated_size}GB, 等待后台任务完成... ---")
                await asyncio.gather(*tasks)
                if accumulated_size > 0:
                    print(f"--- [换出管理] {accumulated_size}GB 已释放 ---")

                async with self.space_available:
                    for model in sleep_models:
                        model.status = ProcessStatus.SLEEPING
                        model.last_switch_time = time.time()

                        del self.running_models[model.model_instance]
                        self.sleeping_models[model.model_instance] = model
                    self.free_space += accumulated_size
                    if sleep_models:
                        self.space_available.notify_all()
        except asyncio.CancelledError:
            pass

    async def swap_in_monitor(self) -> None:
        """模型换入 loop"""
        try:
            session = await self.get_session()
            while True:
                async with self.lock:
                    if not self.is_ready:
                        await asyncio.sleep(self.POLL_INTERVAL)
                        continue
                    loaded = False
                    free_space = self.free_space
                    accumulated_size = 0
                    wake_up_models = []
                    tasks = []

                    while free_space > 0 and self.pending_queue:
                        priority_model = max(self.pending_queue, key=lambda m: m.priority)
                        if priority_model.model_size > free_space:
                            break
                        loaded = True
                        wake_up_models.append(priority_model)
                        priority_model.status = ProcessStatus.SWAPPING
                        self.pending_queue.remove(priority_model)
                        print(f"--- [换入管理] 决定换入模型 {priority_model.model_instance} ({priority_model.model_size}GB)")

                        url = self.instance_infos_map.get(priority_model.model_instance).get("url", None)
                        task = asyncio.create_task(send_control_request(session, url, "wake_up"))
                        tasks.append(task)

                        free_space -= priority_model.model_size
                        accumulated_size += priority_model.model_size

                await asyncio.gather(*tasks)    # 等待换入完成
                if accumulated_size > 0:
                    print(f"--- [换入管理] 模型完成换入，一共使用 ({accumulated_size}GB)")

                async with self.space_available:
                    if not loaded:
                        if self.pending_queue:
                            try:
                                await asyncio.wait_for(self.space_available.wait(), timeout=0.5)
                            except asyncio.TimeoutError:
                                pass
                        else:
                            await asyncio.sleep(self.POLL_INTERVAL)
                    else:
                        for model in wake_up_models:
                            model.status = ProcessStatus.RUNNING
                            model.last_switch_time = time.time()
                            del self.sleeping_models[model.model_instance]
                            self.running_models[model.model_instance] = model
                        self.free_space -= accumulated_size
        except asyncio.CancelledError:
            pass

    async def upload_monitor(self) -> None:
        """负载信息上传 loop"""
        try:
            while True:
                await asyncio.sleep(self.POLL_INTERVAL)
                
                async with self.lock:
                    if not self.is_ready:
                        continue
                    
                    load_data = {}
                    for status in self.running_models.values():
                        instance =  status.model_instance
                        model_name = self.instance_infos_map[instance].get("model_name", None)
                        load_num = self.model_load_status.get(instance)
                        load_data[model_name] = load_num + status.pending_requests
                
                    for status in self.sleeping_models.values():
                        instance =  status.model_instance
                        model_name = self.instance_infos_map[instance].get("model_name", None)
                        load_num = self.model_load_status.get(instance)
                        load_data[model_name] = status.pending_requests
                    
                    data = {
                        "type" : "load_info",
                        "load_info" : {}
                    }
                    data["load_info"] = load_data

                    await self.send_message(data)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"负载信息上传失败: {e}")

    async def start(self) -> None:
        """调度器执行入口"""
        print("=== [启动监控任务] ===")

        self.update_task = asyncio.create_task(self.update_monitor())
        self.swap_out_task = asyncio.create_task(self.swap_out_monitor())
        self.swap_in_task = asyncio.create_task(self.swap_in_monitor())    
        self.upload_task = asyncio.create_task(self.upload_monitor())

        await self._stop_event.wait()    # 等待停止信号

    async def get_user_input(self) -> None:
        """等待手动触发"""
        # 建立长连接
        await self._build_connect()

        print("[启动] 请在控制台按回车以开始调度：")
        await asyncio.to_thread(input, "")
        print("[启动] 开始调度：")

        await self._init_model_stats()
    
        async with self.lock:
            self.is_ready = True

    async def stop(self) -> None:
        """资源释放"""
        self._stop_event.set()  # 发送停止信号
        await self.close_session()
        await self.websocket.close_connect()

        all_tasks = [self.update_task, self.swap_out_task, self.swap_in_task]
        tasks = [t for t in all_tasks if t is not None]
        for task in tasks:
            if not task.done():
                task.cancel()
        try:
            await asyncio.gather(*tasks, return_exceptions=True)
        except asyncio.CancelledError:
            pass

    def _init_config(self, config_path: str) -> None:
        """配置文件加载"""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"配置文件不存在: {config_path}")

        with open(config_path, 'r', encoding='utf-8') as scheduler_file:
            data = json.load(scheduler_file)
            self.instance_infos = data.get("InstanceInfos", None)
            self.pod_name = data.get("PodName", None)
            self.websocket_url = data.get("WebsocketUrl", None)

        def check_url(url: str):
            parsed = urlparse("http://"+ url)
            if not parsed.scheme or not parsed.netloc:
                raise ValueError(f"Invalid URL {url}")

        for instance_info in self.instance_infos:
            name = instance_info.get("url", None)
            check_url(name)
            if name in self.instance_infos_map:
                raise FileNotFoundError(f"实例url重复, url : {name}")
            self.instance_infos_map[name] = instance_info

    async def _init_model_stats(self) -> None:
        """初始化模型状态"""
        try:
            session = await self.get_session()
            async with self.lock:
                running_tasks = []
                model_size_tasks = []
                for info in self.instance_infos:
                    url = info.get("url", None)
                    running_task = asyncio.create_task(get_model_running_stats(session, url))
                    running_tasks.append(running_task)
                    model_size_task = asyncio.create_task(get_model_size(session, url))
                    model_size_tasks.append(model_size_task)

            running_results = await asyncio.gather(*running_tasks)
            model_size_results = await asyncio.gather(*model_size_tasks)

            async with self.lock:
                for index, info in enumerate(self.instance_infos):
                    model_instance = info.get("url", None)
                    status = ProcessStatus.RUNNING if running_results[index] else ProcessStatus.SLEEPING
                    model_size = model_size_results[index]
                    model_stats = ModelStats(model_instance=model_instance, last_switch_time=time.time(), status=status,
                                            model_size=model_size)

                    if status == ProcessStatus.RUNNING:
                        self.running_models[model_instance] = model_stats
                        self.free_space -= model_size
                    else:
                        self.sleeping_models[model_instance] = model_stats
                    
                    self.model_load_status[model_instance] = 0

        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"模型状态初始化失败: {e}")

    async def _build_connect(self) -> None:
        """与 websocket 建立长连接"""
        self.websocket = WebSocketClient(self.websocket_url)
        if await self.websocket.build_connect():
            data = {
                "type" : "register",
                "pod_name" : self.pod_name
            }
            await self.send_message(data)

async def main():
    import argparse
    parser = argparse.ArgumentParser(description="模型切换调度器")
    parser.add_argument('-config-path', required=True, help='配置文件的路径')
    parser.add_argument('-weight-space', type=float, required=True, help='设置的最大使用的权重空间大小，单位 GB')

    args = parser.parse_args()
    try:
        kv_scheduler = ModelScheduler(args.config_path, args.weight_space)
    except Exception as e:
        print(f"捕获到异常，创建失败: {e}")
        return

    server_task = asyncio.create_task(kv_scheduler.start())
    input_task = asyncio.create_task(kv_scheduler.get_user_input())
    try:
        await input_task
        await server_task
    except asyncio.CancelledError:
        print("接收到取消信号，正在清理资源...")
    except KeyboardInterrupt:
        print("接收到键盘中断，正在清理资源...")
    except Exception as e:
        print(f"错误: {e}")
    finally:
        await kv_scheduler.stop()
        all_tasks = [server_task, input_task]
        tasks = [t for t in all_tasks if t is not None]
        for task in tasks:
            if not task.done():
                task.cancel()
        try:
            await asyncio.gather(*tasks, return_exceptions=True)
        except asyncio.CancelledError:
            pass

        print("资源清理完毕，程序退出。")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    finally:
        print('退出')