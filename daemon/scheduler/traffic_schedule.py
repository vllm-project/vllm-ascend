import os
import json
import asyncio
from collections import defaultdict
from dataclasses import dataclass

import websockets
from websockets.exceptions import ConnectionClosed
from kubernetes import client, config


class K8sManager:
    def __init__(self):
        """init k8s"""
        try:
            config.load_incluster_config()
        except config.ConfigException as e:
            # Not supported
            raise RuntimeError(f"Used only in the cluster: {e}") from e

        self.v1 = client.CoreV1Api()
        self.namespace = "default"

    async def patch_pod_labels(self, pod_name, labels_dict: dict[str, str]) -> None:
        """修改Pod的标签"""
        body = {
            "metadata": {
                "labels": labels_dict
            }
        }

        def _sync_patch():
            self.v1.patch_namespaced_pod(name=pod_name, namespace=self.namespace, body=body)
        try:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, _sync_patch)
        except Exception as e:
            raise RuntimeError(f"Failed to patch pod {pod_name}: {e}")

    def get_pod_labels(self, pod_name) -> dict[str, any]:
        """获取 Pod 的标签"""
        pod = self.v1.read_namespaced_pod(name=pod_name, namespace=self.namespace)
        return pod.metadata.labels

@dataclass
class PodLoadInfo:
    pod_name : str              # pod 名称
    load_info : dict = None     # pod上模型对应负载
    using_model : str = None    # 占用的模型


class TrafficScheduler:
    def __init__(self, host, port, config_path):
        self.host = host
        self.port = port
        self.model_name_labels_map: dict[str, str] = {}
        self.model_threshold : dict[str, int] = {}
        self._init_config(config_path)

        self.manager = K8sManager()    # k8s 

        self.schedule_clients = set()
        self.pod_load_infos: dict[str, PodLoadInfo] = {}        
        self.model_labels = defaultdict(dict[str, str])

        self.server = None
        self.POLL_INTERVAL = 0.5                          # loop time(s)
        self._stop_event = asyncio.Event()

        self.lock = asyncio.Lock()
        self.is_ready = False

    async def handle_client(self, websocket) -> None:
        """websocket 消息接收，采集负载信息"""
        print("-- 客户端信息： --")
        print("remote address: ", websocket.remote_address)
        self.schedule_clients.add(websocket)
        pod_name = ""
        try:
            async for message in websocket:
                message_data = json.loads(message)
                # 注册信息
                async with self.lock:
                    if message_data.get("type") == "register":
                        pod_name = message_data.get("pod_name")
                        load_info = PodLoadInfo(pod_name=pod_name)
                        self.pod_load_infos[pod_name] = load_info
                        
                    # 负载信息
                    if message_data.get("type") == "load_info":
                        load_info = message_data.get("load_info")
                        self.pod_load_infos[pod_name].load_info = load_info

        except json.JSONDecodeError:
            raise ValueError("Received malformed JSON data from client") from json.JSONDecodeError
        except ConnectionClosed:
            self.schedule_clients.remove(websocket)
            if pod_name in self.pod_load_infos:
                del self.pod_load_infos[pod_name]
        finally:
            print("退出连接")
            pass

    def group_load_metrics(self) -> dict[str, list[tuple]]:
        """统计模型在不同pod的负载情况"""
        model_pod_loads = defaultdict(list[tuple])

        for pod_name, pod_load in self.pod_load_infos.items():
            for model_name, model_load in pod_load.load_info.items():
                model_pod_loads[model_name].append((pod_name, model_load))
 
        for pod_loads in model_pod_loads.values():
            pod_loads.sort(key=lambda x: x[1], reverse=True)    #从大到小 

        return model_pod_loads

    def update_instance_count(self, model_pod_loads: dict[str, list]) -> None:
        """根据负载和阈值，重新分配实例数"""
        need_expand = []
        is_shrinkable = {}
        for model_name, pod_loads in model_pod_loads.items():    # 统计需要扩容的模型实例以及可以缩容的模型实例
            all_load = 0
            use_instance_count = 0
            for pod_load in pod_loads:
                pod_name = pod_load[0]
                all_load += pod_load[1]
                use_instance_count += 1 if self.pod_load_infos.get(pod_name).using_model == model_name else 0
                if self.pod_load_infos.get(pod_name).using_model is None:
                    use_instance_count += 1
                    self.pod_load_infos.get(pod_name).using_model = model_name

            threshold = self.model_threshold[model_name]
            need_count = max(1, all_load // threshold + (1 if all_load % threshold else 0))

            if need_count > use_instance_count:
                expand_count = (need_count - use_instance_count)
                need_expand.append((model_name, expand_count))
            elif need_count < use_instance_count:
                shrink_count = (use_instance_count - need_count)
                is_shrinkable[model_name] = shrink_count

        for model_name, expend_num in need_expand:     # 需要扩容的实例按需扩容，从可缩容实例中选择负载低的替换
            pod_loads = model_pod_loads.get(model_name)
            all_shrinkable = []

            for pod_load in pod_loads:     # 排序可以替换的实例
                pod_name = pod_load[0]
                using_model = self.pod_load_infos.get(pod_name).using_model
                using_model_load = self.pod_load_infos.get(pod_name).load_info.get(using_model)
                if using_model != model_name and using_model in is_shrinkable:
                    all_shrinkable.append((using_model_load, using_model, pod_name))
            all_shrinkable.sort(key=lambda x: x[0], reverse=False)

            model_map = {}
            expend_pod = []
            for _, other_model, pod_name in all_shrinkable:    # 决定被替换的模型实例
                if len(expend_pod) >= expend_num:
                    break
                if other_model in model_map and model_map[other_model] >= is_shrinkable[other_model]:
                    break
                if other_model not in model_map:
                    model_map[other_model] = 1
                else:
                    model_map[other_model] += 1
                expend_pod.append(pod_name)
            
            for pod_name in expend_pod:
                using_model = self.pod_load_infos.get(pod_name).using_model
                self.pod_load_infos.get(pod_name).using_model = model_name
                is_shrinkable[using_model] -= 1

    def calculate_even_forwarding(self, model_pod_loads: dict[str, list]) -> list[tuple[str, dict]]:
        """计算均匀转发条件下的标签修改"""
        revise_results = []
        for model_name, pod_loads in model_pod_loads.items():
            all_load = 0
            using_pods = []
            revise_labels = {}
            for pod_load in pod_loads:
                pod_name = pod_load[0]
                load_num = pod_load[1]
                using_model = self.pod_load_infos.get(pod_name).using_model
                label_status = self.model_labels.get(model_name).get(pod_name)
                if using_model != model_name:
                    if label_status == "active":
                        revise_labels[pod_name] = 0
                    continue
                all_load += load_num
                using_pods.append(pod_load)
            
            active_pods = []
            min_load_level = None
            avg_load = all_load/len(using_pods)
            for using_pod in using_pods:
                label_status = self.model_labels.get(model_name).get(using_pod[0])
                if using_pod[1] < avg_load and label_status == "deactive":
                    revise_labels[using_pod[0]] = 1    # 开启
                    active_pods.append(using_pod)
                    min_load_level = using_pod[1] if not min_load_level else min(min_load_level, using_pod[1])
                if label_status == "active" :
                    active_pods.append(using_pod)
                    min_load_level = using_pod[1] if not min_load_level else min(min_load_level, using_pod[1])

            tolerance = 10
            has_active = False
            for active_pod in active_pods:
                load_num = active_pod[1]
                if load_num - min_load_level > tolerance:
                    revise_labels[active_pod[0]] = 0    # 关闭
                else:
                    has_active = True
            if not has_active and active_pods:
                raise RuntimeError("Labels assignment error, active status label does not exist.")
            
            if revise_labels:
                revise_results.append((model_name, revise_labels))
        
        return revise_results

    async def load_monitor(self) -> None:
        """负载监控与实例调度"""
        try:
            while True:
                await asyncio.sleep(self.POLL_INTERVAL)
                if not self.is_ready:
                    continue
                if len(self.pod_load_infos) < 2:
                    continue

                await self._update_model_label()
                async with self.lock:
                    model_pod_loads = self.group_load_metrics()
                    print("model_pod_loads: ", model_pod_loads)

                    self.update_instance_count(model_pod_loads)
                    revise_results = self.calculate_even_forwarding(model_pod_loads)

                revise_order_map = defaultdict(list)
                active_pods = []
                for model_name, revise_labels in revise_results:
                    for pod_name, status in revise_labels.items():
                        revise_order_map[pod_name].append((model_name, status))
                        # 记录开关状态1的pod
                        if status == 1:
                            active_pods.append(pod_name)

                if revise_order_map:
                    await self.notify_k8s(revise_order_map, active_pods)

        except asyncio.CancelledError:
            pass

    async def notify_k8s(self, revise_order_map: dict, active_pods: list) -> None:
        """pod标签修改"""
        def build_labels(revise_order):
            if not revise_order:
                return {}
            return {
                self.model_name_labels_map[mn]: ("active" if st == 1 else "deactive")
                for mn, st in revise_order
                if mn in self.model_name_labels_map 
            }

        priority_tasks = [
            self.manager.patch_pod_labels(pod, build_labels(revise_order_map.get(pod)))
            for pod in active_pods
            if pod in revise_order_map and revise_order_map.get(pod)
        ]
        if priority_tasks:
            await asyncio.gather(*priority_tasks)

        active_set = set(active_pods)
        other_tasks = [
            self.manager.patch_pod_labels(pod, build_labels(order))
            for pod, order in revise_order_map.items()
            if pod not in active_set and order
        ]
        if other_tasks:
            await asyncio.gather(*other_tasks)

    async def start(self) -> None:
        """调度器执行入口"""
        print("=== [启动监控任务] ===")
        self.server = await websockets.serve(self.handle_client, self.host, self.port)
        self.monitoring_task = asyncio.create_task(self.load_monitor())

        await self._stop_event.wait()    # 等待停止信号

    async def get_user_input(self) -> None:
        """等待手动触发"""
        print("[启动] 请在控制台按回车以开始调度：")
        await asyncio.to_thread(input, "")
        print("[启动] 开始调度：")

        await self._update_model_label(True)
        self.is_ready = True

    async def stop(self) -> None:
        """资源释放"""
        self._stop_event.set()
        if self.server:
            self.server.close()
            await self.server.wait_closed()
            print("Server stopped")

        task = self.monitoring_task
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
    
    def _init_config(self, config_path: str) -> None:
        """配置文件加载"""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"配置文件不存在: {config_path}")

        with open(config_path, 'r', encoding='utf-8') as scheduler_file:
            data = json.load(scheduler_file)
            self.model_name_labels_map = data.get("ModelLabels", None)
            self.model_threshold = data.get("ModelThreshold", None)
            self.host = data["WebsocketUrl"].get("host", None)
            self.port = data["WebsocketUrl"].get("port", None)

    async def _update_model_label(self, is_init=False) -> None:
        """更新模型标签"""
        for pod_name, load_info in self.pod_load_infos.items():
            pod_labels = self.manager.get_pod_labels(pod_name)
            if not load_info.load_info:
                raise RuntimeError("Please start the model scheduler first.")
            async with self.lock:
                for model_name in load_info.load_info:
                    model_label = self.model_name_labels_map.get(model_name, None)
                    label_state = pod_labels[model_label]
                    self.model_labels[model_name][pod_name] = label_state
            
                    if is_init and label_state == "active":
                        if load_info.using_model != None:
                            raise RuntimeError("Ensure that only one model label is activated on the pod.")
                        load_info.using_model = model_name

async def main():
    import argparse
    parser = argparse.ArgumentParser(description="流量转发调度器")
    parser.add_argument('-config-path', required=True, help='配置文件的路径')

    args = parser.parse_args()

    host="127.0.0.1"
    port=2233
    ws_server = TrafficScheduler(host=host, port=port, config_path=args.config_path)
    # begin server
    server_task = asyncio.create_task(ws_server.start())
    input_task = asyncio.create_task(ws_server.get_user_input())
    try:
        await input_task
        await server_task
    except asyncio.CancelledError:
        print("接收到取消信号，正在清理资源...")
    except KeyboardInterrupt:
        print("接收到键盘中断，正在清理资源...")
    finally:
        await ws_server.stop()
        print("资源清理完毕，程序退出。")

if __name__ == "__main__":
    asyncio.run(main())