from collections import deque
from operator import truediv
from unittest.mock import MagicMock, patch

from sympy import false
from vllm.config import VllmConfig
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry
from vllm.v1.core.sched.output import NewRequestData, SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.structured_output import StructuredOutputManager

from tests.ut.base import TestBase
from vllm_ascend.core.scheduler import AscendScheduler


class TestAscendScheduler(TestBase):
    @patch("vllm.v1.core.sched.scheduler.Scheduler.__init__")
    def test_init(self, mock_super_init: MagicMock):
        mock_vllm_config = MagicMock(spec=VllmConfig)
        mock_kv_cache_config = MagicMock(spec=KVCacheConfig)
        mock_structured_output_manager = MagicMock(
            spec=StructuredOutputManager)
        mock_mm_registry = MagicMock(spec=MULTIMODAL_REGISTRY)

        scheduler = AscendScheduler(
            vllm_config=mock_vllm_config,
            kv_cache_config=mock_kv_cache_config,
            structured_output_manager=mock_structured_output_manager,
            mm_registry=mock_mm_registry,
            include_finished_set=True)

        mock_super_init.assert_called_once_with(
            mock_vllm_config, mock_kv_cache_config,
            mock_structured_output_manager, mock_mm_registry, True, False)

        self.assertEqual(scheduler.scheduled_req_ids, set())
        self.assertEqual(scheduler.running, [])

    def test_schedule_init(self):
        # 给测试schedule函数初始化一个相应的对象
        self.scheduler = MagicMock(spec=AscendScheduler)
        self.scheduler.waiting = deque()
        self.scheduler.running = []
        self.scheduler.finished_req_ids = set()
        self.scheduler.scheduled_req_ids = set()
        self.scheduler.kv_cache_manager = MagicMock()
        self.scheduler.connector = None
        self.scheduler.lora_config = None
        self.scheduler.scheduler_config = MagicMock()
        self.scheduler.scheduler_config.chunked_prefill_enabled = False
        self.scheduler.scheduler_config.watermark = 0.01
        self.scheduler.max_num_scheduled_tokens = 100
        self.scheduler.max_num_running_reqs = 10  # 最大运行请求
        self.scheduler.num_lookahead_tokens = 0
        self.scheduler.log_stats = True

    def test_get_prompt_limit(self):
        scheduler = MagicMock(spec=AscendScheduler)
        scheduler.scheduler_config = MagicMock()
        scheduler.scheduler_config.max_model_len = 1000
        scheduler.scheduler_config.max_num_batched_tokens = 500
        scheduler.scheduler_config.chunked_prefill_enabled = True
        scheduler.scheduler_config.is_multi_step = False

        request = MagicMock(sepc=Request)
        request.lora_request = True
        request.lora_request.long_lora_max_len = 1500
        res1 = scheduler._get_prompt_limit(request)
        self.assertEqual(res1, request.lora_request.long_lora_max_len)

        request.lora_request = False
        res2 = scheduler._get_prompt_limit(request)
        self.assertEqual(res2, scheduler.scheduler_config.max_model_len)

        scheduler.scheduler_config.chunked_prefill_enabled = False
        scheduler.scheduler_config.max_num_batched_tokens = 999

        res3 = scheduler._get_prompt_limit(request)
        self.assertEqual(res3,
                         scheduler.scheduler_config.max_num_batched_tokens)

    @patch("vllm.v1.core.sched.scheduler.Scheduler.finish_requests")
    def test_finish_requests_single(self, mock_finish_requests):
        scheduler = MagicMock(spec=AscendScheduler)
        scheduler.requests = {}  #! 保存scheduler中的requests
        scheduler.scheduled_req_ids = set()  #! 保存要进行处理的request的ids

        req_id = "req1"
        request = MagicMock(spec=Request)
        request.status = RequestStatus.RUNNING
        request.request_id = req_id
        scheduler.requests[req_id] = request

        scheduler.scheduled_req_ids.add(req_id)
        status = MagicMock(spec=RequestStatus)
        scheduler.finish_requests(req_id, status)

        self.assertNotIn(req_id, self.scheduler.scheduled_req_ids)
        mock_finish_requests.assert_called_once(req_id, status)

    @patch("vllm.v1.core.sched.scheduler.Scheduler.finish_requests")
    def test_finish_requests_single_multiple(self, mock_finish_requests):
        scheduler = MagicMock(spec=AscendScheduler)
        scheduler.requests = {}  # ! 保存scheduler中的requests
        scheduler.scheduled_req_ids = set()  # ! 保存要进行处理的request的ids

        req_ids = ["req1", "req2"]
        request1 = MagicMock(spec=Request)
        request1.status = RequestStatus.RUNNING
        request1.request_id = req_ids[0]
        request2 = MagicMock(spec=Request)
        request2.status = RequestStatus.WAITING
        request2.request_id = req_ids[1]
        scheduler.requests = {req_ids[0]: request1, req_ids[1]: request2}

        scheduler.scheduled_req_ids.update(req_ids)
        status = MagicMock(RequestStatus)
        scheduler.finish_requests(req_ids, status)

        self.assertNotIn(req_ids[0], self.scheduler.scheduled_req_ids)
        self.assertIn(req_ids[1], self.scheduler.scheduled_req_ids)
        mock_finish_requests.finish_requests.assert_called_once_with(
            req_ids, RequestStatus.FINISHED)

    @patch("vllm.v1.core.sched.scheduler.Scheduler.update_from_output")
    def test_update_from_output_single_scheduled(self,
                                                 mock_update_from_output):
        scheduler = MagicMock(spec=AscendScheduler)
        scheduler.running = []
        scheduler.requests = {}  # ! 保存scheduler中的requests
        scheduler.scheduled_req_ids = set()
        scheduler.update_from_output = MagicMock()  # ! 保存要进行处理的request的ids

        req_id = "req1"
        request = MagicMock(spec=Request)
        request.request_id = req_id

        # Mock输入
        scheduler_output = MagicMock(spec=SchedulerOutput)
        model_runner_output = MagicMock(spec=ModelRunnerOutput)
        scheduler_output.num_scheduled_tokens = {"req1": 10}
        scheduler.running.append(request)
        scheduler.scheduled_req_ids.add(req_id)

        scheduler.update_from_output(scheduler_output, model_runner_output)
        self.assertNotIn(req_id, scheduler.scheduled_req_ids)
        mock_update_from_output.assert_called_once_with(
            scheduler_output, model_runner_output)

    @patch("vllm.v1.core.sched.scheduler.Scheduler.update_from_output")
    def test_update_from_output_single_unscheduled(self,
                                                   mock_update_from_output):
        scheduler = MagicMock(spec=AscendScheduler)
        scheduler.running = []
        scheduler.requests = {}  # ! 保存scheduler中的requests
        scheduler.scheduled_req_ids = set()
        scheduler.update_from_output = MagicMock()  # ! 保存要进行处理的request的ids

        req_id = "req1"
        request = MagicMock(spec=Request)
        request.request_id = req_id

        # Mock输入
        scheduler_output = MagicMock(spec=SchedulerOutput)
        model_runner_output = MagicMock(spec=ModelRunnerOutput)
        scheduler_output.num_scheduled_tokens = {"req1": 0}
        scheduler.running.append(request)
        scheduler.scheduled_req_ids.add(req_id)

        scheduler.update_from_output(scheduler_output, model_runner_output)
        self.assertIn(req_id, scheduler.scheduled_req_ids)
        mock_update_from_output.assert_called_once_with(
            scheduler_output, model_runner_output)
