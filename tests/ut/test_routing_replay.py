import asyncio
import os
from unittest.mock import patch

from unittest import TestBase
from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.sampling_params import RequestOutputKind
from vllm.v1.engine.async_llm import AsyncLLM


class TestRoutingReplay(TestBase):
    # Configure engine args with TP=8
    def setUp(self):
        self.engine_args = AsyncEngineArgs(
            model="Qwen/Qwen3-30B-A3B",
            tensor_parallel_size=8,
            enable_return_routed_experts=True,
        )

    async def qwen3_tp8_concurrent(self):
        """Test concurrent inference of Qwen3-30B-A3B with TP=8"""

        # Create AsyncLLM instance
        async_llm = AsyncLLM.from_engine_args(self.engine_args)

        try:
            # Configure sampling parameters
            sampling_params = SamplingParams(
                max_tokens=100,
                temperature=0.8,
                top_p=0.95,
                output_kind=RequestOutputKind.FINAL_ONLY,
            )

            NUM_REQUESTS = 10
            prompts = [
                f"Request {i}: Hello, please introduce yourself."
                for i in range(NUM_REQUESTS)
            ]

            async def generate_single(request_id: str, prompt: str):
                outputs = []
                async for output in async_llm.generate(
                        request_id=request_id,
                        prompt=prompt,
                        sampling_params=sampling_params
                ):
                    outputs.append(output)
                return outputs[-1]  # Return the final output

            # Create concurrent tasks
            tasks = [
                generate_single(f"request-{i}", prompts[i])
                for i in range(NUM_REQUESTS)
            ]

            # Wait for all tasks to finish
            results = await asyncio.gather(*tasks)

            # Validate results
            for i, result in enumerate(results):
                self.assertTrue(result.finished)
                self.assertTrue(len(result.outputs[0].text) > 0)
                print(result.outputs[0].routed_experts.shape)  # [seq_len, layer_num, topk]
        finally:
            # Cleanup
            async_llm.shutdown()

    @patch.dict(os.environ, {"OMP_NUM_THREADS": "1"})
    def test_routing_replay(self):

        asyncio.run(self.qwen3_tp8_concurrent())
