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
# This file is a part of the vllm-ascend project.

from unittest.mock import ANY, MagicMock, patch

from tests.ut.base import TestBase
from vllm_ascend.patch.worker.patch_common.patch_spec_decode_worker import \
    SpecDecodeWorker
from vllm_ascend.worker.draft_model_runner import TP1DraftModelRunner


class TestSpecDecodeWorkerPatch(TestBase):

    COMMON_KWARGS = {
        'disable_mqa_scorer': False,
        'disable_by_batch_size': None,
        'draft_token_acceptance_method': 'rejection_sampler',
        'typical_acceptance_sampler_posterior_threshold': 0.5,
        'typical_acceptance_sampler_posterior_alpha': 0.5,
        'disable_logprobs': False,
        'disable_log_stats': False,
        'num_speculative_tokens': 5
    }

    def setUp(self):
        patcher_spec = patch(
            'vllm_ascend.patch.worker.patch_common.patch_spec_decode_worker.SpecDecodeWorker'
        )
        patcher_smaller_tp = patch(
            'vllm_ascend.patch.worker.patch_common.patch_spec_decode_worker.SmallerTpProposerWorker'
        )

        self.mock_spec_worker = patcher_spec.start()
        self.mock_smaller_tp = patcher_smaller_tp.start()

        self.addCleanup(patcher_spec.stop)
        self.addCleanup(patcher_smaller_tp.stop)

    def _create_scorer_worker(self, device_type='cpu', max_model_len=100):
        scorer_worker = MagicMock()
        scorer_worker.parallel_config.tensor_parallel_size = 2
        scorer_worker.device_config.device.type = device_type
        scorer_worker.model_runner.attn_backend.get_name.return_value = 'FLASH_ATTN'
        scorer_worker.model_config.max_model_len = max_model_len
        scorer_worker.model_config.enforce_eager = True
        return scorer_worker

    def _create_draft_config(self, model_type, tp_size=2):
        draft_config = MagicMock()
        draft_config.model_config.hf_config.model_type = model_type
        draft_config.model_config.max_model_len = 50 if model_type == 'ngram' else 100
        draft_config.parallel_config.tensor_parallel_size = tp_size
        draft_config.parallel_config.expert_parallel_size = 1
        draft_config.parallel_config.expert_tensor_parallel_size = 1
        return draft_config

    @patch(
        'vllm_ascend.patch.worker.patch_common.patch_spec_decode_worker.NGramWorker'
    )
    def test_create_worker_ngram(self, mock_ngram):
        scorer = self._create_scorer_worker('cpu', 100)
        draft_config = self._create_draft_config('ngram', 2)

        SpecDecodeWorker.create_worker(scorer_worker=scorer,
                                       draft_worker_kwargs={
                                           'vllm_config': draft_config,
                                           'ngram_prompt_lookup_max': 5,
                                           'ngram_prompt_lookup_min': 1
                                       },
                                       **self.COMMON_KWARGS)

        mock_ngram.assert_called_once()
        self.mock_spec_worker.assert_called_once_with(
            mock_ngram.return_value,
            scorer,
            disable_mqa_scorer=True,
            disable_logprobs=False,
            disable_log_stats=False,
            disable_by_batch_size=None,
            spec_decode_sampler=ANY,
            allow_zero_draft_token_step=True,
            enable_lm_head_weight_load=False,
            num_spec_prefill_steps=1)

    @patch(
        'vllm_ascend.patch.worker.patch_common.patch_spec_decode_worker.MLPSpeculatorWorker'
    )
    def test_create_worker_mlp_speculator(self, mock_mlp):
        scorer = self._create_scorer_worker('cuda', 100)
        draft_config = self._create_draft_config('mlp_speculator', 1)

        SpecDecodeWorker.create_worker(scorer_worker=scorer,
                                       draft_worker_kwargs={
                                           'vllm_config': draft_config,
                                           'ngram_prompt_lookup_max': 0,
                                           'ngram_prompt_lookup_min': 0
                                       },
                                       **self.COMMON_KWARGS)

        mock_mlp.assert_called_once()
        self.mock_smaller_tp.maybe_wrap_worker.assert_called_once()
        self.assertEqual(
            self.mock_spec_worker.call_args[1]['num_spec_prefill_steps'], 1)

    @patch(
        'vllm_ascend.patch.worker.patch_common.patch_spec_decode_worker.MedusaWorker'
    )
    def test_create_worker_medusa(self, mock_medusa):
        scorer = self._create_scorer_worker('cuda', 100)
        draft_config = self._create_draft_config('medusa', 1)

        SpecDecodeWorker.create_worker(scorer_worker=scorer,
                                       draft_worker_kwargs={
                                           'vllm_config': draft_config,
                                           'ngram_prompt_lookup_max': 0,
                                           'ngram_prompt_lookup_min': 0
                                       },
                                       **self.COMMON_KWARGS)

        mock_medusa.assert_called_once()
        self.mock_smaller_tp.maybe_wrap_worker.assert_called_once()
        self.assertEqual(
            self.mock_spec_worker.call_args[1]['num_spec_prefill_steps'], 1)

    @patch(
        'vllm_ascend.patch.worker.patch_common.patch_spec_decode_worker.MultiStepWorker'
    )
    def test_create_worker_multistep_tp1(self, mock_multistep):
        scorer = self._create_scorer_worker('cuda', 100)
        draft_config = self._create_draft_config('other', 1)

        SpecDecodeWorker.create_worker(scorer_worker=scorer,
                                       draft_worker_kwargs={
                                           'vllm_config': draft_config,
                                           'ngram_prompt_lookup_max': 0,
                                           'ngram_prompt_lookup_min': 0
                                       },
                                       **self.COMMON_KWARGS)

        mock_multistep.assert_called_once()
        self.assertEqual(mock_multistep.call_args[1]['model_runner_cls'],
                         TP1DraftModelRunner)
        self.mock_smaller_tp.maybe_wrap_worker.assert_called_once()
        self.assertEqual(
            self.mock_spec_worker.call_args[1]['allow_zero_draft_token_step'],
            True)

    @patch(
        'vllm_ascend.patch.worker.patch_common.patch_spec_decode_worker.MultiStepWorker'
    )
    def test_create_worker_multistep_tp2(self, mock_multistep):
        scorer = self._create_scorer_worker('cuda', 100)
        draft_config = self._create_draft_config('other', 2)  # TP=2

        SpecDecodeWorker.create_worker(scorer_worker=scorer,
                                       draft_worker_kwargs={
                                           'vllm_config': draft_config,
                                           'ngram_prompt_lookup_max': 0,
                                           'ngram_prompt_lookup_min': 0
                                       },
                                       **self.COMMON_KWARGS)

        mock_multistep.assert_called_once()
        self.assertNotEqual(
            mock_multistep.call_args[1].get('model_runner_cls'),
            TP1DraftModelRunner)
        self.assertEqual(
            self.mock_spec_worker.call_args[1]['allow_zero_draft_token_step'],
            False)

    @patch(
        'vllm_ascend.patch.worker.patch_common.patch_spec_decode_worker.MultiStepWorker'
    )
    def test_create_worker_eagle(self, mock_multistep):
        scorer = self._create_scorer_worker('cuda', 100)
        draft_config = self._create_draft_config('eagle', 1)

        SpecDecodeWorker.create_worker(scorer_worker=scorer,
                                       draft_worker_kwargs={
                                           'vllm_config': draft_config,
                                           'ngram_prompt_lookup_max': 0,
                                           'ngram_prompt_lookup_min': 0
                                       },
                                       **self.COMMON_KWARGS)

        mock_multistep.assert_called_once()
        self.assertEqual(
            self.mock_spec_worker.call_args[1]['enable_lm_head_weight_load'],
            True)

    @patch(
        'vllm_ascend.patch.worker.patch_common.patch_spec_decode_worker.MultiStepWorker'
    )
    def test_create_worker_deepseek_mtp(self, mock_multistep):
        scorer = self._create_scorer_worker('cuda', 100)
        draft_config = self._create_draft_config('deepseek_mtp', 1)
        draft_config.model_config.hf_config.n_predict = 3

        SpecDecodeWorker.create_worker(scorer_worker=scorer,
                                       draft_worker_kwargs={
                                           'vllm_config': draft_config,
                                           'ngram_prompt_lookup_max': 0,
                                           'ngram_prompt_lookup_min': 0
                                       },
                                       **self.COMMON_KWARGS)

        mock_multistep.assert_called_once()
        self.assertEqual(
            self.mock_spec_worker.call_args[1]['num_spec_prefill_steps'], 3)

    @patch(
        'vllm_ascend.patch.worker.patch_common.patch_spec_decode_worker.MultiStepWorker'
    )
    def test_create_worker_typical_acceptance(self, mock_multistep):
        scorer = self._create_scorer_worker('cuda', 100)
        draft_config = self._create_draft_config('other', 1)

        SpecDecodeWorker.create_worker(
            scorer_worker=scorer,
            draft_worker_kwargs={
                'vllm_config': draft_config,
                'ngram_prompt_lookup_max': 0,
                'ngram_prompt_lookup_min': 0
            },
            draft_token_acceptance_method='typical_acceptance_sampler',
            **{
                k: v
                for k, v in self.COMMON_KWARGS.items()
                if k != 'draft_token_acceptance_method'
            })

        sampler = self.mock_spec_worker.call_args[1]['spec_decode_sampler']
        self.assertEqual(type(sampler).__name__, 'TypicalAcceptanceSampler')

    @patch(
        'vllm_ascend.patch.worker.patch_common.patch_spec_decode_worker.MultiStepWorker'
    )
    def test_disable_mqa_scorer_conditions(self, mock_multistep):
        scorer = self._create_scorer_worker('cuda', 100)
        scorer.model_runner.attn_backend.get_name.return_value = 'OTHER_BACKEND'
        draft_config = self._create_draft_config('other', 1)

        SpecDecodeWorker.create_worker(scorer_worker=scorer,
                                       draft_worker_kwargs={
                                           'vllm_config': draft_config,
                                           'ngram_prompt_lookup_max': 0,
                                           'ngram_prompt_lookup_min': 0
                                       },
                                       **self.COMMON_KWARGS)
        self.assertEqual(
            self.mock_spec_worker.call_args[1]['disable_mqa_scorer'], True)

        scorer = self._create_scorer_worker('cuda', 100)
        scorer.model_runner.attn_backend.get_name.return_value = 'FLASH_ATTN'
        draft_config = self._create_draft_config('other', 1)
        draft_config.model_config.max_model_len = 50

        SpecDecodeWorker.create_worker(scorer_worker=scorer,
                                       draft_worker_kwargs={
                                           'vllm_config': draft_config,
                                           'ngram_prompt_lookup_max': 0,
                                           'ngram_prompt_lookup_min': 0
                                       },
                                       **self.COMMON_KWARGS)
        self.assertEqual(
            self.mock_spec_worker.call_args[1]['disable_mqa_scorer'], True)

        scorer = self._create_scorer_worker('cuda', 100)
        scorer.model_config.enforce_eager = False

        SpecDecodeWorker.create_worker(scorer_worker=scorer,
                                       draft_worker_kwargs={
                                           'vllm_config': draft_config,
                                           'ngram_prompt_lookup_max': 0,
                                           'ngram_prompt_lookup_min': 0
                                       },
                                       **self.COMMON_KWARGS)
        self.assertEqual(
            self.mock_spec_worker.call_args[1]['disable_mqa_scorer'], True)

    @patch(
        'vllm_ascend.patch.worker.patch_common.patch_spec_decode_worker.MultiStepWorker'
    )
    def test_eagle_tp_not_supported(self, mock_multistep):
        scorer = self._create_scorer_worker('cuda', 100)
        draft_config = self._create_draft_config('eagle', 2)

        with self.assertRaises(NotImplementedError):
            SpecDecodeWorker.create_worker(scorer_worker=scorer,
                                           draft_worker_kwargs={
                                               'vllm_config': draft_config,
                                               'ngram_prompt_lookup_max': 0,
                                               'ngram_prompt_lookup_min': 0
                                           },
                                           **self.COMMON_KWARGS)
