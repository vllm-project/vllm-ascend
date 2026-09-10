#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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

import time
from urllib.error import HTTPError

import requests
from vllm.logger import logger
from vllm.utils.network_utils import get_ip

REQUEST_TIMEOUT_SEC = 10.0
HEARTBEAT_LOG_EVERY_N = 4


def get_local_seed_key(
    disaggregation_mode: str,
    node_rank: int,
    tp_rank: int,
    model_url: str,
    model_deploy_strategy_name: str,
    seed_key_separator: str = "$",
    is_draft_worker: bool = False,
    pp_rank: int | None = None,
    ep_rank: int | None = None,
) -> str:
    if not model_url or not model_deploy_strategy_name:
        err_msg = (
            f"RFork seed key is not set: model_url={model_url!r}, "
            f"model_deploy_strategy_name={model_deploy_strategy_name!r}. "
            "Ensure model_loader_extra_config contains "
            "`model_url` and `model_deploy_strategy_name`, or set "
            "MODEL_URL and MODEL_DEPLOY_STRATEGY_NAME."
        )
        logger.error(err_msg)
        raise RuntimeError(err_msg)

    seed_key = f"{model_url}{seed_key_separator}{model_deploy_strategy_name}"
    key_parts = [disaggregation_mode, str(node_rank)]
    if pp_rank is not None:
        key_parts.append(f"pp{pp_rank}")
    key_parts.append(str(tp_rank))
    if ep_rank is not None:
        key_parts.append(f"ep{ep_rank}")
    if is_draft_worker:
        key_parts.append("draft")
    return f"{seed_key}{seed_key_separator}{seed_key_separator.join(key_parts)}"


class RForkSeedProtocol:
    def __init__(
        self,
        *,
        disaggregation_mode: str,
        node_rank: int,
        tp_rank: int,
        scheduler_url: str,
        model_url: str,
        model_deploy_strategy_name: str,
        seed_key_separator: str = "$",
        is_draft_worker: bool = False,
        pp_rank: int | None = None,
        ep_rank: int | None = None,
    ):
        self.disaggregation_mode = disaggregation_mode
        self.node_rank = node_rank
        self.tp_rank = tp_rank
        self.pp_rank = pp_rank
        self.ep_rank = ep_rank
        self.scheduler_url = scheduler_url
        self.model_url = model_url
        self.model_deploy_strategy_name = model_deploy_strategy_name
        self.seed_key_separator = seed_key_separator
        self.is_draft_worker = is_draft_worker

        self._local_seed_key = get_local_seed_key(
            disaggregation_mode=self.disaggregation_mode,
            node_rank=self.node_rank,
            tp_rank=self.tp_rank,
            model_url=self.model_url,
            model_deploy_strategy_name=self.model_deploy_strategy_name,
            seed_key_separator=self.seed_key_separator,
            is_draft_worker=self.is_draft_worker,
            pp_rank=self.pp_rank,
            ep_rank=self.ep_rank,
        )

    def get_local_seed_key(self) -> str:
        return self._local_seed_key

    @staticmethod
    def _request_timeout_sec() -> float:
        return REQUEST_TIMEOUT_SEC

    def _ensure_scheduler_url_set(self) -> None:
        if not self.scheduler_url:
            raise RuntimeError(
                "rfork_scheduler_url is not set. Set it through model_loader_extra_config or RFORK_SCHEDULER_URL."
            )

    def get_seed(self):
        try:
            self._ensure_scheduler_url_set()
            seed_key = self.get_local_seed_key()
            response = requests.get(
                f"{self.scheduler_url}/get_seed",
                headers={
                    "SEED_KEY": seed_key,
                },
                timeout=self._request_timeout_sec(),
            )
            if response.status_code != 200:
                raise RuntimeError(
                    f"Failed to get seed from the planner, {response.status_code}, seed_key={seed_key!r}"
                )

            seed_ip = response.headers.get("SEED_IP")
            seed_port = response.headers.get("SEED_PORT")
            user_id = response.headers.get("USER_ID")
            seed_rank = response.headers.get("SEED_RANK")
            logger.debug(
                "seed_ip: %s, seed_port: %s, user_id: %s, seed_rank: %s",
                seed_ip,
                seed_port,
                user_id,
                seed_rank,
            )
            return {
                "seed_ip": seed_ip,
                "seed_port": seed_port,
                "user_id": user_id,
                "seed_rank": seed_rank,
            }

        except RuntimeError as e:
            logger.warning("get_seed from scheduler RuntimeError: %s", e)
            return None
        except HTTPError as e:
            logger.exception("get_seed from scheduler HTTPError: %s", e)
            return None
        except Exception as e:
            logger.exception("get_seed from scheduler Exception: %s", e)
            return None

    def release_seed(self, seed) -> bool:
        try:
            self._ensure_scheduler_url_set()
            user_id = seed["user_id"]
            seed_ip = seed["seed_ip"]
            seed_port = str(seed["seed_port"])
            seed_rank = str(seed["seed_rank"])

            response = requests.post(
                f"{self.scheduler_url}/put_seed",
                headers={
                    "SEED_IP": seed_ip,
                    "SEED_PORT": seed_port,
                    "USER_ID": user_id,
                    "SEED_RANK": seed_rank,
                },
                timeout=self._request_timeout_sec(),
            )

            if response.status_code != 200:
                raise RuntimeError(f"Failed to release seed to the planner, {response.status_code}")
            return True
        except RuntimeError as e:
            logger.exception("release_seed to planner RuntimeError: %s", e)
            return False
        except HTTPError as e:
            logger.exception("release_seed to planner HTTPError: %s", e)
            return False
        except Exception as e:
            logger.exception("release_seed to planner Exception: %s", e)
            return False

    def report_seed(self, port: int, sleep_interval: int = 30):
        heartbeat_idx = 0
        log_every_n = HEARTBEAT_LOG_EVERY_N
        try:
            self._ensure_scheduler_url_set()
            seed_ip = get_ip()
            seed_key = self.get_local_seed_key()
            logger.debug("[rfork_heartbeat] reporting seed key: %s", seed_key)
        except Exception as e:
            logger.exception("report_seed setup Exception: %s", e)
            return

        while True:
            heartbeat_idx += 1
            result = False
            try:
                response = requests.post(
                    f"{self.scheduler_url}/add_seed",
                    headers={
                        "SEED_KEY": seed_key,
                        "SEED_IP": seed_ip,
                        "SEED_PORT": str(port),
                        "SEED_RANK": str(self.tp_rank),
                        "SEED_REFCNT": str(0),
                    },
                    timeout=self._request_timeout_sec(),
                )
                if response.status_code == 200:
                    result = True
            except HTTPError as e:
                logger.warning("report_seed to planner HTTPError: %s", e)
            except Exception as e:
                logger.warning("report_seed to planner Exception: %s", e)

            # Keep heartbeat frequency unchanged, but reduce log noise.
            # Always print failures immediately; keep success in debug logs.
            if result:
                if heartbeat_idx % log_every_n == 0:
                    logger.debug(
                        "[rfork_heartbeat] report seed to planner result: %s (%d/%d), seed_key=%s",
                        result,
                        heartbeat_idx % log_every_n if heartbeat_idx % log_every_n != 0 else log_every_n,
                        log_every_n,
                        seed_key,
                    )
            else:
                logger.warning(
                    "[rfork_heartbeat] report seed to planner result: %s (%d/%d), seed_key=%s",
                    result,
                    heartbeat_idx % log_every_n if heartbeat_idx % log_every_n != 0 else log_every_n,
                    log_every_n,
                    seed_key,
                )
            time.sleep(sleep_interval)
