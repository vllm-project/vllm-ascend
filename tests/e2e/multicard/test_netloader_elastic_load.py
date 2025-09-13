#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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

import os
import time

import pytest

PORT = 29599
IP = "127.0.0.1"
COMM_NAME = f"{IP}:{PORT}"
TMP_PATH = "./p2p_test_param.pt"
SYNC_READY = "./p2p_loader_ready"


def run_loader():
    import torch

    from vllm_ascend.model_loader.netloader.executor.elastic_load import \
        P2PLoad
    torch.npu.set_device("npu:0")
    model = torch.nn.Linear(4, 2).to("npu:0")
    # First, set everything to zero, then wait for the sender to transmit.
    for p in model.parameters():
        p.data.zero_()
    loader = P2PLoad(COMM_NAME, IP, PORT)
    open(SYNC_READY, 'w').close()
    loaded_model = loader.load(model)
    if loaded_model is None:
        raise RuntimeError("Loader failed to load model!")
    # Save the final parameters of the loader
    torch.save({
        n: p.cpu()
        for n, p in loaded_model.named_parameters()
    }, TMP_PATH + ".recv")
    open(TMP_PATH + ".done", "w").close()
    time.sleep(2)
    os.remove(SYNC_READY)


def run_sender():
    import torch

    from vllm_ascend.model_loader.netloader.executor.elastic_load import \
        P2PSend
    torch.npu.set_device("npu:1")
    model = torch.nn.Linear(4, 2).to("npu:1")
    # Sender saves the original parameters.
    torch.save({n: p.cpu() for n, p in model.named_parameters()}, TMP_PATH)
    # Wait for the loader to be ready.
    for _ in range(40):
        if os.path.exists(SYNC_READY):
            break
        time.sleep(1)
    else:
        raise RuntimeError("Sender wait loader ready timeout!")
    sender = P2PSend(IP, PORT, COMM_NAME)
    sender.send(model, {})
    # Validate parameters
    ref_param = torch.load(TMP_PATH)
    # Wait for the loader to save received tensors.
    for _ in range(40):
        if os.path.exists(TMP_PATH + ".done"):
            break
        time.sleep(1)
    else:
        raise RuntimeError("Sender wait loader done timeout!")
    recv_param = torch.load(TMP_PATH + ".recv")
    for k in ref_param:
        print(f"Checking param: {k}")
        print(f"Sender param: {ref_param[k].flatten()[:5]}")
        print(f"Loader param: {recv_param[k].flatten()[:5]}")
        assert torch.allclose(ref_param[k], recv_param[k],
                              atol=1e-5), f"Param {k} mismatch!"
    try:
        os.remove(TMP_PATH)
        os.remove(TMP_PATH + ".recv")
        os.remove(TMP_PATH + ".done")
    except FileNotFoundError:
        print("File does not exist, skip to delete")


def test_p2p_auto_mp():
    from multiprocessing import Process, set_start_method
    set_start_method("spawn", force=True)
    # Start the loader and sender processes
    pl = Process(target=run_loader)
    ps = Process(target=run_sender)
    pl.start()
    ps.start()
    pl.join()
    ps.join()
    assert pl.exitcode == 0
    assert ps.exitcode == 0


if __name__ == "__main__":
    pytest.main()
