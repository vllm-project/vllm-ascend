# SPDX-License-Identifier: Apache-2.0
"""Configuration rendering and result checks shared with the native runner.

These functions do not download inputs, create processes, or contact a service.
"""

import re


def render_dataset_config(content: str, dataset_path: str) -> str:
    return re.sub(r"path=.*", lambda _: f"path={dataset_path!r},", content)


def render_request_config(content: str, options: dict) -> str:
    fields = {
        "model": options["model"],
        "host_port": options["port"],
        "host_ip": options["host_ip"],
        "max_out_len": options["max_out_len"],
        "batch_size": options["batch_size"],
        "trust_remote_code": options.get("trust_remote_code", True),
    }
    for key, value in fields.items():
        pattern = key + (r"=.*" if key in ("model", "trust_remote_code") else r".*")
        content = re.sub(pattern, lambda _, key=key, value=value: f"{key}={value!r},", content)
    for key in ("top_p", "top_k", "seed", "min_p", "presence_penalty", "repetition_penalty"):
        if options.get(key):
            content = re.sub(
                r"ignore_eos.*", lambda _, key=key: f"ignore_eos=False,\n            {key}={options[key]!r},", content
            )
    if options.get("thinking"):
        content = re.sub(
            r"ignore_eos.*", 'ignore_eos=False,\n            chat_template_kwargs={"thinking": True},', content
        )
    if options["task_type"] == "performance":
        content = re.sub(r"path=.*", lambda _: f"path={options['model_path']!r},", content)
        content = re.sub(r"request_rate.*", lambda _: f"request_rate={options.get('request_rate', 0)!r},", content)
        content = re.sub(r"temperature.*", "temperature=0,", content)
        content = re.sub(r"ignore_eos.*", "ignore_eos=True,", content)
    if options["task_type"] in ("accuracy", "spec_decode"):
        content = re.sub(r"temperature.*", "temperature=0.6,", content)
    if options.get("temperature") is not None:
        content = re.sub(r"temperature.*", lambda _: f"temperature={options['temperature']!r},", content)
    if options.get("no_pred"):
        content = re.sub(r"pred_postprocessor.*", "#pred_postprocessor", content)
    return content


def verify_performance(
    result_json, baseline, threshold=0.97, *, input_throughput_threshold=None, tpot_threshold=None, tpot=None
):
    """The native runner's performance assertions, without result discovery/run."""
    output_throughput = result_json["Output Token Throughput"]["total"].replace("token/s", "")
    if not float(output_throughput) >= threshold * baseline:
        raise AssertionError(
            "Performance verification failed. "
            f"The current Output Token Throughput is {output_throughput} token/s, "
            f"which is not greater than or equal to {threshold} * baseline {baseline}."
        )
    if input_throughput_threshold is not None:
        input_throughput = str(result_json["Input Token Throughput"]["total"]).replace("token/s", "")
        if not float(input_throughput) >= float(input_throughput_threshold):
            raise AssertionError(
                f"Input Token Throughput verification failed. The current value is {input_throughput} token/s, "
                f"which is not greater than {input_throughput_threshold} token/s."
            )
    if tpot_threshold is not None:
        tpot = float(str(tpot).replace("ms", ""))
        if not tpot <= float(tpot_threshold):
            raise AssertionError(
                f"TPOT verification failed. The current TPOT is {tpot} ms, which is greater than {tpot_threshold} ms."
            )
