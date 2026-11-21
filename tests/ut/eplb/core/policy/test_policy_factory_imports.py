# SPDX-License-Identifier: Apache-2.0
"""Import-isolation tests for PolicyFactory lazy policy loading.

Run in subprocesses so assertions about sys.modules are not affected by
policy modules imported by other test files in the same pytest process.
"""

import subprocess
import sys

LAZY_IMPORT_PROBE = """\
import sys

import vllm_ascend.eplb.core.policy.policy_factory

eagerly_loaded = sorted(
    module
    for module in sys.modules
    if module.startswith("vllm_ascend.eplb.core.policy.")
    and not module.endswith((".policy_factory", ".policy_abstract"))
)
assert not eagerly_loaded, eagerly_loaded
print("LAZY_OK")
"""

FALLBACK_PROBE = """\
import vllm_ascend.eplb.core.policy.policy_factory as factory

fallback = factory.PolicyFactory.generate_policy(99)
assert type(fallback).__name__ == "RandomLoadBalance", type(fallback).__name__
explicit = factory.PolicyFactory.generate_policy(0)
assert type(explicit).__name__ == "RandomLoadBalance", type(explicit).__name__
print("FALLBACK_OK")
"""


def _run_probe(source: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-c", source], capture_output=True, text=True, timeout=300)


def test_importing_policy_factory_does_not_load_policy_implementations():
    result = _run_probe(LAZY_IMPORT_PROBE)
    assert result.returncode == 0, result.stdout + result.stderr


def test_unknown_policy_type_falls_back_to_random_load_balance():
    result = _run_probe(FALLBACK_PROBE)
    assert result.returncode == 0, result.stdout + result.stderr
