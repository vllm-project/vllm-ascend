# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import pytest
import _pytest.terminal

_original_get_line = _pytest.terminal._get_line_with_reprcrash_message
_fail_msgs = {}


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    rep = outcome.get_result()
    if rep.when == "call" and rep.failed:
        msg = None
        try:
            if isinstance(rep.longrepr, str):
                msg = rep.longrepr
            else:
                msg = rep.longrepr.reprcrash.message
        except AttributeError:
            pass
        if msg and msg.startswith("Failed: "):
            msg = msg[len("Failed: "):]
        _fail_msgs[rep.nodeid] = msg


def _custom_get_line(config, rep, tw, word_markup):
    if hasattr(rep, 'when') and rep.when == "call" and rep.failed:
        fail_msg = _fail_msgs.get(rep.nodeid)
        if fail_msg:
            verbose_word, verbose_markup = rep._get_verbose_word_with_markup(
                config, word_markup
            )
            word = tw.markup(verbose_word, **verbose_markup)
            return f"{word} {fail_msg}"
    return _original_get_line(config, rep, tw, word_markup)


_pytest.terminal._get_line_with_reprcrash_message = _custom_get_line
