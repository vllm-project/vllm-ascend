# SPDX-License-Identifier: Apache-2.0
"""Clean up an attached shared Engram reader if model construction fails."""

import logging
from functools import wraps


def close_engram_on_init_failure(reader_path):
    def decorate(init):
        @wraps(init)
        def wrapped(self, *args, **kwargs):
            try:
                return init(self, *args, **kwargs)
            except BaseException:
                target = self
                for name in reader_path:
                    target = getattr(target, name, None)
                    if target is None:
                        break
                if target is not None:
                    try:
                        target.close()
                    except BaseException:
                        logging.getLogger(__name__).exception("Failed to close shared Engram after model init failure")
                raise

        return wrapped

    return decorate
