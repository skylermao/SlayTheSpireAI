"""Single import point for the compiled `slaythespire` module.

Every other module imports the sim through here (`from ._sts import sts`) so the
build-dir path insertion lives in exactly one place.
"""

import os
import sys

_BUILD_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "sts_lightspeed", "build")
)
if _BUILD_DIR not in sys.path:
    sys.path.insert(0, _BUILD_DIR)

import slaythespire as sts  # noqa: E402

__all__ = ["sts"]
