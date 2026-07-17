"""Minimal YOLOX package bootstrap used by the tracker copy.

The original project re-exported a large utility surface here. For this
educational copy we only need the startup environment tweak that the tracker
relies on indirectly.
"""

from .utils.setup_env import configure_module

configure_module()

__version__ = "0.1.0"