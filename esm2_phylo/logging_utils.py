"""Consistent logging setup for all pipeline stages.

Replaces the bare ``print()`` calls used throughout the original scripts
with a standard logger, so verbosity can be controlled with ``-v/-q`` and
output is timestamped and greppable.
"""

from __future__ import annotations

import logging
import sys


def get_logger(name: str, verbose: bool = False) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        # Avoid duplicate handlers if get_logger is called more than once
        # (e.g. when a script is imported by run_pipeline.py).
        logger.setLevel(logging.DEBUG if verbose else logging.INFO)
        return logger

    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.propagate = False
    return logger
