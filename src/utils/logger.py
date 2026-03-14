"""
Structured logging for InsuranceCopilot AI.
"""

import logging
import sys
from pathlib import Path

from src.utils.config import LOGS_DIR


def setup_logger(
    name: str = "insurance_copilot",
    level: int = logging.INFO,
    log_to_file: bool = True,
) -> logging.Logger:
    """Create and configure a logger."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(level)
    fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    formatter = logging.Formatter(fmt)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    if log_to_file:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(LOGS_DIR / "insurance_copilot.log", encoding="utf-8")
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def get_logger(name: str = "insurance_copilot") -> logging.Logger:
    """Get existing logger or create one."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        return setup_logger(name)
    return logger
