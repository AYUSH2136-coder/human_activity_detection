"""
Logger utility — structured console + file logging.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime


def get_logger(name: str, log_dir: str | Path = "outputs/logs", level: int = logging.INFO) -> logging.Logger:
    """
    Create and return a logger with both console and file handlers.

    Parameters
    ----------
    name : str
        Logger name (typically __name__ of the calling module).
    log_dir : str | Path
        Directory where log files will be written.
    level : int
        Logging level (default: INFO).

    Returns
    -------
    logging.Logger
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # Already configured

    logger.setLevel(level)

    # ── Formatter ─────────────────────────────────────────────────────────────
    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    date_fmt = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(fmt, datefmt=date_fmt)

    # ── Console handler ───────────────────────────────────────────────────────
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(formatter)
    logger.addHandler(console)

    # ── File handler ──────────────────────────────────────────────────────────
    log_file = log_dir / f"{datetime.now().strftime('%Y%m%d')}_had.log"
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.propagate = False
    return logger
