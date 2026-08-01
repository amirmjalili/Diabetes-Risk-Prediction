"""Logging configuration using loguru."""

import sys
from pathlib import Path

from loguru import logger


def setup_logging(level: str = "INFO", log_dir: str | Path | None = None) -> None:
    """Configure loguru logger for the application.

    Parameters
    ----------
    level : str
        Logging level (DEBUG, INFO, WARNING, ERROR).
    log_dir : str or Path, optional
        Directory for log files. If None, logs only to stderr.
    """
    logger.remove()
    logger.add(
        sys.stderr,
        level=level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        ),
        colorize=True,
    )

    if log_dir is not None:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        logger.add(
            log_path / "diabetes_risk_{time:YYYY-MM-DD}.log",
            level=level,
            rotation="10 MB",
            retention="30 days",
            compression="zip",
            format=(
                "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | "
                "{name}:{function}:{line} | {message}"
            ),
        )