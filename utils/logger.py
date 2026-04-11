"""
utils/logger.py
───────────────
Structured logging using loguru.
Every agent logs its actions so you can trace the full decision path —
great for debugging AND for showing in interviews.
"""

import sys
from pathlib import Path
from loguru import logger
from utils.config import config


def setup_logger() -> None:
    """Configure loguru with console + rotating file sink."""

    # Remove the default handler
    logger.remove()

    # ── Console output (coloured, human-readable) ─────────────────────────────
    logger.add(
        sys.stdout,
        format=(
            "<green>{time:HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{extra[agent]: <18}</cyan> | "
            "{message}"
        ),
        level="INFO",
        colorize=True,
    )

    # ── File output (JSON-structured, great for log analysis) ─────────────────
    log_path = Path(config.LOG_PATH)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger.add(
        log_path,
        format="{time} | {level} | {extra[agent]} | {message}",
        rotation="5 MB",      # New file every 5 MB
        retention="7 days",   # Keep logs for a week
        level="DEBUG",
    )


def get_logger(agent_name: str):
    """
    Return a logger bound to a specific agent name.
    Usage:
        log = get_logger("RouterAgent")
        log.info("Routing to RAG agent")
    """
    setup_logger()
    return logger.bind(agent=agent_name)
