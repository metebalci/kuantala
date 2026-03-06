"""Logging and progress utilities."""

from __future__ import annotations

import logging

from rich.console import Console
from rich.logging import RichHandler

console = Console()


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=logging.WARNING,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )
    # Only set our loggers to the desired level; keep third-party loggers quiet
    logging.getLogger("kuantala").setLevel(level)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
