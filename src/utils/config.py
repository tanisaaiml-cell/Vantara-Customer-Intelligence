"""Configuration, artifact paths and structured logs."""

import json
import logging
import os
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]


def config() -> dict[str, Any]:
    """Load reproducible YAML settings."""
    return yaml.safe_load(Path(os.getenv("CONFIG_PATH", ROOT / "config/config.yaml")).read_text())


def artifacts() -> Path:
    """Return the environment-configurable artifact directory."""
    return Path(os.getenv("MODEL_ARTIFACT_PATH", ROOT / "models_artifacts"))


def log(event: str, **fields: Any) -> None:
    """Emit a structured event without customer records."""
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(message)s")
    logging.getLogger("vantara").info(json.dumps({"event": event, **fields}, default=str))
