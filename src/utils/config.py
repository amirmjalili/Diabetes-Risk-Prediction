"""Configuration loading utilities."""

from pathlib import Path
from typing import Any, Dict

import yaml


def get_project_root() -> Path:
    """Return absolute path to project root (directory containing configs/)."""
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "configs" / "config.yaml").exists():
            return parent
    # Fallback for installed package or different layout
    return Path.cwd()


def load_config(config_path: str | Path | None = None) -> Dict[str, Any]:
    """Load YAML configuration file.

    Parameters
    ----------
    config_path : str or Path, optional
        Path to config file. Defaults to configs/config.yaml relative to project root.

    Returns
    -------
    dict
        Parsed configuration dictionary.
    """
    if config_path is None:
        config_path = get_project_root() / "configs" / "config.yaml"
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Resolve relative paths against project root
    root = get_project_root()
    if "paths" in config:
        for key, value in config["paths"].items():
            if not Path(value).is_absolute():
                config["paths"][key] = str(root / value)

    return config
