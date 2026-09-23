"""Configuration loader for causal_rs.

Priority (highest to lowest):
  1. CLI arguments:  ``--config.key=value``
  2. Env variables:  ``CAUSAL_RS_KEY=value``
  3. YAML config file at ``<project_root>/config.yaml``
  4. Hardcoded defaults (below)

All module-level constants (``SEED``, ``DATA_DIR``, …) are kept for backward
compatibility — existing ``from src.config import X`` statements continue to
work unchanged.
"""

import os
import sys
from pathlib import Path
from typing import Any

import yaml

# ---------------------------------------------------------------------------
# Project root discovery
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
_DEFAULTS: dict[str, Any] = {
    "seed": 42,
    "dataset": "small",
    "neg_ratio": 4,
    "pca_components": 32,
    "title_embed_dim": 768,
    "title_embed_model": "sentence-transformers/all-mpnet-base-v2",
    "entity_embed_dim": 100,
    "split_ratios": [0.70, 0.15, 0.15],
    "max_behavior_rows": 5000,
    "gpu_device": 0,
    "gpu_enabled": True,
    "gpu_batch_size": 4096,
}

# --- Directory paths (computed from PROJECT_ROOT) ---
_DEFAULTS.update({
    "data_dir": PROJECT_ROOT / "data",
    "notebooks_dir": PROJECT_ROOT / "notebooks",
    "raw_dir": PROJECT_ROOT / "data" / "raw",
    "interim_dir": PROJECT_ROOT / "data" / "interim",
    "processed_dir": PROJECT_ROOT / "data" / "processed",
    "raw_small_dir": PROJECT_ROOT / "data" / "raw" / "MIND-small",
    "raw_small_train_dir": PROJECT_ROOT / "data" / "raw" / "MIND-small" / "train",
    "raw_small_dev_dir": PROJECT_ROOT / "data" / "raw" / "MIND-small" / "dev",
    "raw_small_test_dir": PROJECT_ROOT / "data" / "raw" / "MIND-small" / "test",
    "raw_large_dir": PROJECT_ROOT / "data" / "raw" / "MIND-large",
    "raw_large_train_dir": PROJECT_ROOT / "data" / "raw" / "MIND-large" / "train",
    "raw_large_dev_dir": PROJECT_ROOT / "data" / "raw" / "MIND-large" / "dev",
    "raw_large_test_dir": PROJECT_ROOT / "data" / "raw" / "MIND-large" / "test",
})


# ---------------------------------------------------------------------------
# Load helpers
# ---------------------------------------------------------------------------
def _parse_value(value: str) -> Any:
    """Parse a string from env/CLI into a Python literal."""
    lowered = value.lower()
    if lowered in ("true", "yes", "1"):
        return True
    if lowered in ("false", "no", "0"):
        return False
    if lowered in ("null", "none", ""):
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def _load_env(prefix: str = "CAUSAL_RS_") -> dict[str, Any]:
    config: dict[str, Any] = {}
    for key, value in os.environ.items():
        if key.startswith(prefix):
            config_key = key[len(prefix):].lower()
            config[config_key] = _parse_value(value)
    return config


def _load_cli() -> dict[str, Any]:
    config: dict[str, Any] = {}
    for arg in sys.argv[1:]:
        if arg.startswith("--config."):
            _, rest = arg.split("--config.", 1)
            if "=" in rest:
                key, value = rest.split("=", 1)
                config[key] = _parse_value(value)
    return config


# ---------------------------------------------------------------------------
# Assemble final config
# ---------------------------------------------------------------------------
def _build_config() -> dict[str, Any]:
    cfg = dict(_DEFAULTS)

    # Layer 1: YAML
    yaml_cfg = _load_yaml(CONFIG_PATH)
    cfg.update(yaml_cfg)

    # Layer 2: env vars
    env_cfg = _load_env()
    cfg.update(env_cfg)

    # Layer 3: CLI args
    cli_cfg = _load_cli()
    cfg.update(cli_cfg)

    # Coerce max_behavior_rows: None / 0 → sys.maxsize (unlimited)
    max_rows = cfg.get("max_behavior_rows")
    if max_rows is None or max_rows == 0:
        cfg["max_behavior_rows"] = sys.maxsize

    # Recompute derived dir paths if data_dir was overridden
    data_dir = cfg["data_dir"]
    if isinstance(data_dir, str):
        data_dir = Path(data_dir)
    cfg["data_dir"] = data_dir
    if data_dir != _DEFAULTS["data_dir"] or any(
        k in yaml_cfg or k in env_cfg or k in cli_cfg
        for k in ("data_dir", "raw_dir", "interim_dir", "processed_dir",
                  "raw_small_dir", "raw_small_train_dir", "raw_small_dev_dir",
                  "raw_small_test_dir", "raw_large_dir", "raw_large_train_dir",
                  "raw_large_dev_dir", "raw_large_test_dir")
    ):
        cfg.setdefault("raw_dir", data_dir / "raw")
        cfg.setdefault("interim_dir", data_dir / "interim")
        cfg.setdefault("processed_dir", data_dir / "processed")
        cfg.setdefault("raw_small_dir", cfg["raw_dir"] / "MIND-small")
        cfg.setdefault("raw_small_train_dir", cfg["raw_small_dir"] / "train")
        cfg.setdefault("raw_small_dev_dir", cfg["raw_small_dir"] / "dev")
        cfg.setdefault("raw_small_test_dir", cfg["raw_small_dir"] / "test")
        cfg.setdefault("raw_large_dir", cfg["raw_dir"] / "MIND-large")
        cfg.setdefault("raw_large_train_dir", cfg["raw_large_dir"] / "train")
        cfg.setdefault("raw_large_dev_dir", cfg["raw_large_dir"] / "dev")
        cfg.setdefault("raw_large_test_dir", cfg["raw_large_dir"] / "test")

    # Ensure paths are Path objects
    for key in ("data_dir", "raw_dir", "interim_dir", "processed_dir",
                "notebooks_dir",
                "raw_small_dir", "raw_small_train_dir",
                "raw_small_dev_dir", "raw_small_test_dir",
                "raw_large_dir", "raw_large_train_dir",
                "raw_large_dev_dir", "raw_large_test_dir"):
        val = cfg.get(key)
        if isinstance(val, str):
            cfg[key] = Path(val)

    return cfg


_config = _build_config()

# ---------------------------------------------------------------------------
# Module-level constants  (backward compat — importers keep working)
# ---------------------------------------------------------------------------
DATA_DIR: Path = _config["data_dir"]
RAW_DIR: Path = _config["raw_dir"]
INTERIM_DIR: Path = _config["interim_dir"]
PROCESSED_DIR: Path = _config["processed_dir"]
NOTEBOOKS_DIR: Path = _config["notebooks_dir"]
RAW_SMALL_DIR: Path = _config["raw_small_dir"]
RAW_SMALL_TRAIN_DIR: Path = _config["raw_small_train_dir"]
RAW_SMALL_DEV_DIR: Path = _config["raw_small_dev_dir"]
RAW_SMALL_TEST_DIR: Path = _config["raw_small_test_dir"]
RAW_LARGE_DIR: Path = _config["raw_large_dir"]
RAW_LARGE_TRAIN_DIR: Path = _config["raw_large_train_dir"]
RAW_LARGE_DEV_DIR: Path = _config["raw_large_dev_dir"]
RAW_LARGE_TEST_DIR: Path = _config["raw_large_test_dir"]

DATASET: str = _config["dataset"]
SEED: int = _config["seed"]
NEG_RATIO: int = _config["neg_ratio"]
PCA_COMPONENTS: int = _config["pca_components"]
TITLE_EMBED_DIM: int = _config["title_embed_dim"]
TITLE_EMBED_MODEL: str = _config["title_embed_model"]
ENTITY_EMBED_DIM: int = _config["entity_embed_dim"]
SPLIT_RATIOS = tuple(_config["split_ratios"])
MAX_BEHAVIOR_ROWS: int = _config["max_behavior_rows"]

GPU_DEVICE: int = _config["gpu_device"]
GPU_ENABLED: bool = _config["gpu_enabled"]
GPU_BATCH_SIZE: int = _config["gpu_batch_size"]

_DIRS = [
    DATA_DIR, RAW_DIR, INTERIM_DIR, PROCESSED_DIR, NOTEBOOKS_DIR,
    RAW_SMALL_DIR, RAW_SMALL_TRAIN_DIR, RAW_SMALL_DEV_DIR, RAW_SMALL_TEST_DIR,
    RAW_LARGE_DIR, RAW_LARGE_TRAIN_DIR, RAW_LARGE_DEV_DIR, RAW_LARGE_TEST_DIR,
]


# ---------------------------------------------------------------------------
# Public helper (for ad-hoc overrides in scripts)
# ---------------------------------------------------------------------------
def reload() -> None:
    """Re-read config.yaml + env + CLI and refresh all module-level constants.
    Useful when the YAML file is modified at runtime.
    """
    cfg = _build_config()
    globals().update({
        "DATA_DIR": cfg["data_dir"],
        "RAW_DIR": cfg["raw_dir"],
        "INTERIM_DIR": cfg["interim_dir"],
        "PROCESSED_DIR": cfg["processed_dir"],
        "NOTEBOOKS_DIR": cfg["notebooks_dir"],
        "RAW_SMALL_DIR": cfg["raw_small_dir"],
        "RAW_SMALL_TRAIN_DIR": cfg["raw_small_train_dir"],
        "RAW_SMALL_DEV_DIR": cfg["raw_small_dev_dir"],
        "RAW_SMALL_TEST_DIR": cfg["raw_small_test_dir"],
        "RAW_LARGE_DIR": cfg["raw_large_dir"],
        "RAW_LARGE_TRAIN_DIR": cfg["raw_large_train_dir"],
        "RAW_LARGE_DEV_DIR": cfg["raw_large_dev_dir"],
        "RAW_LARGE_TEST_DIR": cfg["raw_large_test_dir"],
        "DATASET": cfg["dataset"],
        "SEED": cfg["seed"],
        "NEG_RATIO": cfg["neg_ratio"],
        "PCA_COMPONENTS": cfg["pca_components"],
        "TITLE_EMBED_DIM": cfg["title_embed_dim"],
        "TITLE_EMBED_MODEL": cfg["title_embed_model"],
        "ENTITY_EMBED_DIM": cfg["entity_embed_dim"],
        "SPLIT_RATIOS": tuple(cfg["split_ratios"]),
        "MAX_BEHAVIOR_ROWS": cfg["max_behavior_rows"],
        "GPU_DEVICE": cfg["gpu_device"],
        "GPU_ENABLED": cfg["gpu_enabled"],
        "GPU_BATCH_SIZE": cfg["gpu_batch_size"],
        "_DIRS": [
            cfg["data_dir"], cfg["raw_dir"], cfg["interim_dir"],
            cfg["processed_dir"], cfg["notebooks_dir"],
            cfg["raw_small_dir"], cfg["raw_small_train_dir"],
            cfg["raw_small_dev_dir"], cfg["raw_small_test_dir"],
            cfg["raw_large_dir"], cfg["raw_large_train_dir"],
            cfg["raw_large_dev_dir"], cfg["raw_large_test_dir"],
        ],
    })
