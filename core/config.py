"""
config.py – Configuration loader and derived-value helpers.

Usage
─────
    from config import load_config, compute_segment_length, setup_logging

    cfg = load_config("config.yaml")
    seg_len = compute_segment_length(cfg)
    setup_logging(cfg)
"""

import logging
from pathlib import Path
from typing import Any, Dict

import yaml

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# Loader
# ──────────────────────────────────────────────────────────────

def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load and validate the YAML configuration file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    _validate(cfg)
    logger.info("Configuration loaded from: %s", config_path)
    return cfg


def _validate(cfg: Dict[str, Any]) -> None:
    """Raise ValueError on missing or invalid required fields."""
    required_paths = [
        ("scanner", "trigger_mode"),
        ("track",   "total_length_mm"),
        ("roi",     "x_min"),
        ("roi",     "x_max"),
        ("roi",     "z_min"),
        ("roi",     "z_max"),
    ]
    for section, key in required_paths:
        if section not in cfg or key not in cfg[section]:
            raise ValueError(f"Missing required config field: {section}.{key}")

    mode = cfg["scanner"]["trigger_mode"]
    if mode not in ("encoder", "fixed_rate"):
        raise ValueError(
            f"Invalid scanner.trigger_mode '{mode}'. Must be 'encoder' or 'fixed_rate'."
        )

    if cfg["track"]["total_length_mm"] <= 0:
        raise ValueError("track.total_length_mm must be a positive number.")

    roi = cfg["roi"]
    if roi["x_min"] >= roi["x_max"]:
        raise ValueError("roi.x_min must be less than roi.x_max.")
    if roi["z_min"] >= roi["z_max"]:
        raise ValueError("roi.z_min must be less than roi.z_max.")


# ──────────────────────────────────────────────────────────────
# Derived values
# ──────────────────────────────────────────────────────────────

def compute_segment_length(cfg: Dict[str, Any]) -> float:
    """
    Return the physical length (mm) covered by one acquisition batch.

    Encoder mode
    ────────────
        length = scan_line_count × trigger_interval × (travel_per_rev_mm / resolution)

        Each encoder pulse moves the platform by (travel_per_rev_mm / resolution) mm.
        The scanner captures one profile every `trigger_interval` pulses, so each
        profile covers (trigger_interval × mm_per_pulse) mm along the track.
        With `scan_line_count` profiles per batch, the total coverage is as above.

    Fixed-rate mode (debug only)
    ────────────────────────────
        Returns `track.debug_segment_length_mm` because the physical distance
        covered per batch depends on train speed, which is unknown at config time.
    """
    mode = cfg["scanner"]["trigger_mode"]

    if mode == "encoder":
        enc = cfg["scanner"]["encoder"]
        odo = cfg["odometry"]
        mm_per_pulse = odo["travel_per_rev_mm"] / odo["resolution"]
        # length = enc["scan_line_count"] * enc["trigger_interval"] * mm_per_pulse  # 原始计算方式
        length = enc["scan_line_count"] * enc["trigger_interval"]                   # 单编码器
        logger.info("Segment length (encoder mode): %.2f mm", length)
        return length

    # fixed_rate
    length = cfg["track"].get("debug_segment_length_mm", 5000.0)
    logger.info("Segment length (fixed_rate / debug): %.2f mm", length)
    return length


# ──────────────────────────────────────────────────────────────
# Logging setup
# ──────────────────────────────────────────────────────────────

def setup_logging(cfg: Dict[str, Any]) -> None:
    """Configure the root logger from config.logging."""
    log_cfg  = cfg.get("logging", {})
    level    = getattr(logging, log_cfg.get("level", "INFO").upper(), logging.INFO)
    log_file = log_cfg.get("log_file", "rail_inspection.log")

    fmt = "%(asctime)s [%(levelname)-8s] %(name)s: %(message)s"
    handlers = [
        logging.StreamHandler(),
        logging.FileHandler(log_file, encoding="utf-8"),
    ]
    logging.basicConfig(level=level, format=fmt, handlers=handlers, force=True)
