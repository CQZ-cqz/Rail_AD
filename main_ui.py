import argparse
import logging
import sys
from pathlib import Path

import yaml
from PyQt5.QtWidgets import QApplication

from ui.main_window import MainWindow

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def setup_logging(cfg: dict) -> None:
    log_cfg = cfg.get("logging", {})
    level_name = str(log_cfg.get("level", "INFO")).upper()
    level = getattr(logging, level_name, logging.INFO)

    handlers = [logging.StreamHandler()]
    log_file = log_cfg.get("log_file")
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path, encoding="utf-8"))

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="轨道检测系统 GUI 入口")
    parser.add_argument("--config", default="config.yaml", help="配置文件路径")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = str(Path(args.config).resolve())
    cfg = load_config(config_path)
    setup_logging(cfg)
    logger.info("GUI starting with config: %s", config_path)

    app = QApplication(sys.argv)
    window = MainWindow(cfg=cfg, config_path=config_path)
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
