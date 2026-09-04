"""Every training/eval run gets a unique run_id, a results dir, and a log file,
so results stay traceable back to the exact config and code state that
produced them."""
from __future__ import annotations

import logging
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fwi_attn.config import Config, save_config


def make_run_id(prefix: str = "run") -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}_{ts}_{uuid.uuid4().hex[:8]}"


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def init_run(config: Config, run_name: str = "run") -> tuple[Path, logging.Logger, str]:
    run_id = make_run_id(run_name)
    results_dir = Path(config.paths.results_dir) / run_id
    log_dir = Path(config.paths.log_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(run_id)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    fh = logging.FileHandler(log_dir / f"{run_id}.log")
    sh = logging.StreamHandler(sys.stdout)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh.setFormatter(fmt)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)

    logger.info(f"run_id={run_id} git_commit={_git_commit()}")
    save_config(config, results_dir / "config.yaml")

    return results_dir, logger, run_id
