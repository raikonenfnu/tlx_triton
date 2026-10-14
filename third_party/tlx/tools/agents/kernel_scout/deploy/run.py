"""Cron entry point: execute a measurement, then publish only its completed report."""

import argparse
import json
import shlex
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def commands(config, mode, pointer):
    command = [
        config["python"],
        str(ROOT / "appctl.py"), "run", "--mode", mode, "--targets", *config["targets"], "--state", config["state"],
        "--python", config["python"], "--pythonpath",
        config.get("pythonpath", ""), "--gpu",
        str(config.get("gpu", 0)), "--provider",
        config.get("provider", "codex"), "--report-pointer",
        str(pointer), "--execute"
    ]
    for key in ("context", "runtimes", "model"):
        if config.get(key):
            command.extend(["--" + key, config[key]])
    for key in ("max_candidates", "rounds", "agent_timeout", "worker_timeout", "gpu_wait_seconds", "gpu_poll_seconds"):
        if key in config:
            command.extend(["--" + key.replace("_", "-"), str(config[key])])
    if config.get("aiter_search", False):
        command.append("--aiter-search")
    return command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mode", choices=["weekly", "nightly"], required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    with tempfile.TemporaryDirectory(prefix="scout-cron-") as directory:
        pointer = Path(directory) / "report.json"
        command = commands(config, args.mode, pointer)
        if not args.execute:
            print(shlex.join(command))
            print("After success: appctl.py publish --report <completed report> --repo " + config["repository"] +
                  " --execute")
            return
        refresh = config.get("refresh_command")
        if refresh:
            if not isinstance(refresh, list) or not refresh or not all(isinstance(v, str) for v in refresh):
                raise ValueError("refresh_command must be an argument list")
            subprocess.run(refresh, check=True, timeout=config.get("refresh_timeout", 3600))
        subprocess.run(command, check=True)
        report = json.loads(pointer.read_text())["report"]
        subprocess.run([
            config["python"],
            str(ROOT / "appctl.py"), "publish", "--report", report, "--repo", config["repository"], "--execute"
        ], check=True)


if __name__ == "__main__":
    main()
