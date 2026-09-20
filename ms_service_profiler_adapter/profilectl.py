#!/usr/bin/env python3
"""Run and control MS Service Profiler from an experiment YAML.

The YAML selects symbols at service startup. The generated JSON's enable field
is the runtime capture switch. No vLLM or model-specific module is imported.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml


NAME_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")


def atomic_text(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=path.name + ".", delete=False
    ) as handle:
        temp_path = Path(handle.name)
        handle.write(contents)
    os.replace(temp_path, path)


def atomic_json(path: Path, value: dict) -> None:
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def load_experiment(path: Path) -> tuple[dict, str, str]:
    raw = path.read_bytes()
    data = yaml.safe_load(raw)
    if not isinstance(data, dict):
        raise ValueError("experiment YAML must contain a mapping")
    name = data.get("name", path.stem)
    if not isinstance(name, str) or not NAME_PATTERN.fullmatch(name):
        raise ValueError("experiment name must contain only letters, digits, _, . or -")
    profile = data.get("profile", {})
    if not isinstance(profile, dict):
        raise ValueError("profile must be a mapping")
    symbols = data.get("symbols", {})
    if not isinstance(symbols, dict):
        raise ValueError("symbols must be a mapping")
    additions = symbols.get("add", [])
    if not isinstance(additions, list) or any(not isinstance(item, dict) or "symbol" not in item for item in additions):
        raise ValueError("symbols.add must be a list of mappings with symbol")
    return data, name, hashlib.sha256(raw).hexdigest()


def package_symbols() -> list:
    spec = importlib.util.find_spec("ms_service_profiler")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("ms_service_profiler is unavailable in this Python environment")
    base_path = Path(next(iter(spec.submodule_search_locations))) / "patcher/vllm/config/service_profiling_symbols.yaml"
    if not base_path.is_file():
        raise RuntimeError(f"installed profiler default symbols not found: {base_path}")
    parsed = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    if not isinstance(parsed, list):
        raise RuntimeError(f"installed profiler default symbols are not a YAML list: {base_path}")
    return parsed


def paths(args, name: str) -> tuple[Path, Path, Path]:
    node = args.node
    if not NAME_PATTERN.fullmatch(node):
        raise ValueError("node must contain only letters, digits, _, . or -")
    root = args.state_root or args.config.parent / ".profilectl"
    state_dir = root.resolve() / name / node
    return state_dir, state_dir / "ms_service_profiler_config.json", state_dir / "manifest.json"


def prepare(args) -> dict:
    data, name, digest = load_experiment(args.config)
    state_dir, config_path, manifest_path = paths(args, name)
    profile = dict(data.get("profile", {}))
    if "enable" in profile:
        raise ValueError("set capture enable with start/stop, not in experiment YAML")
    prof_dir = profile.get("prof_dir", "./profiles/{experiment}/{node}")
    if not isinstance(prof_dir, str):
        raise ValueError("profile.prof_dir must be a string")
    rendered_dir = Path(prof_dir.format(experiment=name, node=args.node))
    if not rendered_dir.is_absolute():
        rendered_dir = args.config.parent / rendered_dir
    profile["prof_dir"] = str(rendered_dir.resolve())
    profile["enable"] = 0

    additions = data.get("symbols", {}).get("add", [])
    symbols_path = None
    if additions:
        base = package_symbols()
        present = {entry.get("symbol") for entry in base if isinstance(entry, dict)}
        for entry in additions:
            if entry["symbol"] not in present:
                base.append(entry)
                present.add(entry["symbol"])
        symbols_path = state_dir / "service_profiling_symbols.yaml"
        atomic_text(symbols_path, yaml.safe_dump(base, allow_unicode=True, sort_keys=False))

    atomic_json(config_path, profile)
    manifest = {
        "experiment": name,
        "node": args.node,
        "yaml": str(args.config.resolve()),
        "yaml_sha256": digest,
        "config_path": str(config_path),
        "symbols_path": str(symbols_path) if symbols_path else None,
        "prof_dir": profile["prof_dir"],
    }
    atomic_json(manifest_path, manifest)
    return manifest


def load_prepared(args) -> tuple[dict, Path, dict]:
    _, name, digest = load_experiment(args.config)
    _, config_path, manifest_path = paths(args, name)
    if not manifest_path.is_file() or not config_path.is_file():
        raise RuntimeError("experiment is not prepared; launch or prepare it before start/stop")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("yaml_sha256") != digest:
        raise RuntimeError("experiment YAML changed after preparation; restart the service with this YAML")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    return manifest, config_path, config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    for action in ("prepare", "launch", "start", "stop", "status"):
        command = sub.add_parser(action)
        command.add_argument("--config", "-c", type=Path, required=True)
        command.add_argument("--node", required=True)
        command.add_argument("--state-root", type=Path)
        if action == "launch":
            command.add_argument("command", nargs=argparse.REMAINDER)
    parse_cmd = sub.add_parser("parse")
    parse_cmd.add_argument("--input-path", type=Path, required=True)
    parse_cmd.add_argument("--output-path", type=Path, required=True)
    args = parser.parse_args()

    if args.action in ("prepare", "launch"):
        manifest = prepare(args)
        print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
        if args.action == "prepare":
            return 0
        command = args.command[1:] if args.command and args.command[0] == "--" else args.command
        if not command:
            raise ValueError("launch requires a command after --")
        env = os.environ.copy()
        env["SERVICE_PROF_CONFIG_PATH"] = manifest["config_path"]
        if manifest["symbols_path"]:
            env["PROFILING_SYMBOLS_PATH"] = manifest["symbols_path"]
        else:
            env.pop("PROFILING_SYMBOLS_PATH", None)
        return subprocess.call(command, env=env)

    if args.action == "parse":
        return subprocess.call([
            sys.executable, "-m", "ms_service_profiler", "parse",
            f"--input-path={args.input_path.resolve()}",
            f"--output-path={args.output_path.resolve()}",
        ])

    manifest, config_path, config = load_prepared(args)
    if args.action in ("start", "stop"):
        config["enable"] = 1 if args.action == "start" else 0
        atomic_json(config_path, config)
    print(json.dumps({"experiment": manifest["experiment"], "node": manifest["node"], "enable": config["enable"], "prof_dir": manifest["prof_dir"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"profilectl: {exc}", file=sys.stderr)
        raise SystemExit(2) from None
