#!/usr/bin/env python3
"""
DreamerV3 shim for embodied counting world.
Patches make_env() to recognize the 'embodied' suite, then calls dreamer.main().
"""

import argparse
import os
import pathlib
import sys

import yaml

import dreamer
from envs import wrappers


_orig_make_env = dreamer.make_env


def make_env(config, mode, id):
    suite, task = config.task.split("_", 1)
    if suite == "embodied":
        from envs.embodied import EmbodiedCountingWorld
        env = EmbodiedCountingWorld(task, seed=config.seed + id)
        env = wrappers.TimeLimit(env, config.time_limit)
        env = wrappers.SelectAction(env, key="action")
        env = wrappers.UUID(env)
        return env
    return _orig_make_env(config, mode, id)


dreamer.make_env = make_env


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+")
    args, remaining = parser.parse_known_args()

    # Resolve configs.yaml (same logic as dreamer.py)
    config_path = pathlib.Path(sys.argv[0]).parent / "configs.yaml"
    if not config_path.exists():
        dreamer_dir = os.environ.get("DREAMER_DIR", str(pathlib.Path.home() / "dreamerv3-torch"))
        config_path = pathlib.Path(dreamer_dir) / "configs.yaml"
    configs = yaml.safe_load(config_path.read_text())

    def recursive_update(base, update):
        for key, value in update.items():
            if isinstance(value, dict) and key in base:
                recursive_update(base[key], value)
            else:
                base[key] = value

    name_list = ["defaults", *args.configs] if args.configs else ["defaults"]
    defaults = {}
    for name in name_list:
        recursive_update(defaults, configs[name])

    # Displacement loss defaults (inherited from dreamer.py)
    defaults.setdefault("disp_lambda", 0.0)
    defaults.setdefault("disp_ema_decay", 0.99)
    defaults.setdefault("disp_warmup", 50)

    # PyYAML 6.0+ parses scientific notation (5e5, 1e-4) as strings.
    # Coerce string values that look like numbers into actual numbers
    # so args_type picks the right parser (int/float vs str identity).
    def _coerce_yaml_numbers(d):
        for k, v in d.items():
            if isinstance(v, dict):
                _coerce_yaml_numbers(v)
            elif isinstance(v, str):
                try:
                    fv = float(v)
                    d[k] = int(fv) if fv == int(fv) and "." not in v else fv
                except (ValueError, OverflowError):
                    pass
    _coerce_yaml_numbers(defaults)

    parser = argparse.ArgumentParser()
    for key, value in sorted(defaults.items(), key=lambda x: x[0]):
        from tools import args_type
        arg_type = args_type(value)
        parser.add_argument(f"--{key}", type=arg_type, default=arg_type(value))
    dreamer.main(parser.parse_args(remaining))
