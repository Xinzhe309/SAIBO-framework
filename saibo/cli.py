"""Command-line entry point for SAIBO 1.0."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .backends import available_methods
from .runners import run_dry, run_wet
from .smoke import run_all_smokes, run_labo_smoke, run_lgbo_smoke


def _run_smoke(args) -> object:
    output_dir = Path(args.output_dir)
    if args.method == "all":
        return run_all_smokes(output_dir, online=args.online)
    if args.method == "labo":
        return run_labo_smoke(output_dir, online=args.online)
    return run_lgbo_smoke(output_dir, online=args.online)


def _legacy_smoke(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Run SAIBO LABO/LGBO smoke experiments.")
    parser.add_argument("--method", choices=[*available_methods(), "all"], default="all")
    parser.add_argument("--online", action="store_true", help="Call the configured LLM API.")
    parser.add_argument("--output-dir", default="results/saibo_smoke")
    args = parser.parse_args(argv)
    print(json.dumps(_run_smoke(args), indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0].startswith("-"):
        return _legacy_smoke(argv)

    parser = argparse.ArgumentParser(description="SAIBO dry/wet/smoke interfaces.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    smoke_parser = subparsers.add_parser("smoke", help="Run compact method smoke checks.")
    smoke_parser.add_argument("--method", choices=[*available_methods(), "all"], default="all")
    smoke_parser.add_argument("--online", action="store_true", help="Call the configured LLM API.")
    smoke_parser.add_argument("--output-dir", default="results/saibo_smoke")

    dry_parser = subparsers.add_parser("dry", help="Run dry optimization with an internal evaluator.")
    dry_parser.add_argument("--method", choices=[*available_methods(), "all"], default="all")
    dry_parser.add_argument("--online", action="store_true", help="Call the configured LLM API.")
    dry_parser.add_argument("--rounds", type=int, default=1)
    dry_parser.add_argument("--batch-q", type=int, default=1)
    dry_parser.add_argument("--seed", type=int, default=7)
    dry_parser.add_argument("--task-json", help="Optional task profile with background and core experience.")
    dry_parser.add_argument(
        "--evaluator",
        help="Optional dry evaluator as path.py:object or module:object.",
    )
    dry_parser.add_argument("--output-dir", default="results/saibo_dry")

    wet_parser = subparsers.add_parser("wet", help="Plan a wet batch from observed history.")
    wet_parser.add_argument("--method", choices=available_methods(), required=True)
    wet_parser.add_argument("--data-json", required=True)
    wet_parser.add_argument("--online", action="store_true", help="Call the configured LLM API.")
    wet_parser.add_argument("--batch-q", type=int, default=1)
    wet_parser.add_argument("--seed", type=int, default=11)
    wet_parser.add_argument("--output-dir", default="results/saibo_wet")

    args = parser.parse_args(argv)
    if args.command == "smoke":
        result = _run_smoke(args)
    elif args.command == "dry":
        result = run_dry(
            args.method,
            args.output_dir,
            online=args.online,
            rounds=args.rounds,
            batch_q=args.batch_q,
            seed=args.seed,
            task_json=args.task_json,
            evaluator_spec=args.evaluator,
        )
    else:
        result = run_wet(
            args.method,
            args.data_json,
            args.output_dir,
            online=args.online,
            batch_q=args.batch_q,
            seed=args.seed,
        )

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
