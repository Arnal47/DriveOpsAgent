import argparse
import json
from pathlib import Path

from .agent import DriveOpsAgent
from .evals.runner import run_evals


def paths():
    root = Path(__file__).resolve().parents[2]
    return root / "data", root / "reports", root


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run")
    run.add_argument("--task", required=True)
    run.add_argument("--provider", default="mock", choices=["mock"])
    sub.add_parser("ingest")
    sub.add_parser("eval")
    args = parser.parse_args()
    data, reports, root = paths()
    if args.cmd == "ingest":
        print(json.dumps({"chunks": len(DriveOpsAgent(data, reports).registry.retriever.chunks)}))
        return
    if args.cmd == "eval":
        print(json.dumps(run_evals(root), indent=2))
        return
    state = DriveOpsAgent(data, reports).run(args.task)
    print(json.dumps(state.model_dump(), ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
