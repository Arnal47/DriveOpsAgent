import argparse
import json
from pathlib import Path

from .agent import DriveOpsAgent
from .evals.runner import run_evals
from .providers import MockProvider, OpenAICompatibleProvider


def main():
    p = argparse.ArgumentParser()
    s = p.add_subparsers(dest="cmd", required=True)
    r = s.add_parser("run")
    r.add_argument("--task", required=True)
    r.add_argument("--provider", choices=["mock", "openai-compatible"], default="mock")
    s.add_parser("ingest")
    s.add_parser("eval")
    a = p.parse_args()
    root = Path(__file__).resolve().parents[2]
    if a.cmd == "ingest":
        print(
            json.dumps(
                {
                    "chunks": len(
                        DriveOpsAgent(root / "data", root / "reports").registry.retriever.chunks
                    )
                }
            )
        )
        return
    if a.cmd == "eval":
        print(json.dumps(run_evals(root), indent=2))
        return
    provider = MockProvider() if a.provider == "mock" else OpenAICompatibleProvider()
    print(
        json.dumps(
            DriveOpsAgent(root / "data", root / "reports", provider).run(a.task).model_dump(),
            ensure_ascii=False,
            default=str,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
