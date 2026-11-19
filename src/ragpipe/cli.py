from __future__ import annotations

import argparse
import json

from .config import Config
from .evaluate import evaluate
from .pipeline import RAGPipeline


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="ragpipe")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("ingest"); s.add_argument("path")
    s = sub.add_parser("ask"); s.add_argument("question"); s.add_argument("-k", type=int, default=5)
    s = sub.add_parser("eval"); s.add_argument("dataset"); s.add_argument("-k", type=int, default=5)
    s = sub.add_parser("serve"); s.add_argument("--port", type=int, default=8000)
    args = p.parse_args(argv)

    if args.cmd == "serve":
        import uvicorn

        uvicorn.run("ragpipe.api:app", host="0.0.0.0", port=args.port)
        return
    pipe = RAGPipeline(Config())
    if args.cmd == "ingest":
        print(f"Indexed {pipe.ingest(args.path)} chunks -> {pipe.cfg.index_dir}/")
    elif args.cmd == "ask":
        a = pipe.ask(args.question, args.k)
        print(a.answer, "\n")
        for i, h in enumerate(a.hits, 1):
            print(f"[{i}] {h.chunk.source} (score {h.score:.3f})")
    elif args.cmd == "eval":
        rows = [evaluate(pipe, args.dataset, args.k, m, r) for m, r in
                [("dense", False), ("bm25", False), ("hybrid", False), ("hybrid", True)]]
        for r in rows:
            print(json.dumps(r))


if __name__ == "__main__":
    main()
