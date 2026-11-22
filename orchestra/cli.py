"""CLI: ``python -m orchestra serve --config examples/config.json``."""

from __future__ import annotations

import argparse
import sys

from .config import Config
from .server import serve


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="orchestra", description="single-GPU model orchestrator")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("serve", help="start the HTTP server")
    s.add_argument("--config", required=True)
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=9881)
    s.add_argument("--backend", default="fake", help="fake | subprocess")
    s.add_argument("--debug", action="store_true")

    g = sub.add_parser("check", help="validate a config and print the budget")
    g.add_argument("--config", required=True)

    args = ap.parse_args(argv)
    cfg = Config.load(args.config)
    cfg.debug = getattr(args, "debug", False)  # type: ignore[attr-defined]

    if args.cmd == "check":
        total = sum(m.vram_gb for m in cfg.models)
        print(f"card            {cfg.vram_gb:.1f} GB")
        print(f"models          {len(cfg.models)}")
        for m in cfg.models:
            print(f"  {m.id:<16} {m.kind:<6} {m.vram_gb:5.1f} GB  backend={m.backend} tags={','.join(m.tags)}")
        print(f"sum of models   {total:.1f} GB  "
              f"({'fits at once' if total <= cfg.vram_gb else 'needs eviction'})")
        return 0

    httpd = serve(cfg, host=args.host, port=args.port, backend=args.backend)
    host, port = httpd.server_address[:2]
    print(f"listening on http://{host}:{port}  (backend={args.backend}, "
          f"models={len(cfg.models)}, vram={cfg.vram_gb:.1f}GB)", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
