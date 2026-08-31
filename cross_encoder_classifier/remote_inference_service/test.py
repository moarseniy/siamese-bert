import argparse
import json
from pathlib import Path

import httpx


parser = argparse.ArgumentParser()
parser.add_argument("file", type=Path)
parser.add_argument("--url", default="http://127.0.0.1:8800")
parser.add_argument("--codebook")
parser.add_argument("--output", type=Path, default=Path("result.json"))
args = parser.parse_args()

response = httpx.post(
    f"{args.url.rstrip('/')}/predict",
    params={"codebook": args.codebook} if args.codebook else None,
    json=json.loads(args.file.read_text(encoding="utf-8")),
    timeout=None,
)
response.raise_for_status()
args.output.write_text(
    json.dumps(response.json(), ensure_ascii=False, indent=2), encoding="utf-8"
)
print(args.output)
