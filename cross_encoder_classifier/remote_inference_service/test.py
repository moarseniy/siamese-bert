import argparse
from pathlib import Path

import httpx


parser = argparse.ArgumentParser()
parser.add_argument("file", type=Path)
parser.add_argument("--url", default="http://127.0.0.1:8800")
parser.add_argument("--output", type=Path, default=Path("result.zip"))
args = parser.parse_args()

with args.file.open("rb") as source:
    response = httpx.post(
        f"{args.url.rstrip('/')}/predict",
        files={"file": (args.file.name, source)},
        timeout=None,
    )
response.raise_for_status()
args.output.write_bytes(response.content)
print(args.output)
