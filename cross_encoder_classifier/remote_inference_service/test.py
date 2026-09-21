import argparse
import json
from pathlib import Path

import httpx


def send_batch(
    client: httpx.Client,
    url: str,
    params: dict[str, str] | None,
    records: list[dict],
) -> list[dict]:
    response = client.post(url, params=params, json=records)
    if response.status_code == 413 and len(records) > 1:
        middle = len(records) // 2
        return send_batch(client, url, params, records[:middle]) + send_batch(
            client, url, params, records[middle:]
        )
    response.raise_for_status()
    result = response.json()
    if not isinstance(result, list) or len(result) != len(records):
        raise RuntimeError("Service returned an unexpected number of records")
    return result


parser = argparse.ArgumentParser()
parser.add_argument("file", type=Path)
parser.add_argument("--url", default="http://127.0.0.1:8800")
parser.add_argument("--codebook")
parser.add_argument("--batch-size", type=int, default=500)
parser.add_argument("--output", type=Path, default=Path("result.json"))
args = parser.parse_args()

records = json.loads(args.file.read_text(encoding="utf-8"))
if not isinstance(records, list):
    raise SystemExit("Input JSON must contain a list")
if args.batch_size < 1:
    raise SystemExit("--batch-size must be positive")

url = f"{args.url.rstrip('/')}/predict"
params = {"codebook": args.codebook} if args.codebook else None
result = []
with httpx.Client(timeout=None) as client:
    for start in range(0, len(records), args.batch_size):
        result.extend(
            send_batch(client, url, params, records[start : start + args.batch_size])
        )
        print(f"Processed {min(start + args.batch_size, len(records))}/{len(records)}")

args.output.write_text(
    json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
)
print(args.output)
