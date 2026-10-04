"""Send every question of a data set to a hosted Clef model; save the raw replies.

    .venv/bin/python run_clef.py --data rvlcdip-v0 --model clef-flash

One request per question (questions in one request are scored jointly and can
see each other). Writes outputs/<model>-<data>.jsonl, one line per request:
the item id, the request without the image, the raw response, HTTP status,
latency and time. Never edit this file; evaluate.py scores it. Re-running
resumes: items that already have a successful reply are skipped.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from PIL import Image

ENDPOINT = "https://api.cloudflare.com/client/v4/accounts/{account}/ai/run/@cf/cloudflare/{model}"


def load_env(path: str = ".env") -> dict[str, str]:
    return dict(l.rstrip("\n").split("=", 1) for l in open(path) if "=" in l)


def encode_image(path: Path, jpeg_quality: int | None) -> tuple[str, str]:
    """Base64 image for the request. The hosted API estimates tokens from the base64
    length (about 4 characters per token), so large PNGs are refused; JPEG keeps
    them under the limit. Returns (content_type, base64)."""
    if jpeg_quality is None:
        return "image/png", base64.b64encode(path.read_bytes()).decode()
    buf = io.BytesIO()
    Image.open(path).convert("L").save(buf, "JPEG", quality=jpeg_quality)
    return "image/jpeg", base64.b64encode(buf.getvalue()).decode()


def done_ids(out: Path) -> set[str]:
    if not out.exists():
        return set()
    return {r["item_id"] for r in map(json.loads, out.open()) if r["status"] == 200 and r["response"].get("success")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="rvlcdip-v0")
    ap.add_argument("--model", default="clef-flash", choices=["clef-flash", "clef"])
    ap.add_argument("--jpeg-quality", type=int, default=None, help="send pages as JPEG at this quality")
    ap.add_argument("--limit", type=int, default=None, help="only the first N items (smoke test)")
    args = ap.parse_args()

    env = load_env()
    url = ENDPOINT.format(account=env["CLOUDFLARE_ACCOUNT_ID"], model=args.model)
    headers = {"Authorization": f"Bearer {env['CLOUDFLARE_AUTH_TOKEN']}"}
    root = Path("data") / args.data
    items = [json.loads(l) for l in (root / "items.jsonl").open()][: args.limit]
    out = Path("outputs") / f"{args.model}-{args.data}.jsonl"
    out.parent.mkdir(exist_ok=True)
    skip = done_ids(out)
    todo = [i for i in items if i["item_id"] not in skip]
    print(f"{len(items)} items, {len(skip)} done, {len(todo)} to send -> {out}", flush=True)

    images: dict[str, tuple[str, str]] = {}
    with out.open("a") as f:
        for n, item in enumerate(todo, 1):
            if item["image"] not in images:
                images[item["image"]] = encode_image(root / item["image"], args.jpeg_quality)
            content_type, b64 = images[item["image"]]
            body = {"model": args.model, "state": item["state"], "questions": {"q": item["question"]}}
            payload = {**body, "images": [{"content_type": content_type, "base64": b64}]}
            for attempt in range(5):
                t0 = time.time()
                try:
                    r = requests.post(url, headers=headers, json=payload, timeout=180)
                    status, response = r.status_code, r.json()
                except (requests.RequestException, ValueError) as e:
                    status, response = -1, {"success": False, "errors": [str(e)]}
                latency = time.time() - t0
                if status == 200 or status in (400, 401, 403, 413):
                    break
                time.sleep(2 ** attempt)
            f.write(json.dumps({
                "item_id": item["item_id"], "model": args.model, "data": args.data,
                "sent_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "request": body, "image": item["image"], "image_type": content_type, "jpeg_quality": args.jpeg_quality,
                "image_bytes": len(b64) * 3 // 4,
                "status": status, "latency_s": round(latency, 3), "attempts": attempt + 1,
                "response": response,
            }) + "\n")
            f.flush()
            if n % 25 == 0 or status != 200:
                print(f"  {n}/{len(todo)} status={status} latency={latency:.2f}s", flush=True)
            if status in (401, 403):
                raise SystemExit(f"auth error: {response.get('errors')}")


if __name__ == "__main__":
    main()
