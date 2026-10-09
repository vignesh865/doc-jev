"""Run one experiment: send a data set's questions to a hosted Clef model.

Start an experiment (its settings are frozen in experiments/<exp>/config.json):

    .venv/bin/python run_clef.py --exp E01 --data rvlcdip-v0 --model clef-flash --max-image-kb 185 --pages 5

Grow it later; settings come from the config, only --pages changes:

    .venv/bin/python run_clef.py --exp E01 --pages 20

Pages are taken in the data set's fixed `page_order` (one class at a time), so
5 -> 20 -> 208 are nested: every run extends the previous one. With
--max-image-kb, pages whose encoded image is larger are never sent (the hosted
API refuses them, JOURNAL entry 7); they are skipped in the order and listed
in experiments/<exp>/skipped_pages.json. --pages counts pages actually sent. Re-running is
safe: questions that already have a successful reply are skipped, so an
interrupted run resumes where it stopped.

One request per question (questions in one request are scored jointly and can
see each other). Raw replies are appended to outputs/<exp>.jsonl: item id, the
request without the image, the raw response, HTTP status, latency and time.
Never edit that file; evaluate.py scores it.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from PIL import Image

ENDPOINT = "https://api.cloudflare.com/client/v4/accounts/{account}/ai/run/@cf/cloudflare/{model}"
FROZEN = ("data", "model", "jpeg_quality", "max_image_kb", "question_version")


def load_env(path: str = ".env") -> dict[str, str]:
    return dict(l.rstrip("\n").split("=", 1) for l in open(path) if "=" in l)


def encode_image(path: Path, jpeg_quality: int | None) -> tuple[str, str]:
    """Base64 image for the request. The hosted API estimates tokens from the base64
    length (about 4 characters per token), so large PNGs are refused (JOURNAL entry
    7); JPEG keeps them under the limit. Returns (content_type, base64)."""
    if jpeg_quality is None:
        content_type = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}[path.suffix.lower()]
        return content_type, base64.b64encode(path.read_bytes()).decode()
    buf = io.BytesIO()
    Image.open(path).convert("L").save(buf, "JPEG", quality=jpeg_quality)
    return "image/jpeg", base64.b64encode(buf.getvalue()).decode()


def load_config(args: argparse.Namespace) -> dict:
    """Create the experiment's config on first use; afterwards refuse any change to it."""
    folder = Path("experiments") / args.exp
    path = folder / "config.json"
    if path.exists():
        config = json.loads(path.read_text())
        for key in FROZEN:
            given = getattr(args, key, None)
            if given is not None and given != config[key]:
                raise SystemExit(f"{args.exp} was created with {key}={config[key]!r}; "
                                 f"start a new experiment id to change it")
        return config
    if args.data is None or args.model is None:
        raise SystemExit("a new experiment needs --data and --model")
    manifest = json.loads((Path("data") / args.data / "manifest.json").read_text())
    config = {
        "exp": args.exp, "data": args.data, "model": args.model, "jpeg_quality": args.jpeg_quality,
        "max_image_kb": args.max_image_kb,
        "question_version": manifest["questions"]["version"],
        "endpoint": ENDPOINT.format(account="<account>", model=args.model),
        "model_version": "hosted Workers AI; not pinnable",
        "one_request_per_question": True,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip(),
        "outputs": f"outputs/{args.exp}.jsonl",
        "note": args.note,
    }
    folder.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=1) + "\n")
    print(f"created {path}")
    return config


def done_ids(out: Path) -> set[str]:
    if not out.exists():
        return set()
    return {r["item_id"] for r in map(json.loads, out.open()) if r["status"] == 200 and r["response"].get("success")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True, help="experiment id, e.g. E01")
    ap.add_argument("--data")
    ap.add_argument("--model", choices=["clef-flash", "clef"])
    ap.add_argument("--jpeg-quality", type=int, default=None)
    ap.add_argument("--max-image-kb", type=int, default=None, help="skip pages whose encoded image is larger")
    ap.add_argument("--pages", type=int, required=True, help="run the first N pages of the fixed page order")
    ap.add_argument("--note", default="")
    ap.add_argument("--dry-run", action="store_true", help="show what would be sent; make no calls")
    args = ap.parse_args()
    args.question_version = None

    config = load_config(args)
    env = load_env()
    url = ENDPOINT.format(account=env["CLOUDFLARE_ACCOUNT_ID"], model=config["model"])
    headers = {"Authorization": f"Bearer {env['CLOUDFLARE_AUTH_TOKEN']}"}
    root = Path("data") / config["data"]
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest["questions"]["version"] != config["question_version"]:
        raise SystemExit("data questions changed since this experiment was created; start a new experiment id")
    image_of = {p["page_id"]: p.get("image", f"pages/{p['page_id']}.png") for p in manifest["pages"]}
    images: dict[str, tuple[str, str]] = {}
    chosen, skipped = [], []
    for page_id in manifest["page_order"]:
        if len(chosen) == args.pages:
            break
        image = image_of[page_id]
        images[image] = encode_image(root / image, config["jpeg_quality"])
        size_kb = len(images[image][1]) * 3 / 4 / 1024
        if config["max_image_kb"] is not None and size_kb > config["max_image_kb"]:
            skipped.append({"page_id": page_id, "image_kb": round(size_kb, 1)})
        else:
            chosen.append(page_id)
    folder = Path("experiments") / config["exp"]
    (folder / "skipped_pages.json").write_text(json.dumps(skipped, indent=1) + "\n")
    pages = set(chosen)
    items = [i for i in map(json.loads, (root / "items.jsonl").open()) if i["page_id"] in pages]
    out = Path(config["outputs"])
    out.parent.mkdir(exist_ok=True)
    skip = done_ids(out)
    todo = [i for i in items if i["item_id"] not in skip]
    print(f"{config['exp']}: {len(pages)} pages ({len(skipped)} too big, skipped), {len(items)} questions, {len(items) - len(todo)} already done, "
          f"{len(todo)} to send -> {out}", flush=True)

    if args.dry_run:
        for page_id in chosen:
            kb = len(images[image_of[page_id]][1]) * 3 / 4 / 1024
            gold = next(i["gold_label"] for i in items if i["page_id"] == page_id)
            left = sum(1 for i in todo if i["page_id"] == page_id)
            print(f"  {page_id}  {gold:24s} {kb:6.1f} KB  calls to make: {left}")
        for s_ in skipped:
            print(f"  {s_['page_id']}  skipped, {s_['image_kb']} KB")
        print(f"dry run: {len(todo)} calls would be made")
        return
    failures = 0
    with out.open("a") as f:
        for n, item in enumerate(todo, 1):
            if item["image"] not in images:
                images[item["image"]] = encode_image(root / item["image"], config["jpeg_quality"])
            content_type, b64 = images[item["image"]]
            body = {"model": config["model"], "state": item["state"], "questions": {"q": item["question"]}}
            payload = {**body, "images": [{"content_type": content_type, "base64": b64}]}
            for attempt in range(5):
                t0 = time.time()
                try:
                    r = requests.post(url, headers=headers, json=payload, timeout=180)
                    status, response = r.status_code, r.json()
                except (requests.RequestException, ValueError) as e:
                    status, response = -1, {"success": False, "errors": [str(e)]}
                latency = time.time() - t0
                daily_cap = status == 429 and any(isinstance(e, dict) and e.get("code") == 4006
                                                  for e in response.get("errors", []))
                if status == 200 or status in (400, 401, 403, 413) or daily_cap:
                    break
                time.sleep(2 ** attempt)
            f.write(json.dumps({
                "exp": config["exp"], "item_id": item["item_id"], "page_id": item["page_id"],
                "model": config["model"], "data": config["data"],
                "sent_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "request": body, "image": item["image"], "image_type": content_type,
                "jpeg_quality": config["jpeg_quality"], "image_bytes": len(b64) * 3 // 4,
                "status": status, "latency_s": round(latency, 3), "attempts": attempt + 1,
                "response": response,
            }) + "\n")
            f.flush()
            if status != 200:
                failures += 1
                print(f"  {item['item_id']}: status={status} {response.get('errors')}", flush=True)
            elif n % 25 == 0:
                print(f"  {n}/{len(todo)} latency={latency:.2f}s", flush=True)
            if status in (401, 403):
                raise SystemExit("auth error; stopping")
            if daily_cap:
                raise SystemExit("daily free allocation used up (Workers AI 4006); re-run after the daily reset to resume")
    print(f"done: {len(todo) - failures} sent ok, {failures} failed (re-run to retry failures)")


if __name__ == "__main__":
    main()
