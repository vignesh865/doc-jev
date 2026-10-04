"""Build a Loop 0 data set from the official RVL-CDIP test split.

    .venv/bin/python prepare_data.py --name rvlcdip-v0 --per-class 13 --seed 0

Samples pages per class from the official `test.txt` labels, then streams the
official 38 GB archive once, keeping only the sampled pages (TIF -> PNG). The
archive is never stored on disk.

Writes data/<name>/:
  manifest.json   source, revision, seed, label list, counts, question wording
  items.jsonl     one line per question (choice, noul_true, noul_false)
  pages/*.png     the sampled pages (git-ignored; rebuildable from the manifest)
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import tarfile
import time
from pathlib import Path

import numpy as np
import requests
from PIL import Image

REPO = "aharley/rvl_cdip"
API = f"https://huggingface.co/api/datasets/{REPO}"
LABELS = [
    "letter", "form", "email", "handwritten", "advertisement", "scientific report",
    "scientific publication", "specification", "file folder", "news article",
    "budget", "invoice", "presentation", "questionnaire", "resume", "memo",
]

STATE = "A scanned document page."
CHOICE_INSTRUCTIONS = "What type of document is this?"


def option_id(label: str) -> str:
    return label.replace(" ", "_")


def noul_instructions(label: str) -> str:
    article = "an" if label[0] in "aeiou" else "a"
    return f"Is this document {article} {label}?"


def revision() -> str:
    return requests.get(API, timeout=30).json()["sha"]


def resolve(rev: str, path: str) -> str:
    return f"https://huggingface.co/datasets/{REPO}/resolve/{rev}/{path}"


def sample(test_lines: list[str], per_class: int, seed: int) -> list[dict]:
    by_class: dict[int, list[str]] = {i: [] for i in range(len(LABELS))}
    for line in test_lines:
        path, label = line.rsplit(" ", 1)
        by_class[int(label)].append(path)
    rng = np.random.default_rng(seed)
    pages = []
    for label, paths in by_class.items():
        for i in sorted(rng.choice(len(paths), per_class, replace=False)):
            pages.append({"tif": paths[i], "label": label})
    return pages


def questions(pages: list[dict], seed: int) -> list[dict]:
    rng = np.random.default_rng(seed + 1)
    items = []
    for page in pages:
        gold = LABELS[page["label"]]
        wrong = LABELS[int(rng.choice([i for i in range(len(LABELS)) if i != page["label"]]))]
        base = {"page_id": page["page_id"], "image": page["image"], "gold_label": gold, "state": STATE}
        items.append({**base, "item_id": f"{page['page_id']}-choice", "kind": "choice",
                      "question": {"type": "choice", "instructions": CHOICE_INSTRUCTIONS,
                                   "criteria": {option_id(l): None for l in LABELS}},
                      "answer": option_id(gold)})
        for kind, asked, ans in (("noul_true", gold, "true"), ("noul_false", wrong, "false")):
            items.append({**base, "item_id": f"{page['page_id']}-{kind}", "kind": kind, "asked_label": asked,
                          "question": {"type": "noul", "instructions": noul_instructions(asked)},
                          "answer": ans})
    return items


def stream_pages(url: str, wanted: dict[str, dict], out: Path) -> None:
    """Read the .tar.gz once from the network, saving only the wanted TIFs as PNG."""
    left = {p for p in wanted if not (out / wanted[p]["image"]).exists()}
    print(f"{len(wanted) - len(left)} pages already saved; streaming for {len(left)}", flush=True)
    if not left:
        return
    t0, seen = time.time(), 0
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        r.raw.decode_content = False
        with tarfile.open(fileobj=r.raw, mode="r|gz") as tar:
            for member in tar:
                seen += 1
                if seen % 20000 == 0:
                    print(f"  {seen} members, {len(left)} left, {time.time() - t0:.0f}s", flush=True)
                if not member.isfile():
                    continue
                key = next((p for p in left if member.name.endswith(p)), None) if member.name.endswith(".tif") else None
                if key is None:
                    continue
                data = tar.extractfile(member).read()
                Image.open(io.BytesIO(data)).convert("L").save(out / wanted[key]["image"])
                left.discard(key)
                if not left:
                    break
    if left:
        sys.exit(f"missing {len(left)} pages: {sorted(left)[:5]}")
    print(f"done in {time.time() - t0:.0f}s", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="rvlcdip-v0")
    ap.add_argument("--per-class", type=int, default=13)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    root = Path("data") / args.name
    (root / "pages").mkdir(parents=True, exist_ok=True)
    rev = revision()
    test_txt = requests.get(resolve(rev, "data/test.txt"), timeout=60).text
    lines = [l for l in test_txt.splitlines() if l.strip()]
    pages = sample(lines, args.per_class, args.seed)
    for i, page in enumerate(pages):
        page["page_id"] = f"p{i:03d}"
        page["image"] = f"pages/{page['page_id']}.png"
    stream_pages(resolve(rev, "data/rvl-cdip.tar.gz"), {p["tif"]: p for p in pages}, root)

    items = questions(pages, args.seed)
    with open(root / "items.jsonl", "w") as f:
        for item in items:
            f.write(json.dumps(item) + "\n")
    manifest = {
        "name": args.name,
        "source": {"repo": REPO, "revision": rev, "split": "test", "labels_file": "data/test.txt",
                   "test_txt_sha256": hashlib.sha256(test_txt.encode()).hexdigest()[:16]},
        "sampling": {"per_class": args.per_class, "seed": args.seed, "classes": len(LABELS)},
        "labels": LABELS,
        "pages": [{"page_id": p["page_id"], "tif": p["tif"], "label": LABELS[p["label"]]} for p in pages],
        "questions": {"state": STATE, "choice": CHOICE_INSTRUCTIONS,
                      "choice_options": "16 RVL-CDIP names, spaces -> '_', no descriptions",
                      "noul": "Is this document a/an <label>?",
                      "noul_false": "one wrong label per page, uniform, seed + 1"},
        "counts": {"pages": len(pages), "items": len(items)},
        "labels_checked": False,
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"wrote {root}: {len(pages)} pages, {len(items)} questions")


if __name__ == "__main__":
    main()
