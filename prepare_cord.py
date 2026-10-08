"""Build a content-question data set from the CORD-v2 test split (receipt photos).

    .venv/bin/python prepare_cord.py --name cord-v0 --seed 0

Uses the official parquet at a pinned revision (downloaded once to
data/_raw/, git-ignored). Every receipt with a labelled total gets three
questions about that total; all answers come from CORD's human labels:

  choice      "Which of these amounts is the total on this receipt?"
              options = the total + up to 3 other amounts *labelled on the same
              receipt* (item prices, subtotal, tax, cash, change, ...)
  noul_true   "Is the total amount on this receipt <true total>?"
  noul_false  the same question with ONE digit of the total changed
              (the format stays identical, so only reading the digits helps)

Images: originals are full-resolution PNG photos (median 1.5 MB); the hosted
API refuses images over ~185 KB (JOURNAL entry 7). Every receipt is resized to
a longest side of at most 1024 px and saved as JPEG quality 85: the one
uniform setting that fits all 100 (max 170 KB).

Writes data/<name>/: manifest.json, items.jsonl, pages/*.jpg (git-ignored).
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import requests
from PIL import Image

REPO = "naver-clova-ix/cord-v2"
REVISION = "7f0115a4b758a71d6473b8d085751692da2fef98"
TEST_FILE = "data/test-00000-of-00001-9c204eb3f4e11791.parquet"
RAW = Path("data/_raw/cord-v2-test.parquet")

QUESTION_VERSION = "c1"
STATE = "A photo of a shop receipt."
CHOICE_INSTRUCTIONS = "Which of these amounts is the total on this receipt?"
MAX_SIDE, JPEG_QUALITY = 1024, 85

# Labelled fields that hold money amounts (counts, names and codes are excluded).
AMOUNT_FIELDS = {
    "menu": ["price", "unitprice", "itemsubtotal", "discountprice"],
    "menu.sub": ["price"],
    "sub_total": ["subtotal_price", "tax_price", "service_price", "discount_price"],
    "total": ["cashprice", "changeprice", "creditcardprice", "emoneyprice"],
}


def amount_value(text: str) -> float | None:
    """Numeric value of an amount as written on an Indonesian receipt.
    '.' and ',' are thousands separators unless followed by exactly 2 final digits."""
    s = re.sub(r"[^0-9.,-]", "", str(text))
    if not re.search(r"\d", s):
        return None
    m = re.match(r"^(-?[\d.,]*)[.,](\d{2})$", s)
    whole, frac = m.groups() if m else (s, "0")
    digits = whole.replace(".", "").replace(",", "")
    if digits in ("", "-"):
        return None
    return float(f"{digits}.{frac}")


def as_list(x) -> list:
    return x if isinstance(x, list) else [x] if x else []


def labelled_amounts(gt: dict) -> list[str]:
    out = []
    for item in as_list(gt.get("menu")):
        out += [item[k] for k in AMOUNT_FIELDS["menu"] if isinstance(item.get(k), str)]
        for sub in as_list(item.get("sub")):
            out += [sub[k] for k in AMOUNT_FIELDS["menu.sub"] if isinstance(sub.get(k), str)]
    for group in ("sub_total", "total"):
        block = gt.get(group) if isinstance(gt.get(group), dict) else {}
        out += [block[k] for k in AMOUNT_FIELDS[group] if isinstance(block.get(k), str)]
    return out


def change_one_digit(text: str, rng: np.random.Generator) -> str:
    """Replace one digit with a different one; never creates a leading zero."""
    positions = [i for i, c in enumerate(text) if c.isdigit()]
    first = positions[0]
    while True:
        i = int(rng.choice(positions))
        choices = [d for d in "0123456789" if d != text[i] and not (i == first and d == "0")]
        new = text[:i] + str(rng.choice(choices)) + text[i + 1:]
        if amount_value(new) != amount_value(text):
            return new


def build(rows: list[dict], seed: int, out: Path) -> tuple[list[dict], list[dict], list[dict]]:
    rng = np.random.default_rng(seed)
    pages, items, skipped = [], [], []
    for idx, row in enumerate(rows):
        page_id = f"c{idx:03d}"
        gt = json.loads(row["ground_truth"])["gt_parse"]
        total = (gt.get("total") or {}).get("total_price") if isinstance(gt.get("total"), dict) else None
        if not isinstance(total, str) or amount_value(total) is None:
            skipped.append({"page_id": page_id, "reason": "no labelled total"})
            continue
        im = Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB")
        if max(im.size) > MAX_SIDE:
            s = MAX_SIDE / max(im.size)
            im = im.resize((round(im.size[0] * s), round(im.size[1] * s)), Image.LANCZOS)
        image = f"pages/{page_id}.jpg"
        im.save(out / image, "JPEG", quality=JPEG_QUALITY)
        pages.append({"page_id": page_id, "image": image, "total": total,
                      "original_size": list(Image.open(io.BytesIO(row["image"]["bytes"])).size), "sent_size": list(im.size)})

        # Distractors: distinct labelled amounts on this receipt whose value differs from the total.
        tv, seen, pool = amount_value(total), {amount_value(total)}, []
        for a in labelled_amounts(gt):
            v = amount_value(a)
            if v is None or v <= 0 or v in seen:
                continue
            seen.add(v)
            pool.append(a)
        distractors = [pool[i] for i in rng.permutation(len(pool))[:3]] if pool else []
        base = {"page_id": page_id, "image": image, "gold_label": total, "state": STATE,
                "question_version": QUESTION_VERSION}
        if distractors:
            amounts = [total] + distractors
            letters = "abcd"[: len(amounts)]
            order = rng.permutation(len(amounts))
            criteria = {letters[j]: amounts[k] for j, k in enumerate(order)}
            answer = next(l for l, a in criteria.items() if a == total)
            items.append({**base, "item_id": f"{page_id}-choice", "kind": "choice", "n_options": len(amounts),
                          "question": {"type": "choice", "instructions": CHOICE_INSTRUCTIONS, "criteria": criteria},
                          "answer": answer})
        wrong = change_one_digit(total, rng)
        for kind, value, ans in (("noul_true", total, "true"), ("noul_false", wrong, "false")):
            items.append({**base, "item_id": f"{page_id}-{kind}", "kind": kind, "asked_value": value,
                          "question": {"type": "noul", "instructions": f"Is the total amount on this receipt {value}?"},
                          "answer": ans})
    return pages, items, skipped


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="cord-v0")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not RAW.exists():
        RAW.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{TEST_FILE}"
        RAW.write_bytes(requests.get(url, timeout=600).content)
    rows = pq.read_table(RAW).to_pylist()
    root = Path("data") / args.name
    (root / "pages").mkdir(parents=True, exist_ok=True)
    pages, items, skipped = build(rows, args.seed, root)

    rng = np.random.default_rng(args.seed + 2)
    order = [pages[i]["page_id"] for i in rng.permutation(len(pages))]
    with open(root / "items.jsonl", "w") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    kinds = {k: sum(1 for i in items if i["kind"] == k) for k in ("choice", "noul_true", "noul_false")}
    manifest = {
        "name": args.name,
        "source": {"repo": REPO, "revision": REVISION, "split": "test", "file": TEST_FILE,
                   "file_sha256": hashlib.sha256(RAW.read_bytes()).hexdigest()[:16], "license": "CC-BY-4.0"},
        "seed": args.seed,
        "image_transform": f"RGB, longest side <= {MAX_SIDE} px (LANCZOS), JPEG quality {JPEG_QUALITY}",
        "page_order": order,
        "pages": pages,
        "skipped": skipped,
        "questions": {"version": QUESTION_VERSION, "state": STATE, "choice": CHOICE_INSTRUCTIONS,
                      "choice_options": "total + up to 3 other distinct amounts labelled on the same receipt; ids a-d, shuffled",
                      "noul": "Is the total amount on this receipt <value>?",
                      "noul_false": "true total with one digit changed (same format)"},
        "counts": {"receipts": len(pages), "items": len(items), **kinds},
        "labels_checked": False,
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {root}: {len(pages)} receipts ({len(skipped)} skipped), {len(items)} questions {kinds}")


if __name__ == "__main__":
    main()
