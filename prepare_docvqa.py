"""Build a content-question data set from DocVQA validation (real business pages).

    .venv/bin/python prepare_docvqa.py --name docvqa-v0 --per-skill 20 --seed 0

Source: lmms-lab-encoder/DocVQA, validation split, pinned revision; parquet
files downloaded once to data/_raw/docvqa/ (git-ignored). All answers are
human (DocVQA's crowd answers, verified by a second annotator).

DocVQA answers are open text. We turn one question per page into three typed
questions, using the answers to the OTHER questions about the SAME page as the
wrong options (so every option is a human-labelled string from that page):

  choice      the DocVQA question; options = the true answer + up to 3
              same-page answers of the same kind (number / date / code / words of
              similar length: 1, 2-3 or 4+ words)
  noul_true   'Question about this page: "<q>" Is the answer "<true answer>"?'
  noul_false  the same with a same-page answer of the same kind

A wrong option is never a near-duplicate of any accepted answer (normalised
equality, containment, or similarity > 0.8), so it is never secretly right.

Sampling: N questions for each of 6 skills (DocVQA question types), each from
a different page; pages are used once across the whole set. Images: greyscale,
longest side <= 1024 px, JPEG q75 (the API refuses ~185 KB+); a sampled page
that is still over --max-kb is replaced and listed in the manifest.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

REPO = "lmms-lab-encoder/DocVQA"
REVISION = "539088ef8a8ada01ac8e2e6d4e372586748a265e"
RAW = Path("data/_raw/docvqa")
SKILLS = ["table/list", "layout", "form", "free_text", "handwritten", "figure/diagram"]

QUESTION_VERSION = "d1"
STATE = "A scanned document page."
MONTHS = "jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec"


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def length_band(answer: str) -> str:
    n = len(answer.split())
    return "1" if n == 1 else "2-3" if n <= 3 else "4+"


def kind(answer: str) -> str:
    s = answer.strip().lower()
    if (re.search(rf"\b({MONTHS})[a-z]*\b", s) and re.search(r"\d", s)) or re.fullmatch(r"\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}", s):
        return "date"
    if re.fullmatch(r"[-+$€£(]*\s*[\d.,]+\s*%?\)?\s*[a-z%]{0,6}\.?", s):
        return "number"
    if re.search(r"\d", s):
        return "code"
    return f"words:{length_band(s)}"  # word answers only compete with similar-length ones


MONTH_NUM = {m: i + 1 for i, m in enumerate(MONTHS.split("|"))}


def date_key(s: str) -> tuple[int, int, int] | None:
    """(yy, month, day) for common date spellings, so '6-2-97' == 'June 2, 1997'.
    Added after d2389 in docvqa-v0 (see data/docvqa-v0/errata.json)."""
    s = s.lower().strip()
    m = re.search(rf"\b({MONTHS})[a-z]*\.?\s+(\d{{1,2}}),?\s+(\d{{2,4}})", s)
    if m:
        return int(m.group(3)) % 100, MONTH_NUM[m.group(1)], int(m.group(2))
    m = re.search(rf"\b(\d{{1,2}})\s+({MONTHS})[a-z]*\.?,?\s+(\d{{2,4}})", s)
    if m:
        return int(m.group(3)) % 100, MONTH_NUM[m.group(2)], int(m.group(1))
    m = re.fullmatch(r"(\d{1,2})[/.-](\d{1,2})[/.-](\d{2,4})", s)
    if m:
        return int(m.group(3)) % 100, int(m.group(1)), int(m.group(2))
    return None


def clashes(a: str, others: list[str]) -> bool:
    na = norm(a)
    for b in others:
        nb = norm(b)
        if not na or not nb or na == nb or na in nb or nb in na or SequenceMatcher(None, na, nb).ratio() > 0.8:
            return True
        if date_key(a) is not None and date_key(a) == date_key(b):
            return True
    return False


def distractors(target: dict, page_questions: list[dict]) -> list[str]:
    k, out = kind(target["answers"][0]), []
    for other in page_questions:
        if other["questionId"] == target["questionId"]:
            continue
        a = other["answers"][0]
        if kind(a) == k and not clashes(a, target["answers"]) and not clashes(a, out):
            out.append(a)
    return out


def load_rows() -> list[dict]:
    rows = []
    for f in sorted(RAW.glob("validation-*.parquet")):
        rows += pq.read_table(f, columns=["questionId", "question", "question_types", "answers", "docId"]).to_pylist()
    return rows


def load_images(doc_ids: set[int]) -> dict[int, bytes]:
    out = {}
    for f in sorted(RAW.glob("validation-*.parquet")):
        for r in pq.read_table(f, columns=["docId", "image"]).to_pylist():
            if r["docId"] in doc_ids and r["docId"] not in out:
                out[r["docId"]] = r["image"]["bytes"]
    return out


def encode(raw: bytes, max_side: int, quality: int) -> tuple[bytes, tuple[int, int]]:
    im = Image.open(io.BytesIO(raw)).convert("L")
    if max(im.size) > max_side:
        s = max_side / max(im.size)
        im = im.resize((round(im.size[0] * s), round(im.size[1] * s)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=quality)
    return buf.getvalue(), im.size


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="docvqa-v0")
    ap.add_argument("--per-skill", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-side", type=int, default=1024, help="longest side of the sent image, px")
    ap.add_argument("--jpeg-quality", type=int, default=75)
    ap.add_argument("--max-kb", type=int, default=180, help="pages whose encoded image is larger are replaced")
    args = ap.parse_args()

    rows = load_rows()
    by_page: dict[int, list[dict]] = {}
    for r in rows:
        by_page.setdefault(r["docId"], []).append(r)

    images = load_images(set(by_page))
    encoded: dict[int, tuple[bytes, tuple[int, int]]] = {}
    too_big: list[dict] = []

    def fits(doc_id: int) -> bool:
        if doc_id not in encoded:
            encoded[doc_id] = encode(images[doc_id], args.max_side, args.jpeg_quality)
        return len(encoded[doc_id][0]) <= args.max_kb * 1024

    rng = np.random.default_rng(args.seed)
    used_pages, chosen = set(), []
    for skill in SKILLS:
        pool = []
        for r in rows:
            if skill not in r["question_types"] or r["docId"] in used_pages:
                continue
            ds = distractors(r, by_page[r["docId"]])
            if len(ds) >= 2:  # at least 3 options in the choice question
                pool.append((r, ds))
        picked = 0
        for i in rng.permutation(len(pool)):
            r, ds = pool[i]
            if r["docId"] in used_pages:
                continue
            if not fits(r["docId"]):
                too_big.append({"skill": skill, "docId": r["docId"], "kb": round(len(encoded[r["docId"]][0]) / 1024, 1)})
                used_pages.add(r["docId"])
                continue
            used_pages.add(r["docId"])
            chosen.append((skill, r, ds))
            picked += 1
            if picked == args.per_skill:
                break

    root = Path("data") / args.name
    (root / "pages").mkdir(parents=True, exist_ok=True)
    pages, items = [], []
    for skill, r, ds in chosen:
        page_id = f"d{r['docId']}"
        data, sent_size = encoded[r["docId"]]
        image = f"pages/{page_id}.jpg"
        (root / image).write_bytes(data)
        original = Image.open(io.BytesIO(images[r["docId"]])).size
        pages.append({"page_id": page_id, "image": image, "skill": skill, "question_id": r["questionId"],
                      "original_size": list(original), "sent_size": list(sent_size), "kb": round(len(data) / 1024, 1)})
        truth = r["answers"][0]
        wrong = [ds[j] for j in rng.permutation(len(ds))[:3]]
        base = {"page_id": page_id, "image": image, "gold_label": truth, "accepted_answers": r["answers"],
                "skill": skill, "question_types": r["question_types"], "answer_kind": kind(truth),
                "docvqa_question": r["question"], "state": STATE, "question_version": QUESTION_VERSION}
        options = [truth] + wrong
        letters = "abcd"[: len(options)]
        order = rng.permutation(len(options))
        criteria = {letters[j]: options[k] for j, k in enumerate(order)}
        items.append({**base, "item_id": f"{page_id}-choice", "kind": "choice", "n_options": len(options),
                      "question": {"type": "choice", "instructions": r["question"], "criteria": criteria},
                      "answer": next(l for l, a in criteria.items() if a == truth)})
        for kind_, value, ans in (("noul_true", truth, "true"), ("noul_false", wrong[0], "false")):
            items.append({**base, "item_id": f"{page_id}-{kind_}", "kind": kind_, "asked_value": value,
                          "question": {"type": "noul",
                                       "instructions": f'Question about this page: "{r["question"]}" Is the answer "{value}"?'},
                          "answer": ans})

    order = [pages[i]["page_id"] for i in np.random.default_rng(args.seed + 2).permutation(len(pages))]
    with open(root / "items.jsonl", "w") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    files = sorted(RAW.glob("validation-*.parquet"))
    manifest = {
        "name": args.name,
        "source": {"repo": REPO, "revision": REVISION, "split": "validation",
                   "files_sha256": {f.name: hashlib.sha256(f.read_bytes()).hexdigest()[:16] for f in files},
                   "license": "Apache-2.0 (HF tag); DocVQA terms apply"},
        "seed": args.seed,
        "sampling": {"per_skill": args.per_skill, "skills": SKILLS, "one_question_per_page": True,
                     "min_same_kind_distractors": 2},
        "image_transform": f"greyscale, longest side <= {args.max_side} px (LANCZOS), JPEG quality {args.jpeg_quality}",
        "max_kb": args.max_kb,
        "replaced_too_big": too_big,
        "page_order": order,
        "pages": pages,
        "questions": {"version": QUESTION_VERSION, "state": STATE,
                      "choice": "the DocVQA question; options = true answer + up to 3 same-page answers of the same kind; ids a-d shuffled",
                      "noul": 'Question about this page: "<q>" Is the answer "<value>"?',
                      "noul_false": "a same-page answer of the same kind (never a near-duplicate of an accepted answer)"},
        "counts": {"pages": len(pages), "items": len(items),
                   "by_skill": {s: sum(1 for p in pages if p["skill"] == s) for s in SKILLS}},
        "labels_checked": False,
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {root}: {len(pages)} pages, {len(items)} questions, by skill {manifest['counts']['by_skill']}")


if __name__ == "__main__":
    main()
