"""Score raw model replies against a data set.

    .venv/bin/python evaluate.py --data rvlcdip-v0 --predictions outputs/clef-flash-rvlcdip-v0.jsonl --save

Reads data/<data>/items.jsonl and the raw replies, never calls a model. With
--save, writes experiments/<name>-<id>/metrics.json and appends one line to
experiments/runs.jsonl (append-only; tables use the latest line per run id).

Which confidence measure DocJev will use is undecided (JOURNAL entry 6), so
risk-coverage numbers are reported for every candidate measure.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from docjev import metrics as M

# Replies are rounded to 4 decimals, so 0.0000 means "below 0.00005".
P_FLOOR = 5e-5

MEASURES = {
    "top": lambda p: p.max(),
    "margin": lambda p: np.sort(p)[-1] - np.sort(p)[-2],
    "lead_vs_mean": lambda p: np.sort(p)[-1] - np.sort(p)[:-1].mean(),
    "concentration": lambda p: (len(p) * (p ** 2).sum() - 1) / (len(p) - 1),
}


def load(data: str, predictions: Path) -> list[dict]:
    """One row per item with a successful reply: probabilities, label index, kind."""
    items = {i["item_id"]: i for i in map(json.loads, (Path("data") / data / "items.jsonl").open())}
    replies = {}
    for r in map(json.loads, predictions.open()):
        if r["status"] == 200 and r["response"].get("success"):
            replies[r["item_id"]] = r  # the latest successful reply wins
    rows = []
    for item_id, item in items.items():
        if item_id not in replies:
            continue
        reply = replies[item_id]
        answer = reply["response"]["result"]["answers"]["q"]
        if item["question"]["type"] == "choice":
            options = list(item["question"]["criteria"])
            probs = np.array([answer["probabilities"][o] for o in options], float)
            label = options.index(item["answer"])
            reported = answer.get("confidence")
        else:
            options = ["false", "true"]
            probs = np.array([1.0 - answer["noul"], answer["noul"]], float)
            label = options.index(item["answer"])
            reported = None
        rows.append({
            "item_id": item_id, "page_id": item["page_id"], "kind": item["kind"],
            "gold_label": item["gold_label"], "options": options, "raw_probs": probs,
            "probs": np.maximum(probs, P_FLOOR) / np.maximum(probs, P_FLOOR).sum(),
            "label": label, "reported_confidence": reported,
            "latency_s": reply["latency_s"], "input_tokens": reply["response"]["result"]["usage"]["input_tokens"],
        })
    return rows


def score(rows: list[dict]) -> dict:
    probs = [r["probs"] for r in rows]
    labels = [r["label"] for r in rows]
    out = M.evaluate([np.log(p) for p in probs], labels)
    out["by_measure"] = {}
    correct = np.array([float(p.argmax() == y) for p, y in zip(probs, labels)])
    for name, f in MEASURES.items():
        conf = np.array([f(p) for p in probs])
        out["by_measure"][name] = {
            "aurc": M.aurc(conf, correct),
            "cov@0.01": M.coverage_at_risk(conf, correct, 0.01),
            "cov@0.05": M.coverage_at_risk(conf, correct, 0.05),
        }
    return out


def summarize(rows: list[dict], n_items: int) -> dict:
    groups = defaultdict(list)
    for r in rows:
        groups[r["kind"]].append(r)
        if r["kind"].startswith("noul"):
            groups["noul"].append(r)
    result = {"items": n_items, "scored": len(rows), "missing": n_items - len(rows),
              "groups": {k: score(v) for k, v in sorted(groups.items())}}

    choice = groups["choice"]
    per_class = defaultdict(list)
    for r in choice:
        per_class[r["gold_label"]].append(float(r["probs"].argmax() == r["label"]))
    result["choice_accuracy_by_class"] = {k: float(np.mean(v)) for k, v in sorted(per_class.items())}
    confusion = defaultdict(int)
    for r in choice:
        confusion[(r["gold_label"], r["options"][int(r["probs"].argmax())])] += 1
    result["choice_confusions"] = sorted(
        [{"gold": g, "pred": p, "n": n} for (g, p), n in confusion.items() if g.replace(" ", "_") != p],
        key=lambda d: -d["n"])[:15]

    # Does the hosted "confidence" equal normalised concentration? (LEARNING lesson 1)
    diffs = [abs(r["reported_confidence"] - MEASURES["concentration"](r["raw_probs"]))
             for r in choice if r["reported_confidence"] is not None]
    result["hosted_confidence_vs_concentration_max_abs_diff"] = float(max(diffs)) if diffs else None

    # Do the choice answer and the yes/no answer about the true type agree, per page?
    by_page = defaultdict(dict)
    for r in rows:
        by_page[r["page_id"]][r["kind"]] = r
    pairs = [(p["choice"], p["noul_true"]) for p in by_page.values() if "choice" in p and "noul_true" in p]
    result["choice_vs_noul_true"] = {
        "pages": len(pairs),
        "both_right": sum(c["probs"].argmax() == c["label"] and t["probs"][1] > 0.5 for c, t in pairs),
        "choice_only": sum(c["probs"].argmax() == c["label"] and t["probs"][1] <= 0.5 for c, t in pairs),
        "noul_only": sum(c["probs"].argmax() != c["label"] and t["probs"][1] > 0.5 for c, t in pairs),
        "both_wrong": sum(c["probs"].argmax() != c["label"] and t["probs"][1] <= 0.5 for c, t in pairs),
    }
    lat = np.array([r["latency_s"] for r in rows])
    result["latency_s"] = {"p50": float(np.median(lat)), "p95": float(np.percentile(lat, 95)), "max": float(lat.max())}
    result["input_tokens"] = {"total": int(sum(r["input_tokens"] for r in rows)),
                              "per_request_median": float(np.median([r["input_tokens"] for r in rows]))}
    return result


def jsonable(x):
    if isinstance(x, dict):
        return {k: jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    return x


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="rvlcdip-v0")
    ap.add_argument("--predictions", type=Path, required=True)
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()

    manifest = json.loads((Path("data") / args.data / "manifest.json").read_text())
    rows = load(args.data, args.predictions)
    metrics = summarize(rows, manifest["counts"]["items"])
    pred_sha = hashlib.sha256(args.predictions.read_bytes()).hexdigest()[:16]
    name = args.predictions.stem
    run_id = f"{name}-{pred_sha[:8]}"
    first = json.loads(args.predictions.open().readline())
    record = jsonable({
        "run_id": run_id,
        "scored_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip(),
        "predictions": str(args.predictions), "predictions_sha256": pred_sha,
        "model": first["model"], "model_version": "hosted Workers AI; not pinnable",
        "produced_by": None,
        "data": {"name": args.data, "source": manifest["source"], "labels_checked": manifest["labels_checked"]},
        "p_floor": P_FLOOR,
        "metrics": metrics,
    })
    print(json.dumps(record["metrics"]["groups"], indent=1)[:3000])
    if args.save:
        folder = Path("experiments") / run_id
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "metrics.json").write_text(json.dumps(record, indent=1) + "\n")
        with open(Path("experiments") / "runs.jsonl", "a") as f:
            f.write(json.dumps(record) + "\n")
        print(f"saved {folder}")


if __name__ == "__main__":
    main()
