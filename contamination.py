"""Compare real-image runs with their blank and swapped twins (contamination check).

    .venv/bin/python contamination.py

Reads outputs/ only. For each data set and question kind: accuracy and the
mean probability given to the correct answer, for real / blank / swapped
images, plus the chance level of the choice questions (1 / number of options).
"""
import json
from pathlib import Path

TWINS = {"cord-v0": ("E03", "E07", "E08"), "docvqa-v0": ("E05", "E09", "E10")}


def rows(exp, items, bad):
    out = {}
    for r in map(json.loads, open(f"outputs/{exp}.jsonl")):
        if r["status"] != 200 or r["item_id"] in bad:
            continue
        it, a = items[r["item_id"]], r["response"]["result"]["answers"]["q"]
        if it["kind"] == "choice":
            p = a["probabilities"]
            out[r["item_id"]] = (max(p, key=p.get) == it["answer"], p[it["answer"]])
        else:
            pt = a["noul"] if it["answer"] == "true" else 1 - a["noul"]
            out[r["item_id"]] = (pt > 0.5, pt)
    return out


def main():
    summary = {}
    for data, exps in TWINS.items():
        items = {i["item_id"]: i for i in map(json.loads, open(f"data/{data}/items.jsonl"))}
        err = Path(f"data/{data}/errata.json")
        bad = {e["item_id"] for e in json.loads(err.read_text())["items"]} if err.exists() else set()
        runs = {name: rows(e, items, bad) for name, e in zip(("real", "blank", "swapped"), exps)}
        common = set.intersection(*(set(r) for r in runs.values()))
        print(f"\n{data}: {len(common)} questions answered in all three runs")
        summary[data] = {}
        for kind in ("choice", "noul_true", "noul_false"):
            ids = [i for i in common if items[i]["kind"] == kind]
            line = {}
            for name, r in runs.items():
                acc = sum(r[i][0] for i in ids) / len(ids)
                mp = sum(r[i][1] for i in ids) / len(ids)
                line[name] = {"accuracy": round(acc, 3), "mean_p_correct": round(mp, 3)}
            extra = ""
            if kind == "choice":
                chance = sum(1 / items[i]["n_options"] for i in ids) / len(ids)
                line["chance"] = round(chance, 3)
                extra = f"  (chance {chance:.2f})"
            summary[data][kind] = {"n": len(ids), **line}
            print(f"  {kind:10s} n={len(ids):3d}  " + "  ".join(
                f"{n}: acc {v['accuracy']:.2f}, p {v['mean_p_correct']:.2f}" for n, v in line.items() if n != "chance") + extra)
    Path("experiments/contamination.json").write_text(json.dumps(summary, indent=1) + "\n")
    print("\nwrote experiments/contamination.json")


if __name__ == "__main__":
    main()
