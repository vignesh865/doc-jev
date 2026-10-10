"""Fill template.html with numbers read from the saved Clef replies.

    .venv/bin/python docs/blog/build.py       # -> docs/blog/clef-docs.html, docs/blog/tables.md

No number in the post's text, charts or boxes is typed by hand: every one comes
from outputs/, experiments/ and data/, so re-running an experiment and
rebuilding updates the post.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from prepare_cord import amount_value  # noqa: E402

PRICE = {"clef-flash": 0.09e-6, "clef": 0.24e-6}  # list price per input token, Workers AI
REPORTED = ("E01", "E02", "E03", "E04", "E05")
SKILL_NAMES = {"table/list": "tables and lists", "layout": "page layout", "form": "forms",
               "free_text": "running text", "handwritten": "handwriting", "figure/diagram": "charts and diagrams"}   # E06 was stopped by the daily cap; not reported


def jl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open()]


def items(data: str) -> dict:
    return {i["item_id"]: i for i in jl(ROOT / "data" / data / "items.jsonl")}


def replies(exp: str) -> dict:
    return {r["item_id"]: r for r in jl(ROOT / "outputs" / f"{exp}.jsonl") if r["status"] == 200}


def errata(data: str) -> set[str]:
    p = ROOT / "data" / data / "errata.json"
    return {e["item_id"] for e in json.loads(p.read_text())["items"]} if p.exists() else set()


def answer(r: dict) -> dict:
    return r["response"]["result"]["answers"]["q"]


def right(item: dict, a: dict) -> bool:
    if item["kind"] == "choice":
        return max(a["probabilities"], key=a["probabilities"].get) == item["answer"]
    return (a["noul"] > 0.5) == (item["answer"] == "true")


def pct(x: float) -> str:
    return f"{100 * x:.0f}"


def main() -> None:
    m = {e: json.loads((ROOT / "experiments" / e / "metrics.json").read_text()) for e in REPORTED}
    g = {e: m[e]["metrics"]["groups"] for e in REPORTED}

    # Page type (E01), with the blind review of the misses.
    rv = items("rvlcdip-v0")
    review = {r["page_id"]: r for r in jl(ROOT / "data/rvlcdip-v0/label_review/claude_blind_review_E01_misses.jsonl")}
    bands = [(0, .5, "under 50%"), (.5, .7, "50 to 70%"), (.7, .85, "70 to 85%"), (.85, 1.01, "85% and up")]
    conf = {b[2]: {"right": 0, "fits": 0, "wrong": 0} for b in bands}
    best = fits = 0
    for iid, r in replies("E01").items():
        i = rv[iid]
        if i["kind"] != "choice":
            continue
        p = answer(r)["probabilities"]
        top = max(p, key=p.get)
        band = next(b[2] for b in bands if b[0] <= p[top] < b[1])
        rev = review.get(i["page_id"])
        if top == i["answer"]:
            kind = "right"
        else:
            kind = "fits" if top.replace("_", " ") in [c.lower() for c in rev["fitting_categories"]] else "wrong"
        conf[band][kind] += 1
        best += top.replace("_", " ") == (rev["best_label"].lower() if rev else i["gold_label"])
        fits += kind != "wrong"
    n_e01 = g["E01"]["choice"]["n"]
    verdicts = {v: sum(r["dataset_label_verdict"] == v for r in review.values())
                for v in ("correct", "acceptable", "not_visible", "wrong")}
    high = {k: sum(conf[b][k] for b in ("70 to 85%", "85% and up")) for k in ("right", "fits", "wrong")}

    # Receipts: a changed total accepted, by size of change (E03 Flash, E04 27B).
    ci = items("cord-v0")
    r3, r4 = replies("E03"), replies("E04")
    size_bands = [(0, .01, "under 1%"), (.01, .10, "1 to 10%"), (.10, 100, "10% or more")]
    fooled = {b[2]: {"n": 0, "flash": 0, "big": 0} for b in size_bands}
    for iid, i in ci.items():
        if i["kind"] != "noul_false":
            continue
        rel = abs(amount_value(i["asked_value"]) - amount_value(i["gold_label"])) / amount_value(i["gold_label"])
        b = next(b[2] for b in size_bands if b[0] <= rel < b[1])
        fooled[b]["n"] += 1
        fooled[b]["flash"] += answer(r3[iid])["noul"] > .5
        fooled[b]["big"] += answer(r4[iid])["noul"] > .5
    choice3 = [right(ci[k], answer(r)) for k, r in r3.items() if ci[k]["kind"] == "choice"]

    # The receipt shown in the post (c044) and the "40.001" example (c052).
    ex = {e: answer(replies(e)["c044-noul_false"])["noul"] for e in ("E03", "E04")}
    ex_q = ci["c044-noul_false"]
    near = ci["c052-noul_false"]
    near_no = 1 - answer(r3["c052-noul_false"])["noul"]

    # Business pages (E05), errata excluded.
    di, bad = items("docvqa-v0"), errata("docvqa-v0")
    skill: dict[str, list[int]] = {}
    doc_choice = []
    for iid, r in replies("E05").items():
        if iid in bad:
            continue
        ok = right(di[iid], answer(r))
        s = skill.setdefault(di[iid]["skill"], [0, 0])
        s[0] += ok
        s[1] += 1
        if di[iid]["kind"] == "choice":
            doc_choice.append(ok)
    skill_acc = {k: v[0] / v[1] for k, v in skill.items()}

    # Calls, tokens, cost, speed over the reported experiments.
    calls = tokens = 0
    cost = 0.0
    flash_lat = []
    for e in REPORTED:
        model = m[e]["config"]["model"]
        for r in replies(e).values():
            t = r["response"]["result"]["usage"]["input_tokens"]
            calls += 1
            tokens += t
            cost += t * PRICE[model]
            if model == "clef-flash" and e in ("E03", "E05"):
                flash_lat.append(r["latency_s"])
    flash_lat.sort()

    # The receipt box: CORD's own labels for receipt c044 (row 44 of the test split).
    import pyarrow.parquet as pq
    gt = json.loads(pq.read_table(ROOT / "data/_raw/cord-v2-test.parquet", columns=["ground_truth"])
                    .to_pylist()[int(ex_q["page_id"][1:])]["ground_truth"])["gt_parse"]
    menu = gt["menu"] if isinstance(gt["menu"], list) else [gt["menu"]]
    tot = gt["total"]
    lines = [f'<div class="line"><span>{x.get("cnt", "")} {x["nm"]}</span><span>{x["price"]}</span></div>' for x in menu]
    lines.append(f'<div class="line total"><span>TOTAL</span><span>{tot["total_price"]}</span></div>')
    for key, label in (("cashprice", "CASH"), ("changeprice", "CHANGE")):
        if key in tot:
            lines.append(f'<div class="line"><span>{label}</span><span>{tot[key]}</span></div>')
    assert tot["total_price"] == ex_q["gold_label"]

    # The request/response box: a real call from E03.
    box = replies("E03")["c044-noul_false"]

    data = {
        "tasks": [
            {"name": "Document · page type, dataset labels", "acc": g["E01"]["choice"]["accuracy"]},
            {"name": "Document · page type, labels checked", "acc": best / n_e01},
            {"name": "Content · receipt totals", "acc": g["E03"]["choice"]["accuracy"]},
            {"name": "Content · business pages", "acc": g["E05"]["choice"]["accuracy"]},
        ],
        "skills": [{"name": SKILL_NAMES[k], "acc": v[0] / v[1], "n": v[1]} for k, v in skill.items()],
        "confidence": [{"band": b[2], **conf[b[2]]} for b in bands],
        "fooled": [{"band": b[2], **fooled[b[2]]} for b in size_bands],
    }
    v = {
        "__DATA__": json.dumps(data),
        "__E01_PAGES__": str(m["E01"]["pages"]),
        "__E01_ACC__": pct(g["E01"]["choice"]["accuracy"]),
        "__E02_ACC__": pct(g["E02"]["choice"]["accuracy"]),
        "__E01_CHECKED__": pct(best / n_e01),
        "__E01_FITS__": pct(fits / n_e01),
        "__MISSES__": str(len(review)),
        "__LABEL_PROBLEMS__": str(verdicts["wrong"] + verdicts["not_visible"] + verdicts["acceptable"]),
        "__LBL_WRONG__": str(verdicts["wrong"]),
        "__LBL_NOTVIS__": str(verdicts["not_visible"]),
        "__LBL_AMBIG__": str(verdicts["acceptable"]),
        "__HIGH_N__": str(sum(high.values())),
        "__HIGH_WRONG__": str(high["wrong"]),
        "__CORD_N__": str(m["E03"]["pages"]),
        "__CORD_RIGHT__": str(sum(choice3)),
        "__CORD_CHOICE_N__": str(len(choice3)),
        "__CORD_ACC__": pct(g["E03"]["choice"]["accuracy"]),
        "__FOOLED_FLASH__": str(sum(f["flash"] for f in fooled.values())),
        "__FOOLED_BIG__": str(sum(f["big"] for f in fooled.values())),
        "__FOOLED_N__": str(sum(f["n"] for f in fooled.values())),
        "__BIG_CORD_ACC__": pct(g["E04"]["choice"]["accuracy"]),
        "__DOC_PAGES__": str(m["E05"]["pages"]),
        "__DOC_RIGHT__": str(sum(doc_choice)),
        "__DOC_N__": str(len(doc_choice)),
        "__DOC_ACC__": pct(sum(doc_choice) / len(doc_choice)),
        "__EX_ASKED__": ex_q["asked_value"],
        "__EX_LINES__": "\n    ".join(lines),
        "__EX_FLASH_NO__": pct(1 - ex["E03"]),
        "__EX_BIG_YES__": pct(ex["E04"]),
        "__NEAR_ASKED__": near["asked_value"],
        "__NEAR_TRUE__": near["gold_label"],
        "__NEAR_NO__": pct(near_no),
        "__CALLS__": f"{calls:,}",
        "__TOKENS__": f"{tokens / 1e6:.1f}",
        "__LIST_COST__": f"{cost:.2f}",
        "__P50_MS__": f"{1000 * flash_lat[len(flash_lat) // 2]:.0f}",
        "__BOX_STATE__": box["request"]["state"],
        "__BOX_Q__": box["request"]["questions"]["q"]["instructions"],
        "__BOX_P__": f"{answer(box)['noul']:.4f}",
        "__BOX_TOKENS__": str(box["response"]["result"]["usage"]["input_tokens"]),
    }
    html = (HERE / "template.html").read_text()
    unused = [k for k in v if k not in html]
    if unused:
        print("computed but not used in the page:", ", ".join(unused))
    for key, value in v.items():
        html = html.replace(key, value)
    assert "__" not in html, [w for w in html.split() if "__" in w][:5]
    (HERE / "clef-docs.html").write_text(html)

    # tables.md: the main numbers as Markdown, for Medium and LinkedIn.
    rows = [("Page type (RVL-CDIP)", "E01", "E02"), ("Receipt totals (CORD)", "E03", "E04"), ("Business pages (DocVQA)", "E05", None)]
    t = ["# Tables", "", "Generated by `docs/blog/build.py` from `experiments/`. Do not edit.", "",
         "## Accuracy by question type", "",
         "| task | model | pages | multiple choice | yes/no, true answer | yes/no, wrong answer |",
         "|---|---|---|---|---|---|"]
    for name, flash, big in rows:
        for e in (flash, big):
            if e:
                gg = g[e]
                t.append(f"| {name} | {m[e]['config']['model']} | {m[e]['pages']} | {pct(gg['choice']['accuracy'])}% "
                         f"| {pct(gg['noul_true']['accuracy'])}% | {pct(gg['noul_false']['accuracy'])}% |")
    t += ["", "## Page type: answers by the model's confidence (Clef-flash, 60 pages)", "",
          "| confidence | matches the label | different, but fits the page | really wrong |", "|---|---|---|---|"]
    t += [f"| {c['band']} | {c['right']} | {c['fits']} | {c['wrong']} |" for c in data["confidence"]]
    t += ["", "## Receipts: changed totals accepted as true", "",
          "| wrong total is off by | questions | Clef-flash said yes | Clef 27B said yes |", "|---|---|---|---|"]
    t += [f"| {f['band']} | {f['n']} | {f['flash']} | {f['big']} |" for f in data["fooled"]]
    t += ["", "## Business pages by skill (Clef-flash, all three question kinds)", "",
          "| skill | right | questions |", "|---|---|---|"]
    t += [f"| {k} | {v_[0]} | {v_[1]} |" for k, v_ in skill.items()]
    (HERE / "tables.md").write_text("\n".join(t) + "\n")
    print(f"wrote docs/blog/clef-docs.html and tables.md ({calls} calls, list cost ${cost:.2f})")


if __name__ == "__main__":
    main()
