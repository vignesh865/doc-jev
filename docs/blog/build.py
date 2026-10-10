"""Fill template.html with numbers read from the saved Clef replies.

    .venv/bin/python docs/blog/build.py   # -> clef-docs.html, clef-docs-ste.html, tables.md

No number in the post's text, charts or boxes is typed by hand: every one comes
from outputs/, experiments/ and data/, so re-running an experiment and
rebuilding updates the post. The build fails if a placeholder is left unfilled
or a computed number is not used.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from prepare_cord import amount_value  # noqa: E402

PRICE = {"clef-flash": 0.09e-6, "clef": 0.24e-6}  # list price per input token, Workers AI
ALL_RUNS = ("E01", "E02", "E03", "E04", "E05", "E06", "E07", "E08", "E09", "E10")
CONTAM = ("E07", "E08", "E09", "E10")
PAGES = {"template.html": "clef-docs.html", "template-ste.html": "clef-docs-ste.html"}
SKILL_NAMES = {"table/list": "tables and lists", "layout": "page layout", "form": "forms",
               "free_text": "running text", "handwritten": "handwriting", "figure/diagram": "charts and diagrams"}


def jl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open()]


def items(data: str) -> dict:
    return {i["item_id"]: i for i in jl(ROOT / "data" / data / "items.jsonl")}


def errata(data: str) -> set[str]:
    p = ROOT / "data" / data / "errata.json"
    return {e["item_id"] for e in json.loads(p.read_text())["items"]} if p.exists() else set()


def replies(exp: str, bad: set[str] = frozenset()) -> dict:
    return {r["item_id"]: r for r in jl(ROOT / "outputs" / f"{exp}.jsonl")
            if r["status"] == 200 and r["item_id"] not in bad}


def answer(r: dict) -> dict:
    return r["response"]["result"]["answers"]["q"]


def right(item: dict, a: dict) -> bool:
    if item["kind"] == "choice":
        return max(a["probabilities"], key=a["probabilities"].get) == item["answer"]
    return (a["noul"] > 0.5) == (item["answer"] == "true")


def pct(x: float) -> str:
    return f"{100 * x:.0f}"


def main() -> None:
    m = {e: json.loads((ROOT / "experiments" / e / "metrics.json").read_text()) for e in ALL_RUNS}
    rv, ci, di = items("rvlcdip-v0"), items("cord-v0"), items("docvqa-v0")
    dbad = errata("docvqa-v0")
    its = {"rvlcdip-v0": rv, "cord-v0": ci, "docvqa-v0": di}
    R = {e: replies(e, dbad if m[e]["config"]["data"] == "docvqa-v0" else frozenset()) for e in ALL_RUNS}

    def A(e: str, kind: str) -> tuple[int, int]:
        it = its[m[e]["config"]["data"]]
        ids = [k for k in R[e] if it[k]["kind"] == kind]
        return sum(right(it[k], answer(R[e][k])) for k in ids), len(ids)

    def P(e: str, kind: str) -> float:
        r, n = A(e, kind)
        return r / n

    # Part 1: page type (E01), with the blind review of the misses.
    review = {r["page_id"]: r for r in jl(ROOT / "data/rvlcdip-v0/label_review/claude_blind_review_E01_misses.jsonl")}
    bands = [(0, .5, "under 50%"), (.5, .7, "50 to 70%"), (.7, .85, "70 to 85%"), (.85, 1.01, "85% and up")]
    conf = {b[2]: {"right": 0, "fits": 0, "wrong": 0} for b in bands}
    best = fits = 0
    for iid, r in R["E01"].items():
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
    n_e01 = A("E01", "choice")[1]
    verdicts = {v: sum(r["dataset_label_verdict"] == v for r in review.values())
                for v in ("correct", "acceptable", "not_visible", "wrong")}
    high = {k: sum(conf[b][k] for b in ("70 to 85%", "85% and up")) for k in ("right", "fits", "wrong")}

    # Part 2: receipts, a changed total accepted, by size of change (E03 Flash, E04 27B).
    size_bands = [(0, .01, "under 1%"), (.01, .10, "1 to 10%"), (.10, 100, "10% or more")]
    fooled = {b[2]: {"n": 0, "flash": 0, "big": 0} for b in size_bands}
    for iid, i in ci.items():
        if i["kind"] != "noul_false":
            continue
        rel = abs(amount_value(i["asked_value"]) - amount_value(i["gold_label"])) / amount_value(i["gold_label"])
        b = next(b[2] for b in size_bands if b[0] <= rel < b[1])
        fooled[b]["n"] += 1
        fooled[b]["flash"] += answer(R["E03"][iid])["noul"] > .5
        fooled[b]["big"] += answer(R["E04"][iid])["noul"] > .5
    ex_q, near = ci["c044-noul_false"], ci["c052-noul_false"]

    # Part 2: business pages by skill, both models, all three question kinds.
    skill = defaultdict(lambda: {"flash": 0, "big": 0, "n": 0})
    for iid in R["E05"]:
        s = skill[di[iid]["skill"]]
        s["flash"] += right(di[iid], answer(R["E05"][iid]))
        s["big"] += right(di[iid], answer(R["E06"][iid]))
        s["n"] += 1
    big_no_true = A("E06", "noul_true")[1] - A("E06", "noul_true")[0]

    # Part 3: contamination (real E03/E05 vs blank E07/E09 vs swapped E08/E10).
    def chance(data: str) -> float:
        ids = [k for k, i in its[data].items() if i["kind"] == "choice" and k not in dbad]
        return sum(1 / its[data][k]["n_options"] for k in ids) / len(ids)

    contam = [
        {"name": "Receipts · multiple choice", "real": P("E03", "choice"), "blank": P("E07", "choice"),
         "swapped": P("E08", "choice"), "chance": chance("cord-v0")},
        {"name": "Receipts · is the total X? (true X)", "real": P("E03", "noul_true"),
         "blank": P("E07", "noul_true"), "swapped": P("E08", "noul_true"), "chance": None},
        {"name": "Business pages · multiple choice", "real": P("E05", "choice"), "blank": P("E09", "choice"),
         "swapped": P("E10", "choice"), "chance": chance("docvqa-v0")},
        {"name": "Business pages · is the answer X? (true X)", "real": P("E05", "noul_true"),
         "blank": P("E09", "noul_true"), "swapped": P("E10", "noul_true"), "chance": None},
    ]
    leak = defaultdict(lambda: [0, 0])
    for iid, r in R["E09"].items():
        if di[iid]["kind"] == "choice":
            kind = di[iid]["answer_kind"].split(":")[0]
            key = "words" if kind == "words" else "numbers" if kind == "number" else "other"
            leak[key][0] += right(di[iid], answer(r))
            leak[key][1] += 1

    # Calls, tokens, cost, speed.
    calls = tokens = 0
    cost = 0.0
    flash_lat = []
    for e in ALL_RUNS:
        model = m[e]["config"]["model"]
        for r in jl(ROOT / "outputs" / f"{e}.jsonl"):
            if r["status"] != 200:
                continue
            t = r["response"]["result"]["usage"]["input_tokens"]
            calls += 1
            tokens += t
            cost += t * PRICE[model]
            if e in ("E03", "E05"):
                flash_lat.append(r["latency_s"])
    flash_lat.sort()
    contam_calls = sum(1 for e in CONTAM for r in jl(ROOT / "outputs" / f"{e}.jsonl") if r["status"] == 200)

    # The receipt box: CORD's own labels for receipt c044 (row 44 of the test split).
    import pyarrow.parquet as pq
    gt = json.loads(pq.read_table(ROOT / "data/_raw/cord-v2-test.parquet", columns=["ground_truth"])
                    .to_pylist()[int(ex_q["page_id"][1:])]["ground_truth"])["gt_parse"]
    menu = gt["menu"] if isinstance(gt["menu"], list) else [gt["menu"]]
    tot = gt["total"]
    assert tot["total_price"] == ex_q["gold_label"]
    lines = [f'<div class="line"><span>{x.get("cnt", "")} {x["nm"]}</span><span>{x["price"]}</span></div>' for x in menu]
    lines.append(f'<div class="line total"><span>TOTAL</span><span>{tot["total_price"]}</span></div>')
    for key, label in (("cashprice", "CASH"), ("changeprice", "CHANGE")):
        if key in tot:
            lines.append(f'<div class="line"><span>{label}</span><span>{tot[key]}</span></div>')
    box = R["E03"]["c044-noul_false"]

    data = {
        "overview": [
            {"name": "Page type", "flash": P("E01", "choice"), "big": P("E02", "choice")},
            {"name": "Receipt totals", "flash": P("E03", "choice"), "big": P("E04", "choice")},
            {"name": "Business pages", "flash": P("E05", "choice"), "big": P("E06", "choice")},
        ],
        "confidence": [{"band": b[2], **conf[b[2]]} for b in bands],
        "skills": [{"name": SKILL_NAMES[k], "flash": s["flash"] / s["n"], "big": s["big"] / s["n"], "n": s["n"]}
                   for k, s in skill.items()],
        "fooled": [{"band": b[2], **fooled[b[2]]} for b in size_bands],
        "contam": contam,
    }
    cr, cd = contam[1], contam[3]
    v = {
        "__DATA__": json.dumps(data),
        "__E01_PAGES__": str(m["E01"]["pages"]),
        "__E01_ACC__": pct(P("E01", "choice")),
        "__E02_ACC__": pct(P("E02", "choice")),
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
        "__CORD_RIGHT__": str(A("E03", "choice")[0]),
        "__CORD_CHOICE_N__": str(A("E03", "choice")[1]),
        "__CORD_ACC__": pct(P("E03", "choice")),
        "__BIG_CORD_ACC__": pct(P("E04", "choice")),
        "__FOOLED_FLASH__": str(sum(f["flash"] for f in fooled.values())),
        "__FOOLED_BIG__": str(sum(f["big"] for f in fooled.values())),
        "__FOOLED_N__": str(sum(f["n"] for f in fooled.values())),
        "__DOC_PAGES__": str(m["E05"]["pages"]),
        "__DOC_RIGHT__": str(A("E05", "choice")[0]),
        "__DOC_N__": str(A("E05", "choice")[1]),
        "__DOC_ACC__": pct(P("E05", "choice")),
        "__DOC_BIG_ACC__": pct(P("E06", "choice")),
        "__BIG_NO_TRUE__": str(big_no_true),
        "__EX_LINES__": "\n    ".join(lines),
        "__EX_ASKED__": ex_q["asked_value"],
        "__EX_FLASH_NO__": pct(1 - answer(R["E03"]["c044-noul_false"])["noul"]),
        "__EX_BIG_YES__": pct(answer(R["E04"]["c044-noul_false"])["noul"]),
        "__NEAR_ASKED__": near["asked_value"],
        "__NEAR_TRUE__": near["gold_label"],
        "__NEAR_NO__": pct(1 - answer(R["E03"]["c052-noul_false"])["noul"]),
        "__CR_REAL__": pct(cr["real"]), "__CR_BLANK__": pct(cr["blank"]), "__CR_SWAP__": pct(cr["swapped"]),
        "__CD_REAL__": pct(cd["real"]), "__CD_BLANK__": pct(cd["blank"]), "__CD_SWAP__": pct(cd["swapped"]),
        "__CRC_BLANK__": pct(contam[0]["blank"]), "__CRC_SWAP__": pct(contam[0]["swapped"]),
        "__CRC_CHANCE__": pct(contam[0]["chance"]),
        "__CDC_BLANK__": pct(contam[2]["blank"]), "__CDC_SWAP__": pct(contam[2]["swapped"]),
        "__CDC_CHANCE__": pct(contam[2]["chance"]),
        "__LEAK_WORDS__": f"{leak['words'][0]} of {leak['words'][1]}",
        "__LEAK_NUMS__": f"{leak['numbers'][0]} of {leak['numbers'][1]}",
        "__CONTAM_CALLS__": f"{contam_calls:,}",
        "__CALLS__": f"{calls:,}",
        "__TOKENS__": f"{tokens / 1e6:.1f}",
        "__LIST_COST__": f"{cost:.2f}",
        "__P50_MS__": f"{1000 * flash_lat[len(flash_lat) // 2]:.0f}",
        "__BOX_STATE__": box["request"]["state"],
        "__BOX_Q__": box["request"]["questions"]["q"]["instructions"],
        "__BOX_P__": f"{answer(box)['noul']:.4f}",
        "__BOX_TOKENS__": str(box["response"]["result"]["usage"]["input_tokens"]),
    }
    # The same numbers fill the main post and its ASD-STE100 version.
    for template, out in PAGES.items():
        html = (HERE / template).read_text()
        unused = [k for k in v if k not in html]
        assert not unused, f"{template}: computed but not used in the page: {unused}"
        for key, value in v.items():
            html = html.replace(key, value)
        assert "__" not in html, (template, [w for w in html.split() if "__" in w][:5])
        (HERE / out).write_text(html)

    # tables.md: the main numbers as Markdown, for Medium and LinkedIn.
    t = ["# Tables", "", "Generated by `docs/blog/build.py`. Do not edit.", "",
         "## Accuracy by test and question kind", "",
         "| kind of question | test | model | multiple choice | yes/no, true answer | yes/no, wrong answer |",
         "|---|---|---|---|---|---|"]
    for umb, name, pair in (("about the document", "Page type (RVL-CDIP)", ("E01", "E02")),
                            ("about the content", "Receipt totals (CORD)", ("E03", "E04")),
                            ("about the content", "Business pages (DocVQA)", ("E05", "E06"))):
        for e in pair:
            t.append(f"| {umb} | {name} | {m[e]['config']['model']} | {pct(P(e, 'choice'))}% "
                     f"| {pct(P(e, 'noul_true'))}% | {pct(P(e, 'noul_false'))}% |")
    t += ["", "## Contamination check (Clef-flash): same questions, different image", "",
          "| measure | real page | blank page | another page | chance |", "|---|---|---|---|---|"]
    t += [f"| {c['name']} | {pct(c['real'])}% | {pct(c['blank'])}% | {pct(c['swapped'])}% | "
          f"{pct(c['chance']) + '%' if c['chance'] else ''} |" for c in contam]
    t += ["", "## Page type: answers by the model's confidence (Clef-flash)", "",
          "| confidence | matches the label | different, but fits the page | really wrong |", "|---|---|---|---|"]
    t += [f"| {c['band']} | {c['right']} | {c['fits']} | {c['wrong']} |" for c in data["confidence"]]
    t += ["", "## Receipts: changed totals accepted as true", "",
          "| wrong total is off by | questions | Clef-flash said yes | Clef 27B said yes |", "|---|---|---|---|"]
    t += [f"| {f['band']} | {f['n']} | {f['flash']} | {f['big']} |" for f in data["fooled"]]
    t += ["", "## Business pages by skill (all three question kinds)", "",
          "| skill | questions | Clef-flash right | Clef 27B right |", "|---|---|---|---|"]
    t += [f"| {SKILL_NAMES[k]} | {s['n']} | {s['flash']} | {s['big']} |" for k, s in skill.items()]
    (HERE / "tables.md").write_text("\n".join(t) + "\n")
    print(f"wrote docs/blog/clef-docs.html, clef-docs-ste.html and tables.md ({calls} calls, list cost ${cost:.2f})")


if __name__ == "__main__":
    main()
