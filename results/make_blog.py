"""Numbers and figures for docs/blog/, derived only from outputs/, experiments/ and data/.

    .venv/bin/python results/make_blog.py

Writes docs/blog/numbers.json (every number the post quotes) and
docs/blog/figures/*.png. Never edit those by hand; re-run this script.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from prepare_cord import amount_value  # noqa: E402

OUT = ROOT / "docs" / "blog"
FIG = OUT / "figures"
# Reference palette (dataviz skill, light mode): surface, text, categorical slots 1-3.
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"


def metrics(exp: str) -> dict:
    return json.loads((ROOT / "experiments" / exp / "metrics.json").read_text())


def items(data: str) -> dict:
    return {i["item_id"]: i for i in map(json.loads, (ROOT / "data" / data / "items.jsonl").open())}


def replies(exp: str) -> dict:
    out = {}
    for r in map(json.loads, (ROOT / "outputs" / f"{exp}.jsonl").open()):
        if r["status"] == 200:
            out[r["item_id"]] = r["response"]["result"]["answers"]["q"]
    return out


plt.rcParams.update({"font.family": "Avenir Next", "font.size": 10, "text.color": INK,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK})


def style(ax, grid_axis="x"):
    ax.set_facecolor(SURFACE)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.tick_params(length=0, labelsize=10, pad=6)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, lw=0.7)
    ax.set_axisbelow(True)


def header(fig, title, subtitle):
    fig.text(0.03, 0.95, title, fontsize=14, fontweight="demibold", color=INK, va="top")
    fig.text(0.03, 0.885, subtitle, fontsize=10, color=INK2, va="top")


def legend_row(fig, entries, y=0.80):
    x = 0.03
    for color, label in entries:
        fig.patches.append(FancyBboxPatch((x, y - 0.012), 0.014, 0.026, boxstyle="round,pad=0,rounding_size=0.004",
                                          transform=fig.transFigure, facecolor=color, edgecolor="none"))
        t = fig.text(x + 0.022, y, label, fontsize=9.5, color=INK2, va="center")
        x += 0.022 + len(label) * 0.0105 + 0.035


def rounded_bar(ax, x0, y0, w, h, color, radius_px=4):
    """Bar with rounded corners, radius in screen pixels whatever the data scale."""
    if w <= 0 or h <= 0:
        return
    fig = ax.figure
    fig.canvas.draw()
    (px0, py0), (px1, py1) = ax.transData.transform([(0, 0), (1, 1)])
    ux, uy = abs(px1 - px0), abs(py1 - py0)  # pixels per data unit
    r = min(radius_px / ux, w / 2, (h * uy / ux) / 2)
    ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                mutation_aspect=ux / uy, facecolor=color, edgecolor="none"))


def save(fig, name, top=0.74):
    fig.patch.set_facecolor(SURFACE)
    fig.subplots_adjust(top=top)
    fig.savefig(FIG / name, dpi=220, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    n = {}
    g = {e: metrics(e)["metrics"]["groups"] for e in ("E01", "E02", "E03", "E04", "E05")}
    for e in g:
        n[e] = {k: {"n": v["n"], "accuracy": v["accuracy"], "ece": v["ece"]} for k, v in g[e].items()}
        n[e]["pages"] = metrics(e)["pages"]
        n[e]["latency_p50_s"] = metrics(e)["metrics"]["latency_s"]["p50"]

    # Page type (E01) with the blind review of the misses.
    it = items("rvlcdip-v0")
    rev = {r["page_id"]: r for r in map(json.loads, (ROOT / "data/rvlcdip-v0/label_review/claude_blind_review_E01_misses.jsonl").open())}
    bands = [(0, .5, "under 50%"), (.5, .7, "50–70%"), (.7, .85, "70–85%"), (.85, 1.01, "85% and up")]
    counts = {b[2]: {"right": 0, "fits": 0, "wrong": 0} for b in bands}
    best = fits_total = 0
    for iid, a in replies("E01").items():
        i = it[iid]
        if i["kind"] != "choice":
            continue
        p = a["probabilities"]
        top = max(p, key=p.get)
        band = next(b[2] for b in bands if b[0] <= p[top] < b[1])
        if top == i["answer"]:
            kind = "right"
        else:
            r = rev[i["page_id"]]
            kind = "fits" if top.replace("_", " ") in [c.lower() for c in r["fitting_categories"]] else "wrong"
        counts[band][kind] += 1
        ok_best = top.replace("_", " ") == (rev[i["page_id"]]["best_label"].lower() if i["page_id"] in rev else i["gold_label"])
        best += ok_best
        fits_total += kind != "wrong"
    n["E01_review"] = {"by_confidence": counts, "reviewer_best_label_correct": best, "fits_page": fits_total,
                       "verdicts": {v: sum(1 for r in rev.values() if r["dataset_label_verdict"] == v)
                                    for v in ("correct", "acceptable", "not_visible", "wrong")}}

    # Receipts: false "yes" to a changed total, by size of change (E03 Flash vs E04 27B).
    ci = items("cord-v0")
    r3, r4 = replies("E03"), replies("E04")
    size_bands = [(0, .01, "under 1%"), (.01, .10, "1–10%"), (.10, 100, "10% or more")]
    fooled = {b[2]: {"n": 0, "flash": 0, "27b": 0} for b in size_bands}
    for iid, i in ci.items():
        if i["kind"] != "noul_false":
            continue
        rel = abs(amount_value(i["asked_value"]) - amount_value(i["gold_label"])) / amount_value(i["gold_label"])
        b = next(b[2] for b in size_bands if b[0] <= rel < b[1])
        fooled[b]["n"] += 1
        fooled[b]["flash"] += r3[iid]["noul"] > 0.5
        fooled[b]["27b"] += r4[iid]["noul"] > 0.5
    n["cord_fooled_by_change_size"] = fooled

    # DocVQA by skill (E05).
    di = items("docvqa-v0")
    errata = {e["item_id"] for e in json.loads((ROOT / "data/docvqa-v0/errata.json").read_text())["items"]}
    by_skill = {}
    for iid, a in replies("E05").items():
        if iid in errata:
            continue
        i = di[iid]
        ok = (max(a["probabilities"], key=a["probabilities"].get) == i["answer"]) if i["kind"] == "choice" \
            else ((a["noul"] > .5) == (i["answer"] == "true"))
        s = by_skill.setdefault(i["skill"], [0, 0])
        s[0] += ok
        s[1] += 1
    n["docvqa_by_skill"] = {k: {"right": v[0], "n": v[1]} for k, v in by_skill.items()}
    (OUT / "numbers.json").write_text(json.dumps(n, indent=1) + "\n")

    # Figure 1: Clef-flash, one bar per task (choice-question accuracy).
    labels = ["Page type · dataset labels", "Page type · labels checked", "Receipt totals", "Business-page questions"]
    vals = [g["E01"]["choice"]["accuracy"], best / g["E01"]["choice"]["n"],
            g["E03"]["choice"]["accuracy"], g["E05"]["choice"]["accuracy"]]
    fig, ax = plt.subplots(figsize=(8, 4))
    style(ax)
    ax.set_xlim(0, 1.1)
    ax.set_ylim(-0.6, len(labels) - 0.4)
    ys = list(range(len(labels)))[::-1]
    for y, v in zip(ys, vals):
        rounded_bar(ax, 0, y - 0.27, v, 0.54, BLUE)
        ax.text(v + 0.015, y, f"{v:.0%}", va="center", fontsize=11, fontweight="demibold", color=INK)
    ax.set_yticks(ys, labels)
    ax.set_xticks([0, .25, .5, .75, 1], ["0%", "25%", "50%", "75%", "100%"])
    header(fig, "Reading content is easy; labelling page types is harder",
           "Clef-flash, share of multiple-choice questions answered correctly")
    fig.subplots_adjust(left=0.27, right=0.96, bottom=0.1)
    save(fig, "1-accuracy-by-task.png", top=0.80)

    # Figure 2: page-type answers by confidence: right / wrong-but-fits-the-page / really wrong.
    fig, ax = plt.subplots(figsize=(8, 4.3))
    style(ax)
    names = [b[2] for b in bands][::-1]
    totals = [sum(counts[nm].values()) for nm in names]
    ax.set_xlim(0, max(totals) * 1.08)
    ax.set_ylim(-0.6, len(names) - 0.4)
    series = (("right", BLUE, "matches the dataset label", "white"),
              ("fits", ORANGE, "different, but fits the page", "white"),
              ("wrong", AQUA, "really wrong", INK))
    gap = 0.12
    for y, nm in zip(range(len(names)), names):
        left = 0
        for key, color, _, txt in series:
            w = counts[nm][key]
            if w:
                rounded_bar(ax, left, y - 0.27, w - gap, 0.54, color)
                ax.text(left + (w - gap) / 2, y, str(w), ha="center", va="center", fontsize=10,
                        fontweight="demibold", color=txt)
            left += w
    ax.set_yticks(range(len(names)), [f"confidence {nm}" for nm in names])
    ax.set_xlabel("pages", fontsize=9.5)
    header(fig, "When Clef-flash was confident, its answer fit the page",
           "Page-type questions on 60 pages, grouped by how sure the model was")
    legend_row(fig, [(c, l) for _, c, l, _ in series], y=0.80)
    fig.subplots_adjust(left=0.22, right=0.96, bottom=0.14)
    save(fig, "2-page-type-confidence.png", top=0.74)

    # Figure 3: receipts, fooled by a changed total, by size of change.
    fig, ax = plt.subplots(figsize=(8, 4.3))
    style(ax, grid_axis="y")
    names = [b[2] for b in size_bands]
    w = 0.34
    top = max(fooled[nm][k] / fooled[nm]["n"] for nm in names for k in ("flash", "27b"))
    ax.set_ylim(0, top * 1.3)
    ax.set_xlim(-0.6, len(names) - 0.4)
    for off, key, color in ((-w / 2 - 0.01, "flash", BLUE), (w / 2 + 0.01, "27b", ORANGE)):
        for x, nm in enumerate(names):
            v = fooled[nm][key] / fooled[nm]["n"]
            rounded_bar(ax, x + off - w / 2, 0, w, max(v, 0.0015), color)
            ax.text(x + off, v + top * 0.03, f"{fooled[nm][key]} of {fooled[nm]['n']}", ha="center",
                    fontsize=9.5, color=INK)
    ax.set_xticks(range(len(names)), [f"off by {nm}" for nm in names])
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    header(fig, "The bigger model waves through near-miss totals",
           "Receipts: how often each model said “yes” to a total with one digit changed")
    legend_row(fig, [(BLUE, "Clef-flash (9B)"), (ORANGE, "Clef (27B)")], y=0.80)
    fig.subplots_adjust(left=0.09, right=0.97, bottom=0.12)
    save(fig, "3-near-miss-totals.png", top=0.74)
    print(json.dumps({k: n[k] for k in ("E01_review", "cord_fooled_by_change_size", "docvqa_by_skill")}, indent=1))


if __name__ == "__main__":
    main()
