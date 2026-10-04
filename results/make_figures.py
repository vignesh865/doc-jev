"""Build results/figures/*.png from raw replies. Never edit the figures by hand.

    .venv/bin/python results/make_figures.py --exp E01
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluate import MEASURES, load  # noqa: E402
from docjev import metrics as M  # noqa: E402

OUT = Path("results/figures")


def reliability(ax, rows, title):
    conf = np.array([r["probs"].max() for r in rows])
    correct = np.array([float(r["probs"].argmax() == r["label"]) for r in rows])
    e, bins = M.ece(conf, correct, n_bins=10)
    ax.plot([0, 1], [0, 1], ls="--", color="grey", lw=1)
    for b in bins:
        ax.bar((b["lo"] + b["hi"]) / 2, b["acc"], width=b["hi"] - b["lo"], alpha=0.6 if not b["noisy"] else 0.3,
               edgecolor="black", color="tab:blue")
        ax.text((b["lo"] + b["hi"]) / 2, b["acc"] + 0.02, str(b["n"]), ha="center", fontsize=7)
    ax.set(xlim=(0, 1), ylim=(0, 1.08), xlabel="top probability (bin)", ylabel="fraction correct",
           title=f"{title}\nn={len(rows)}, accuracy {correct.mean():.1%}, ECE {e:.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    args = ap.parse_args()
    config = json.loads((Path("experiments") / args.exp / "config.json").read_text())
    rows = load(config["data"], Path(config["outputs"]))
    name = args.exp
    OUT.mkdir(parents=True, exist_ok=True)

    choice = [r for r in rows if r["kind"] == "choice"]
    noul = [r for r in rows if r["kind"].startswith("noul")]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    reliability(axes[0], choice, "choice (16 types)")
    reliability(axes[1], noul, "yes/no (true + wrong type)")
    fig.suptitle(f"Reliability: {name} (numbers above bars = questions per bin; faded = fewer than 30)", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / f"reliability-{name}.png", dpi=150)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, group, title in ((axes[0], choice, "choice (16 types)"), (axes[1], noul, "yes/no")):
        correct = np.array([float(r["probs"].argmax() == r["label"]) for r in group])
        for mname, f in MEASURES.items():
            conf = np.array([f(r["probs"]) for r in group])
            cov, risk = M.risk_coverage(conf, correct)
            ax.plot(cov, risk, label=mname, lw=1.4)
        for level in (0.01, 0.05):
            ax.axhline(level, color="grey", ls=":", lw=1)
        ax.set(xlabel="coverage (fraction answered, most confident first)", ylabel="error rate among answered",
               title=title, xlim=(0, 1), ylim=(0, max(0.3, 1 - correct.mean() + 0.05)))
        ax.legend(fontsize=8)
    fig.suptitle(f"Risk-coverage by confidence measure: {name}", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / f"risk-coverage-{name}.png", dpi=150)
    print(f"wrote {OUT}/reliability-{name}.png, risk-coverage-{name}.png")


if __name__ == "__main__":
    main()
