# DocJev — context for Claude sessions

Research + learning hobby project, open source (Apache-2.0). Discuss before
building: the user and Claude debate ideas from first principles; nothing is
decided until it is in JOURNAL.md with a [U]/[C]/[E] tag.

Read first:
- JOURNAL.md: dated, append-only; the only record of decisions and results.
- LEARNING.md: lessons in plain words, intuition first and formula second, with
  worked examples. Add one whenever something new is learned.

Agreed so far (see JOURNAL.md for detail): a Jev-style typed-decision model for
document page images (noul / choice / score -> a distribution per question, no
generation); trustworthy confidence is the core problem; documents first.
Experiments follow the chain data -> raw outputs (outputs/<exp>.jsonl) ->
experiments/<exp>/metrics.json + experiments/runs.jsonl. Blog material lives in docs/blog/ (template.html -> build.py -> clef-docs.html;
export_png.py -> images/; tables.md), made only when the user asks. Follow the
user's BLOG_PLAYBOOK.md (uft/system1/blog/). Every number comes from build.py.
Never type a result by hand. Push to main.

NOT agreed. These are assumptions from an earlier session; raise them for
discussion, don't apply them silently:
- Which confidence measure to calibrate and set thresholds on (max(p), margin,
  concentration, ...). This is OPEN; see LEARNING.md lesson 1.
- Temperature fitting by NLL; ECE settings; coverage at <=1% / <=5% risk as the
  north star.
- Budget and backbone limits (MLX / Kaggle / RunPod, <= ~4B).
- docs/research_plan.md as a whole.

Dev: `pip install -e '.[dev]' && pytest`. Secrets in .env (git-ignored).
