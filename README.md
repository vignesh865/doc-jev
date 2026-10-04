# DocJev

**Calibrated, non-autoregressive typed decisions over document images.**

Give DocJev a document page and typed questions (`noul` yes/no, `choice`,
`score`); get back a probability distribution per question in a single pass, with
no text generation. The goal is probabilities honest enough that you can decide
*how much to automate at a fixed error rate*.

Status: research, early. Inspired by Jev, Laya, CLM and Visual Jev (see
[docs/prior_art.md](docs/prior_art.md)); built from first principles.

- Research journal (decisions and results): [JOURNAL.md](JOURNAL.md)
- Learning journal (what we learned, in plain words): [LEARNING.md](LEARNING.md)
- Draft research plan: [docs/research_plan.md](docs/research_plan.md)
- Metrics (accuracy, NLL, Brier, ECE, risk–coverage, temperature fitting): `src/docjev/metrics.py`

```bash
pip install -e '.[dev]'
pytest
```

License: Apache-2.0.
