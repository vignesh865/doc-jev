# DocJev — Research Plan

Draft written by an earlier session; not yet agreed. Decisions live only in
[JOURNAL.md](../JOURNAL.md).

## 1. Motivation

Most document work is **deciding, not writing**: *is this signed? which of 16
types is this? does the total on the page equal 1,234.00? how many line items?*
Today this is done by OCR + LLM or a VLM that generates text: slow, costly per
question, occasionally unparseable, and with confidence nobody can trust.

What a user of such a system actually needs is:

> **How many documents can I let through automatically while keeping the error
> rate below X%?**

That answer depends on accuracy *and* on whether the model's probabilities are
honest. DocJev is a research project on building a small, open, zero-shot model
that makes **typed decisions over document images** with **calibrated
probabilities**, in a single pass, without generating text.

Inspirations (not templates): Jev (typed decisions, text-only, closed), Laya
(cross-encoder + proper-scoring-rule training + per-bucket temperatures), CLM
(frozen backbone + dual-encoder heads), Visual Jev (VLM, shared image prefix,
LM-head readout). See [prior_art.md](prior_art.md).

## 2. North-star metric

**Coverage at fixed risk**: sort predictions by confidence, and report the
largest fraction of questions we can answer while the error rate among those
answered stays ≤ 1% (and ≤ 5%). This combines skill and honesty in one number.

Always reported alongside it: accuracy, NLL, Brier, ECE (15 bins, top-label),
AURC. None of these alone is sufficient (a mediocre model can have ECE≈0).

The implementation lives in `src/docjev/metrics.py`.

## 3. Research questions

- **RQ1: Zero-shot floor.** How good, and how calibrated, is a small
  off-the-shelf VLM when we read its probabilities over the candidate answers
  directly (no generation, no training)?
- **RQ2: Calibration transfer.** Folklore (and our own spec) says calibration
  does not transfer across document domains. Is that true? If a temperature fit
  on domains A,B holds on unseen domain C, zero-shot calibration is possible.
  If not, how few labels in C restore it? *This is the question most likely to
  produce a novel, publishable result.*
- **RQ3: Training for decisions.** Does light training (heads only or LoRA)
  with a proper-scoring-rule loss on a mixture of document tasks improve
  **held-out-domain** coverage@risk over RQ1?
- **RQ4: Architecture.** Encode the page once and answer many questions as
  cheap suffixes vs re-encode per question; options in-context (cross) vs
  options as cached vectors (dual). What is the accuracy/cost trade-off?
- **RQ5: Pixels vs text.** On questions whose answer is visual (checkboxes,
  signatures, stamps, layout), how far behind is an OCR→text pipeline?

## 4. Principles

1. **Evaluate before building.** No model work until the benchmark and metrics
   exist and a zero-shot baseline is measured.
2. **Closed answer sets, argmax decode, never sample.** Temperature is then a
   pure calibration knob that cannot change the answer.
3. **Calibrate `max(p)`**, the probability of the returned answer. Margin or
   entropy "confidences" may be displayed but are never what we calibrate or
   gate on.
4. **Honest evals.** Test labels come from human-annotated public datasets, never
   from an LLM. Held-out domains are held out at the *dataset* level, not just
   the document level.
5. **Poor-researcher budget.** Compute, in order of preference: Apple Silicon
   with **MLX** (local), free **Kaggle** GPUs (2×T4), and rented **RunPod** NVIDIA
   GPUs only when a run needs them. Unsloth covers both MLX and CUDA. That means
   backbones of ≤ ~4B parameters, 4-bit or bf16, and LoRA or head-only training.
6. **Backend-agnostic core.** Model runners (MLX or PyTorch) only emit per-question
   logits as NumPy arrays. Data, metrics and calibration are pure NumPy, so
   results are comparable across backends.

## 5. Question types

| Type | Output | Notes |
|------|--------|-------|
| `noul` (yes/no) | P(true) | Verification questions go here ("Does the page state total = 1,234.00?"). |
| `choice` | distribution over supplied options | Options supplied at request time. |
| `score` | distribution over ordered levels | Report argmax + probability, never a bare expected value. Train with an order-aware proper score (RPS). |

Evidence output (which page or region) is **parked**: we revisit it after RQ1–RQ3.

## 6. Benchmark v0 (DocJev-Bench)

Recast public, human-labelled datasets into typed questions. Licences to be
verified before anything is redistributed; we may ship *build scripts* rather
than data.

| Source (to verify) | Type | Construction |
|---|---|---|
| RVL-CDIP | choice (16) | Document type. |
| DocVQA | noul | "Is the answer to <q> '<a>'?" True answer vs. hard distractor (another answer-shaped string from the same page). |
| DocVQA | choice (4) | Gold answer + 3 same-page distractors. |
| FUNSD / CORD / SROIE | noul | Field verification: "Is the <field> '<value>'?" with true or perturbed value. |
| CORD | score | Ordinal bins derived from counts (e.g. line items: 1 / 2–3 / 4–6 / 7+). |
| held-out domain (e.g. InfographicVQA or ChartQA) | noul / choice | Never used for training or temperature fitting. |

Splits: `train`, `calib` (temperature fitting), `test`, and `ood` (held-out datasets).

## 7. Experiment ladder

| ID | What | Answers | Status |
|----|------|---------|--------|
| E0 | Metrics harness + tests | infra | done |
| E1 | DocJev-Bench v0 builders + small sample | infra | next |
| E2 | Zero-shot VLM LM-head readout (2–3 small VLMs) + OCR→text baseline | RQ1, RQ5 | |
| E3 | Temperature fit on in-domain, test on `ood`; label-efficiency curve | RQ2 | |
| E4 | Heads-only vs LoRA, proper-score loss, multi-task mixture | RQ3 | |
| E5 | Shared-prefix vs per-question; cross vs dual options | RQ4 | |

## 8. Open questions

- Which small VLMs to test in E2 (candidates to verify for MLX / T4 memory and quality:
  Qwen3-VL 2B/4B, Qwen2.5-VL-3B, SmolVLM2, Qwen3.5-0.8B).
- Multi-page documents: first page only in v0; revisit.
- Is evidence output in scope (see §5)?
