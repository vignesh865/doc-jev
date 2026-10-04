# DocJev — learning journal

What we learned along the way, explained in plain words. `JOURNAL.md` records
decisions and results; this file records understanding. Dated and append-only,
like the research journal.

---

## 1. "Confidence" is not always "probability of being right"

*2026-10-04. From our first Clef-flash API call.*

**The reply we got.** We asked which team should handle "Checkout has been
failing for every customer":

| option | probability |
|---|---|
| technical | 0.9601 |
| billing | 0.0327 |
| sales | 0.0072 |

The API also returned `"confidence": 0.8844`. Why isn't that 0.9601?

**Two different questions.** A model's output can answer two questions:

1. *How likely is the answer I picked to be right?* That is the top probability,
   **max(p) = 0.9601**. If the model is well calibrated, answers given with 0.96
   are right about 96% of the time.
2. *How spread out is the whole distribution?* That is about the shape of all
   the probabilities together, not about the chosen answer.

Many "confidence" scores answer question 2. Common ones:

| name | formula | here |
|---|---|---|
| top probability | max(p) | 0.9601 |
| margin | top − second | 0.9601 − 0.0327 = 0.9274 |
| margin vs. the rest (Jev's) | top − mean of the others | 0.9402 |
| entropy-based | 1 − H(p)/log K | 0.8303 |
| **concentration (Gini)** | (K·Σp² − 1)/(K − 1) | **0.8844 ← matches** |

**What Clef's hosted API uses.** The last row reproduces 0.8844 exactly, to
4 decimals. Σp² is the chance that two independent draws from the distribution
land on the same option. It is 1 when all the mass is on one option and 1/K
when it is spread evenly. The formula rescales that to run from 0 (even) to 1
(certain). It is a measure of how *peaked* the distribution is.
(One example so far; we will check it on every reply in the baseline.)

**Why the difference matters.** Imagine two answers, each with top probability
0.60:

- A: 0.60 / 0.40 / 0.00 → concentration = 0.28
- B: 0.60 / 0.20 / 0.20 → concentration = 0.16

The chosen answer is equally likely to be right in both: 60%. But the
"confidence" scores differ, because the *losing* options are arranged
differently. So if you set "auto-approve when confidence > 0.9", you are not
setting "auto-approve when the answer is ≥90% likely to be right". That
threshold has no direct error-rate meaning.

This is why our rule (entry 1 of the journal) is to **calibrate and set
thresholds on max(p)** and to ignore any vendor "confidence" field. max(p) is
the one number whose meaning is "probability this answer is right", so it is
the one we can check against reality with a reliability plot. (The open-source
Clef code on Hugging Face sets `confidence = max(p)`; only the hosted API
differs.)

**Small side lesson: rounding.** The API rounds every probability to 4
decimals. A confident wrong answer can come back as 0.0000, and log(0) is
−infinity, so NLL must clamp tiny values (our `metrics.py` uses 1e-12). The
rounding also means we cannot tell 0.00001 from 0.00004, which only matters at
the extreme tail.
