# DocJev — research journal

Dated and append-only: a later entry corrects an earlier one, it does not
rewrite it. This is the only record of decisions for the project.

**Decision key:** **[U]** user decided · **[C]** Claude decided, user informed ·
**[E]** forced by external evidence (measurement or a third party), not a
preference.

---

## 1. Starting point: what we agreed before building

*2026-09-27 → 2026-09-28. Discussion only; nothing run.*

**Where the repo stood.** A branch from an earlier session
(`claude/project-spec-review-xmfvo1`) added a draft research plan, prior-art
notes, a metrics module (`src/docjev/metrics.py`, 7 tests passing) and
`CLAUDE.md`. It was fast-forwarded into `main` [U]. The user asked for all work
to go straight to `main` for now [U].

The draft plan's decision log (D1–D7) was written by that session, not agreed
with the user. It is removed; this journal replaces it [U]. The rest of
`docs/research_plan.md` stays as an unagreed draft.

**Decisions.**

- **Goal [U].** A Jev-style model for documents: a page image plus typed
  questions in, a probability distribution per question out, with no text
  generation. The motivation is the user's day job in intelligent document
  processing with LLMs. It is a hobby project; if it works, publish.
- **Core problem [U].** Trustworthy confidence: probabilities you can set a
  threshold on. Speed and cost matter less.
- **Scope [U].** Documents first, as page images (IDP pipelines already turn
  pages into images). General images come later.
- **Question families [U].** All three are real IDP work: classification
  ("is this an invoice?"), extraction checks ("is the invoice number 4471?")
  and validation ("do the line items add up to the total?").
- **Way of working [U].** Build → test → analyse the failures. No guessing at
  failure modes before there is data.
- **Loop 0 [U, proposed by C].** The smallest build → test loop: one small
  open vision model (about 2B, on MLX on the Mac), a few hundred questions made
  from one human-labelled dataset, the probability of each option read directly
  from the model, no training. Reported: accuracy, a reliability plot, and
  coverage at ≤1% and ≤5% error. The model and dataset are not chosen yet.
- **No OCR → Jev baseline [U].** Its accuracy would just follow the quality of
  the OCR, so it tells us nothing about our question.
- **Structure [U].** Follow the conventions of the user's earlier project: this
  journal, and a chain of data → raw predictions → scored runs → tables and
  figures made by scripts, never by hand.

**Next.** Choose Loop 0's model and dataset.

---

## 2. Loop 0 planned: Qwen3-VL-2B on RVL-CDIP and DocVQA yes/no

*2026-09-28. Plan only; nothing downloaded or run.*

**Model [U].** `Qwen3-VL-2B-Instruct`, chosen because it is the newest small
Qwen vision model. It runs on the Mac through `mlx-vlm`, which supports
Qwen3-VL; an MLX conversion exists at `mlx-community/Qwen3-VL-2B-Instruct-4bit`.
The user has already trained a Qwen3 text model (1.7B) with Unsloth on MLX, so
the toolchain is familiar.

- *To check at build time [C]:* whether a bf16 MLX version exists and fits in
  memory. We read probabilities, and 4-bit weights may shift them, so bf16 is
  preferred.

**Findings on question types [E].**

1. **Classification is naturally a `choice` question, not `noul`.** RVL-CDIP
   asks which of 16 document types a page is: one distribution over 16 options.
   It can also be asked as 16 separate `noul` questions ("Is this an invoice?"),
   but choice is the ready-made form. Start with choice [U].
2. **DocVQA already contains a ready-made `noul` set.** About 6% of DocVQA
   questions are yes/no, with human answers; the dataset tags them with the
   reasoning type "yes/no". The validation split is about 5,300 questions, so
   **roughly 300 yes/no questions** (estimate; to be counted). No conversion is
   needed, so every label in the set comes from a person.
3. **The rest of DocVQA needs conversion.** Its answers are open text ("What is
   the invoice number?"). Using those questions means building `noul` ("Is the
   invoice number 4471?" with a wrong value) or `choice` (true answer plus
   same-page distractors). The true answer stays a human label; only the
   distractors are ours. We found no ready-made human-labelled choice or score
   version. **Conversion is deferred until after Loop 0 [U].**

**Loop 0 set [U].**

| source | type | size | conversion |
|---|---|---|---|
| RVL-CDIP | choice (16) | ~200 pages | none |
| DocVQA val, yes/no subset | noul | ~300 questions (to count) | none |

No training. The probability of each option is read directly from the model.
Reported as agreed in entry 1: accuracy, a reliability plot, and coverage at
≤1% and ≤5% error.

**Open for the build [C, to confirm].** How to phrase the prompt, how an option
maps to tokens (for example " Yes"/" No", or the 16 type names), dataset
licences and download sources, and how the ~200 RVL-CDIP pages are sampled.

**Sources.** [DocVQA question types](https://www.emergentmind.com/topics/docvqa-dataset) ·
[mlx-vlm](https://github.com/Blaizzy/mlx-vlm) ·
[mlx-community/Qwen3-VL-2B-Instruct-4bit](https://huggingface.co/mlx-community/Qwen3-VL-2B-Instruct-4bit)

---

## 3. Correction: DocVQA has almost no yes/no questions; RVL-CDIP labels are noisy

*2026-09-28. Metadata checks only; no images downloaded, no model run.
Corrects entry 2, finding 2.*

**DocVQA yes/no subset is ~37 questions, not ~300 [E].** Pulled the question,
answer and type fields for all 5,349 validation questions from
`lmms-lab-encoder/DocVQA` (Apache-2.0 on Hugging Face; the old name
`lmms-lab/DocVQA` redirects here). Question-type counts:

| type | questions |
|---|---|
| layout | 1,995 |
| table/list | 1,766 |
| form | 1,024 |
| free_text | 771 |
| handwritten | 323 |
| figure/diagram | 261 |
| others | 241 |
| Image/Photo | 98 |
| **Yes/No** | **28** |

A question can carry more than one type. Only **37** questions have "yes" or
"no" as the first answer (19 yes, 18 no), which is 0.7% of the split. The "about
6%" in entry 2 came from a secondary summary page and was wrong; entry 2
should not have repeated it without a count. Thirty-seven questions are too few
to measure coverage at ≤1% error, so **the DocVQA yes/no set cannot be Loop 0's
noul set as planned.** The label is `Yes/No` in this copy (case-sensitive),
not `yes/no`.

**RVL-CDIP test labels are noisy [E, third party].** Larson et al. (EACL
2023) estimate **8.1% label errors** in the RVL-CDIP test set: 1.6% for resume,
up to 16.9% for letter. They also report many ambiguous or multi-label pages
and a large overlap between the test and train splits. A later paper finds
spurious ID-code cues in RVL-CDIP and Tobacco3482. This matters for our
north-star metric: if about 8% of labels are wrong, a model that is always right
still shows about 8% "error", so coverage at ≤1% error cannot be measured
honestly on raw labels.

**Other facts gathered for the Loop 0 decisions [E].**

- `mlx-community/Qwen3-VL-2B-Instruct-bf16` exists (also 3/4/5/6/8-bit).
- RVL-CDIP on Hugging Face (`aharley/rvl_cdip`): licence "other", not gated,
  400k pages. There are small per-class samples from third parties, which we
  would not trust without checking.
- DUDE (41,541 human-annotated questions over 5,019 documents) includes yes/no
  answers; its count has not been checked.

**Status.** The Loop 0 set in entry 2 is on hold until the user decides where
the noul questions come from and how to handle RVL-CDIP's label noise.

**Sources.** [Larson et al., On Evaluation of Document Classification using RVL-CDIP](https://arxiv.org/abs/2306.12550) ·
[Spurious Cues in RVL-CDIP and Tobacco3482: ID Codes](https://dl.acm.org/doi/10.1145/3704268.3748683) ·
[DUDE](https://arxiv.org/abs/2305.08455) ·
counts from the `datasets-server` rows API over `lmms-lab-encoder/DocVQA`, validation split.

---

## 4. Loop 0 re-planned after entry 3

*2026-09-29. Plan only; nothing run. Decided in a structured interview; replaces
the Loop 0 set in entry 2.*

**Decisions.**

- **Noul questions come from RVL-CDIP [U].** The same pages are asked yes/no
  questions ("Is this a letter?"). There is no new dataset and no conversion,
  and we can check whether the model's choice and noul answers agree about the
  same page. DocVQA's 37 yes/no questions are dropped from Loop 0.
- **We hand-check the labels [U].** The user (IDP background) and Claude review
  the sampled pages and fix or drop wrong labels, answering entry 3's ~8% noise.
  The number of fixed and dropped labels is logged, and results are reported on
  both raw and checked labels.
- **Sample: balanced, 13 pages per class, 208 pages [U]**, drawn from the
  RVL-CDIP test split with a fixed random seed.
- **Try both ways of reading the choice answer [U].** On the same pages:
  1. *Letters:* options listed A–P; read the probability of the next token
     being each letter. One step per page.
  2. *Full words:* score the log-probability of each full type name. No letter
     bias, but longer names are penalised and it needs one check per option.

  Compare accuracy and calibration between the two. For noul both ways are the
  same: P("Yes") against P("No").

**Proposed by C, to confirm.** In the letters run, shuffle the option order per
page (fixed seed) so that a bias toward early letters shows up rather than
hiding.

---

## 5. New prior art: Cloudflare Clef, an open decision model with vision

*2026-10-03. Desk research only; nothing downloaded except model cards, configs
and code (no weights); nothing run. The user flagged the release [U].*

**What it is [E].** Released 2026-10-01 by Cloudflare's Workers AI team, Apache
2.0, weights on Hugging Face. Two models:

| | Clef | Clef-flash |
|---|---|---|
| backbone | Qwen3.8-27B with its vision encoder | Qwen3.5-9B with its vision encoder |
| total parameters (bf16) | 27.36B | 9.41B |
| joint head | width 1024, 2 routing + 4 decoder layers, ~256 MB | same width and layers |
| median / p95 latency (self-reported) | 209 / 239 ms | 39 / 122 ms |

A page image plus a state and a schema of typed questions (`noul`, `choice`,
`score`, the Jev API) in; one logit per allowed option per question out, in one
forward pass, with no generation. Jev/SystemOne compatible. Context 64k, up to
64 questions per request. Hosted on Workers AI as `@cf/cloudflare/clef` and
`clef-flash` (Clef: $0.24 per M input tokens; up to 4 images per request, PNG,
JPEG or WebP, base64 only, each ≤4 MiB and ≤16 MP).

**How it works, from `joint_schema_model.py` [E].**

- The prompt is: system prompt → `STATE:` → image tokens → state text →
  `SCHEMA FIELDS:` listing every question and its options as JSON
  (`{"option_id", "description"}`) → `JOINT SCHEMA DECISIONS:`. All questions sit
  in **one sequence**, so later questions can see earlier ones (joint decoding).
- `choice` options are **sorted alphabetically by ID** before encoding; `noul`
  is a two-option choice (`true`/`false`) with default descriptions.
- The head reads the backbone's last hidden states. Each option's span (mean of
  its hidden states) plus its lexical embedding plus the question vector
  becomes a query; 2 cross-attention "evidence routing" layers attend over the
  whole sequence; 4 decoder layers mix the questions; each option gets
  `prior (lexical cosine) + gate × (scaled cosine + MLP residual)`.
- Output is a **raw softmax per question. No temperature or calibration step
  ships in the code.** For `score`, the API reports the *expected* level as
  `score`, plus max(p) as `confidence`.
- Training (blog): rank-256 LoRA on the backbone (merged in the release) plus
  the head, label-smoothed cross-entropy plus a Brier loss, then RLCD (partial
  credit for adjacent ordinal levels, a reference penalty). Data: internal
  synthetic sets that permute field order, prompts and schemas.

**What is missing, which matters for us [E].**

1. **No image or document benchmark.** All 41 Decision Index results and the 4
   Typesafe workflows are text. The only image use cited is an internal domain
   classification example (2.2 s vs 4.7 s for gpt-oss-120b), with no accuracy
   figure. Nothing says whether the head ever saw images in training.
2. **No calibration numbers.** The Decision Index leaderboard
   (`clef-evals.workers-ai-mle.workers.dev`, 73 models) lists ECE and Brier for
   most models (Jev: ECE 0.074, Brier 0.356), but **both are null for Clef and
   Clef-flash**, whose scores are marked self-reported. The only
   probability-quality number is ForecastBench Brier (Clef 13.9, flash 10.6, Jev
   17.4).
3. **Running it locally is not easy yet.** Custom PyTorch code, tested only with
   torch 2.11 and transformers 5.10.2 on one H200. There is no MLX port.
   Clef-flash's weights alone are ~19 GB in bf16.
4. A third-party write-up (flaviocopes.com, one person's testing, not
   verified) reports hosted image requests taking 13–30 s and a practical limit
   of about 190 KB per image.

**Self-reported text results, for scale [E].** Clef has the top Decision Index
score (61.21) against Jev's 57.91; Clef-flash scores 57.07. Clef leads on
BANKING77 (94.2 macro-F1 vs Jev 79.7); Jev leads on reasoning (GPQA Diamond 78.3
vs 48.0, MMLU-Pro 82.7 vs 65.9). On Typesafe's invoice-processing workflow
(text): Clef 64.7 exact actions vs Jev 61.8.

**Status.** No decision yet. Loop 0 planning (entry 4) is paused until the user
decides how Clef changes the plan.

**Sources.** [Cloudflare blog](https://blog.cloudflare.com/clef-decision-models/) ·
[Workers AI changelog](https://developers.cloudflare.com/changelog/post/2026-10-01-clef-workers-ai/) ·
[Workers AI model page](https://developers.cloudflare.com/workers-ai/models/clef/) ·
[HF Cloudflare/clef](https://huggingface.co/Cloudflare/clef) ·
[HF Cloudflare/clef-flash](https://huggingface.co/Cloudflare/clef-flash) ·
[Decision Index leaderboard data](https://clef-evals.workers-ai-mle.workers.dev/data/leaderboard.json) ·
[flaviocopes deep dive](https://flaviocopes.com/clef/)

---

## 6. Clef-flash API reachable; which "confidence" to use is open

*2026-10-04. One text-only call (248 input tokens, under $0.0001).*

**API works [E].** After two token attempts (an account token without working
Workers AI access returned 401 on every Workers AI endpoint), a token made from
the Workers AI "Use REST API" page works. First call: 200 in 0.42 s.

**The hosted `confidence` field is not the top probability [E].** Reply:
technical 0.9601, billing 0.0327, sales 0.0072, `confidence` 0.8844. Of the
measures tried, only the normalised concentration (K·Σp² − 1)/(K − 1) gives
0.8844 (to 4 decimals). The open-source code on Hugging Face instead returns
confidence = max(p). Probabilities are rounded to 4 decimals. To be confirmed on
every reply in the baseline. Explained in LEARNING.md lesson 1.

**Open question [U].** Which confidence measure DocJev should calibrate and set
thresholds on (top probability, lead over the runner-up, lead over the average
loser, concentration, entropy) is **undecided**. The user had not seen the
alternatives before, and wants to discuss it later. Claude had wrongly
described "calibrate max(p)" as agreed; it came from the earlier session's
draft, not from us. CLAUDE.md now lists such draft assumptions as not agreed.
Until it is decided, the baseline saves the full probability vector for every
question, so any measure can be computed later without re-running.

---

## 7. First image calls: the question was wrong for RVL-CDIP; a blank option added

*2026-10-04. Five image calls to clef-flash in total (2 + 1 rejected + 2), all
choice questions, PNG pages. Under $0.001.*

**Large images are refused [E].** A 398 KB PNG (`p157`) returned HTTP 413:
"estimated 136,060 tokens exceeded the 65,536 context window". The real cost of
a page is about 1,050 tokens, so the API's pre-check estimates tokens from the
base64 length (≈ 4 characters per token). In practice, images over about
190 KB are refused. Plan: send pages as JPEG at one fixed quality (recorded
per call), chosen once all 208 pages are in.

**The question did not fit the data [U, spotted by the user].** First question:
"What type of document is this?" with the 16 RVL-CDIP names. Page `p107`
(label `file folder`) is a nearly empty scan of a folder tab, with only a tiny
sideways handwritten note. Nothing on it says "file folder", and there was no
way to answer "blank". RVL-CDIP labels mix what a document *is* (letter,
invoice), the *object* scanned (file folder) and *how it was written*
(handwritten). Clef-flash answered `handwritten` 0.38, with `file_folder` not in
its top 3.

**Changes.**

- Add a 17th option, `blank_or_unreadable` [U]. RVL-CDIP has no such label,
  so it is our addition. A page whose true label cannot be seen will be
  handled in the hand-check.
- Reword to "Which category best describes this scanned page?" [C], because
  the old wording assumed every page is a document. Options still have no
  descriptions [C, open].

**Re-test, same two-page idea [E].**

| page | true label | top answer | p(top) | p(true label) | p(blank) |
|---|---|---|---|---|---|
| p107 | file folder (nearly empty) | blank_or_unreadable | 0.758 | 0.010 | 0.758 |
| p003 | letter (clear, typed) | letter | 0.914 | 0.914 | 0.005 |

On the empty page the model now picks the blank option with 0.76; on the clear
letter it is right with 0.91. Caveat: the wording and the option changed
together, so this test cannot say which change did what.

---

## 8. E01 started: clef-flash zero-shot, first 5 pages

*2026-10-04. Experiment `E01` (`experiments/E01/config.json`): clef-flash, data
`rvlcdip-v0` with questions q3, original PNG pages, pages over 185 KB skipped.
15 calls, all succeeded; 14,399 input tokens (about $0.0013). Scores in
`experiments/E01/metrics.json` and `results/tables/`.*

**Set-up decisions.**

- **Experiments have ids and grow by pages [U].** Settings are frozen per id;
  `--pages N` takes the first N pages of a fixed order that deals one class
  at a time, so 5 → 20 → 208 are nested. Runs resume and only retry failures.
  Any change of setting means a new id.
- **Small batch first [U].** Every growth step is shown before the next one.
- **Image size: send what fits [U].** No lossless format brings every page
  under the API's limit (entry 7): optimised PNG leaves 22 of 106 tested pages
  over 150 KB, lossless WebP 16, and JPEG makes these speckled scans *larger*.
  E01 sends the original PNG and skips pages over 185 KB: **23 of 208 pages**,
  concentrated in scientific publication (7 of 13), handwritten, advertisement
  and news article (3 each). The per-type results for those types rest on
  fewer, simpler pages. If results are promising, the user will rent a GPU to
  run Clef-flash without the limit [U].
- **Yes/no wording asks about the page [U]:** "Is this scanned page a memo?"
  (question version q3).

**Results on 5 pages [E].** Too few for any conclusion; listed in full.

| page | true | choice answer (p) | p(true) | yes/no on true type | yes/no on wrong type |
|---|---|---|---|---|---|
| p067 | scientific report | scientific_report 0.905 ✓ | 0.905 | 0.942 ✓ | 0.013 ✓ |
| p196 | memo | memo 0.874 ✓ | 0.874 | 0.953 ✓ | 0.021 ✓ |
| p187 | resume | resume 0.896 ✓ | 0.896 | 0.928 ✓ | 0.009 ✓ |
| p021 | form | memo 0.381 ✗ | 0.229 | 0.329 ✗ | 0.021 ✓ |
| p160 | presentation | blank_or_unreadable 0.360 ✗ | 0.041 | 0.061 ✗ | 0.011 ✓ |

Looking at the two misses:

- `p021` is a **fax cover sheet**: a printed header plus Date/To/Fax/From
  fields. Labelled `form`; the model split between memo 0.38, form 0.23 and
  letter 0.22. The page plausibly fits all three, and the model was unsure.
- `p160` is a **presentation slide scanned sideways** (rotated 90°): a title
  "PEL's" and one line of text on a mostly white page. The model picked
  `blank_or_unreadable` 0.36. Rotation and sparse text may both play a part;
  5 pages cannot say.

The choice and yes/no answers agreed on every page (3 both right, 2 both
wrong). Latency p50 0.57 s. The hosted `confidence` matched normalised
concentration on all 5 choice replies (max difference 0.00007, which is
rounding), confirming entry 6.

---

## 9. E01 grown to 20 pages; confident misses are label ambiguity

*2026-10-04. `E01` extended from 5 to 20 pages (1 page skipped as too big:
p089, 273 KB). 45 new calls, all succeeded. Scores in
`experiments/E01/metrics.json` (latest) and `experiments/runs.jsonl` (history).*

**Direction noted [U].** These RVL-CDIP experiments ask about a document's
**metadata** (what kind of page it is). A later stage must ask questions about
the document's **content** (fields, values, checks), which is where most IDP
decisions sit.

**Results on 20 pages [E].** Still small; read as a first look, not a measurement.

| group | n | accuracy | NLL | Brier | ECE |
|---|---|---|---|---|---|
| choice (17 options) | 20 | 0.60 | 1.25 | 0.51 | 0.26 |
| yes/no, true type | 20 | 0.65 | 1.00 | 0.57 | 0.28 |
| yes/no, wrong type | 20 | 1.00 | 0.02 | 0.00 | 0.02 |

The choice and yes/no-on-true-type answers agree on 19 of 20 pages (12 both
right, 7 both wrong, 1 right only on yes/no). Saying "no" to a wrong type is
easy (20/20); recognising the true type is the hard part. Latency p50 0.55 s,
p95 0.94 s.

**The 8 choice misses, from looking at the pages [C].**

| page | true | model (p) | what the page shows |
|---|---|---|---|
| p016 | form | handwritten **0.815** | large hand-lettered "TAC CUSTOMER SERVICES" on a cover sheet; no form fields visible |
| p045 | handwritten | scientific_report **0.827** | a *handwritten* monthly research report ("Smoke Analyses") on a printed RJR report form; the yes/no "handwritten?" said 0.83 yes |
| p021 | form | memo 0.381 | fax cover sheet (memo/letter/form all plausible) |
| p160 | presentation | blank_or_unreadable 0.360 | sideways slide with a title and one line |
| p106 | file folder | blank_or_unreadable 0.505 | (folder tab; not yet viewed) |
| p140 | budget | form 0.530 | (not yet viewed) |
| p054 | advertisement | memo 0.204 | (not yet viewed) |
| p163 | presentation | specification 0.391 | (not yet viewed) |

The two **confident** misses (0.82–0.83) are not the model being blind: in both,
its answer is a fair description of the page, and the dataset's single label
picks a different side of an overlap (handwritten vs. its purpose; a
hand-lettered cover vs. "form"). This is the label problem from entry 3
(ambiguous and multi-label pages), seen first-hand. It supports doing the
hand-check before reading anything into the confident-error numbers.

**Correction (same day) [U].** `results/tables/*.md` and `results/figures/*.png`
are blog material, derived purely from `experiments/`. They are removed (with
their scripts; recoverable from git at `1136b5c`) and will be made only when
the user asks. What is tracked for every experiment: raw replies in
`outputs/<exp>.jsonl`, latest scores in `experiments/<exp>/metrics.json`, and
the scoring history in `experiments/runs.jsonl`. Entries 8–9 mention
`results/tables/`; that no longer exists.

---

## 10. E01 at 60 pages: a cautious model; most confident misses are label problems

*2026-10-04. `E01` grown from 20 to 60 sent pages (10 skipped as too big so
far). 120 new calls, all succeeded; 171,874 input tokens in total for E01
(≈ $0.015). Scores: `experiments/E01/metrics.json`.*

**Results on 60 pages [E].**

| group | n | accuracy | NLL | Brier | ECE |
|---|---|---|---|---|---|
| choice (17 options) | 60 | 0.567 | 1.52 | 0.60 | 0.19 |
| yes/no, true type | 60 | 0.600 | 1.22 | 0.68 | 0.31 |
| yes/no, random wrong type | 60 | 0.983 | 0.06 | 0.03 | 0.03 |

- Yes/no as a detector: 36 hits, 24 misses, 1 false alarm, 59 correct
  rejections → **precision 0.97, recall 0.60** (precision only against random,
  easy negatives; see LEARNING lesson 2).
- Choice and yes/no-on-true-type agree on 56 of 60 pages (33 both right,
  23 both wrong).
- Choice accuracy by the model's top probability:

  | top p | pages | accuracy |
  |---|---|---|
  | < 0.50 | 13 | 0.31 |
  | 0.50–0.70 | 8 | 0.12 |
  | 0.70–0.85 | 16 | 0.56 |
  | ≥ 0.85 | 23 | 0.87 |

  Higher confidence does mean more often right, but answers in 0.70–0.85 are
  right only about half the time. Coverage at ≤1% and ≤5% error is 0.08 for
  every confidence measure, on raw labels.
- Per type (choice, 3–5 pages each, so very rough): email and news article
  were all right; budget and file folder all wrong (budget → invoice ×2; file
  folder → blank ×2).

**Ten confident choice misses (top p ≥ 0.70).** Four were viewed this time,
two in entry 9:

| page | label | model (p) | what the page shows | verdict [C] |
|---|---|---|---|---|
| p000 | letter | memo 0.945 | headed "Interoffice Memorandum", To/From/Subject | **label wrong**: it is a memo |
| p069 | scientific report | blank_or_unreadable 0.849 | nearly blank scan, one number "1151" | **label not visible**: the model's answer is fair |
| p182 | resume | blank_or_unreadable 0.766 | a page with only the word "APPENDIX" | **label not visible** |
| p158 | presentation | budget 0.877 | sideways slide: "Boys & Girls Club Budget 2000-2001" | **ambiguous**: a slide about a budget |
| p016 | form | handwritten 0.815 | hand-lettered cover sheet (entry 9) | ambiguous / label doubtful |
| p045 | handwritten | scientific_report 0.827 | handwritten research report (entry 9) | ambiguous: both true |

Not yet viewed: p104 (file folder → blank 0.73), p142 (budget → invoice 0.79),
p150 (invoice → form 0.86), p169 (questionnaire → form 0.79).

**Reading [C].** In all 6 confident misses looked at, the model's answer is a
fair description of the page, and the RVL-CDIP label is wrong, not visible
on the page, or one of two true answers. On raw labels the model looks
overconfident; much of that may be the labels. **The hand-check (entry 4) is now
the most important next step**: without it, the calibration and
coverage-at-risk numbers measure label noise as much as the model.

---

## 11. Blind label review of E01's 26 choice misses (proposals, not yet decided)

*2026-10-04. A Claude subagent reviewed each missed page from the image and its
dataset label only, **blind to the model's answer** [U asked for a subagent;
blind design C]. Output:
`data/rvlcdip-v0/label_review/claude_blind_review_E01_misses.jsonl` (per page:
description, every fitting category, best label, verdict, confidence). These
are an LLM's proposals; the user makes the final call, so test labels stay
human-decided.*

**Verdicts on the dataset label [C, proposed].** Correct 15, acceptable
(ambiguous) 3, not visible on the page 4, wrong 4.

- Wrong: p000 letter → memo; p016 form → file folder; p142 budget → invoice;
  p150 invoice → form.
- Not visible: p069 (cover showing "1151"), p133 (bare title page), p144
  (handwritten cover sheet), p182 ("APPENDIX" only).
- Acceptable: p054, p072, p104 (p104 low confidence).

**Model vs. the blind reviewer [E].** On the 26 misses, the model's answer
equals the reviewer's best label on 7, and is among the reviewer's fitting
categories on 15.

**The key pattern.** All 10 *confident* misses (top p ≥ 0.70) have an answer
the reviewer independently lists as fitting the page. The misses that are
real errors (dataset label correct, model answer not fitting) are all
low-confidence: p160 0.36, p054 0.20, p106 0.50, p140 0.53, p163 0.39, p013
0.28, p094 0.30, p101 0.42, p114 0.40, p133 0.25, p203 0.56. If the review
holds, the model's high confidence is earned and its real errors come with
low confidence: the behaviour the project wants.

**What accuracy would become (provisional, 60 pages).** Raw labels: 34/60 =
0.57. Using the reviewer's best label for the 26 misses: 41/60 = 0.68.
Counting any fitting category as right: 49/60 = 0.82. The 34 pages the model
got right were not reviewed.

**Open [U, pending].** (1) Accept, change or reject each proposed label. (2) How
to score pages where more than one category is true: strict single label, or
any fitting label.
