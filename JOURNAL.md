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

---

## 12. E02: Clef 27B on the same 60 pages

*2026-10-05. `E02` (`experiments/E02/config.json`): the same as E01 except
model `clef` (27B). Same page order and the same 10 skipped pages, so the 60
pages are identical and paired. First 5 pages shown to the user, then the
other 55 [U]. 180 calls, all succeeded; 171,874 input tokens (≈ $0.041).
p50 latency 0.71 s (Flash 0.54 s).*

**Motivation [U].** Clef-flash is weak on the Decision Index's text benchmark
with a "none fits" option (CLINC150+OOS 0.67 vs Clef 0.97); E01's trouble
pages looked similar (near-blank or bare pages).

**Results, raw labels [E].**

| group | E01 Flash | E02 27B |
|---|---|---|
| choice accuracy | 0.567 | **0.633** |
| choice NLL / Brier / ECE | 1.52 / 0.60 / 0.19 | 1.38 / 0.54 / 0.20 |
| yes/no true type (recall) | 0.600 | 0.633 |
| yes/no wrong type | 0.983 | 0.983 |
| yes/no precision | 0.97 | 0.97 |
| choice AURC (top p) | 0.208 | 0.188 |
| choice coverage at ≤5% error (top p, oracle) | 0.08 | 0.18 |

**Paired, choice [E].** Both right 32, only 27B right 6, only Flash right 2,
both wrong 20. 6 vs 2 is not statistically significant at this size (exact
two-sided sign test p ≈ 0.29).

- The 6 pages only 27B got right (p013, p021, p094, p101, p106, p203) are all
  pages where the blind review called the dataset label correct and Flash had
  been wrong at low confidence (0.28–0.56). 27B fixes them at moderate
  confidence (0.42–0.69).
- Flash only right: p049 (handwritten; 27B said letter 0.56), p145 (invoice;
  27B said memo 0.59).

**With the blind review's labels (entry 11, still proposals) [E].**
Reviewer's best label: Flash 41/60, 27B 44/60. Any fitting category: Flash
49/60, 27B 52/60. (Two pages 27B missed but Flash got right were never
reviewed; they use the raw label.)

**Confident misses (top p ≥ 0.70): 9 for 27B.** For 8 of them, the blind
reviewer independently listed the model's answer as fitting the page. The
exception is **p112** (file folder → blank_or_unreadable 0.71), where the
reviewer says the folder is visible: the one confident real error so far.

**On the motivation.** 27B also answers `blank_or_unreadable` on the
near-empty pages (p069, p104, p182), and the reviewer agrees with that. So
the "none fits" behaviour is similar in both models. 27B's gain comes from
pages with real content that Flash misread at low confidence.

**Hosted `confidence` = normalised concentration** held for 27B too (max
difference 0.00014, rounding).

---

## 13. Content benchmark data set: `cord-v0` (CORD-v2 receipts), built, not yet run

*2026-10-07. Data only; no API calls.*

**Direction [U].** E01/E02 compared Clef-flash and Clef 27B on questions *about
the document* (what kind of page). Next, the same head-to-head on questions
*about its content*. CORD first [U]; SROIE, FUNSD and DocILE (invoices, needs
a token, validation labels public) are later options.

**Source [E].** `naver-clova-ix/cord-v2`, test split (100 receipts),
CC-BY-4.0, revision `7f0115a4…`, the official parquet (kept in `data/_raw/`,
git-ignored). Labels are human: every item's name, quantity and price, plus
subtotal, tax, service, total, cash, change. Shop names and private details
are already blurred in the images. 95 receipts have a labelled total; 5 are
skipped.

**Images: one uniform transform [C].** The originals are full-resolution PNG
photos (median 1.5 MB; **99 of 100 over the API's ~185 KB limit**), so "send
what fits" (E01's rule) would leave one receipt. Unlike the speckled RVL-CDIP
scans, these are photos, which JPEG compresses well. Measured: JPEG q85 at
native size still leaves 20 over; **longest side ≤ 1024 px at JPEG q85 fits all
(median 73 KB, max 170 KB)**. The most shrunk receipts (2304×4096 → 576×1024)
were checked by eye: totals and line amounts are still readable. The view
the model gets is recorded in the manifest (original and sent size per receipt).

**Questions, version c1 [U approved the design; wording C].** All answers come
from CORD labels:

| kind | question | answer source | count |
|---|---|---|---|
| choice | "Which of these amounts is the total on this receipt?" | options = total + up to 3 other distinct amounts *labelled on the same receipt*; ids a–d, shuffled | 87 (61 with 4 options, 23 with 3, 3 with 2) |
| noul_true | "Is the total amount on this receipt 31.000?" | the labelled total | 95 |
| noul_false | the same with **one digit changed** ("31.000" → "31.050"), same format | constructed, so certainly false | 95 |

- 8 receipts have no other labelled amount differing from the total, so they
  get no choice question.
- Amounts that equal the total in value (e.g. cash = total) are never used as
  wrong options, so no wrong option is accidentally right.
- Amount parsing treats '.' and ',' as thousands separators unless followed by
  exactly two final digits (Indonesian receipts).

**Pipeline.** `prepare_cord.py` builds `data/cord-v0/` (manifest, items,
pages/*.jpg, git-ignored). `run_clef.py` now reads each page's image path and
type from the manifest, so it handles JPEG and the existing PNG sets alike
(E01/E02 dry runs unchanged: 0 calls). Planned: **E03 = Clef-flash, E04 =
Clef 27B** on cord-v0, small batch first.

---

## 14. E03: Clef-flash reads receipt totals almost perfectly

*2026-10-09. `E03`: Clef-flash, data `cord-v0` (questions c1), all 95 receipts,
5 first and then the rest [U]. 277 calls, all succeeded; 212,035 input tokens
(≈ $0.019). p50 latency 0.37 s.*

**Results [E].**

| group | n | accuracy | NLL | Brier | ECE |
|---|---|---|---|---|---|
| choice ("which amount is the total?", 2–4 options) | 87 | 0.989 | 0.059 | 0.019 | 0.037 |
| yes/no, true total | 95 | 1.000 | 0.049 | 0.010 | 0.045 |
| yes/no, one digit changed | 95 | 0.989 | 0.046 | 0.020 | 0.031 |

Yes/no as a detector: precision 0.99, recall 1.00 (95 hits, 0 misses, 1 false
alarm, 94 correct rejections). These negatives are *hard* (one digit
changed), unlike E01's random wrong types. Choice and yes/no agree on 86 of 87
receipts. Coverage at ≤1% error (oracle): choice 0.98, yes/no 1.00.

**The two errors, viewed [C].**

- **c085, yes/no:** "Is the total 197.050?" (true total 197.450) got P(yes)
  0.78. The digits are clear in the sent image ("197.450" in large print), so
  this is a **real reading error**, and a fairly confident one. One digit in
  the middle was changed (4 → 0).
- **c095, choice:** picked 56,181 (0.84) over the labelled 61,799 (0.14). The
  receipt prints **"TOTAL 56,181"** (before tax) and **"GRAND TOTAL 61,799"**.
  The model chose the amount printed next to the word "TOTAL"; the label means
  the grand total. **The question is ambiguous**, not the model blind. A later
  question version could say "the final amount to pay".

**Reading [C].** On content (finding and reading one labelled amount), Clef-flash
is far stronger than on page-type classification (E01: 0.57 raw), and its
probabilities are well calibrated on this set (ECE 0.03–0.05). The task is
also cleaner: one unambiguous human label per question, so label noise does
not cloud the numbers the way it did on RVL-CDIP.

**Note on image size [E].** Token use varies with the original photo: c091
(228×336 px, the smallest) used 231–261 tokens, against ~830 for 1024 px
receipts. It was also the least confident receipt (0.79 on the true total,
0.23 on the changed one), and still right.

---

## 15. E04: Clef 27B on receipts; better at choosing, worse at catching a changed digit

*2026-10-09. `E04`: Clef 27B, the same as E03 except model; all 95 receipts
straight through [U]. 277 calls, all succeeded; 212,035 input tokens
(≈ $0.051). p50 latency 0.53 s (Flash 0.37 s).*

**Results, paired with E03 [E].**

| group | E03 Flash acc | E04 27B acc | Flash NLL / ECE | 27B NLL / ECE |
|---|---|---|---|---|
| choice ("which amount is the total?") | 0.989 | **1.000** | 0.059 / 0.037 | 0.014 / 0.014 |
| yes/no, true total | 1.000 | 1.000 | 0.049 / 0.045 | 0.013 / 0.013 |
| yes/no, one digit changed | **0.989** | 0.916 | 0.046 / 0.031 | 0.219 / 0.045 |

- **27B fixes c095** (the TOTAL vs GRAND TOTAL ambiguity) and is sharper
  wherever it is right (NLL 0.013–0.014).
- **27B says "yes" to 8 changed totals; Flash to 1.** Both miss c085. The other
  7 are 27B-only: c005 31.100 vs 31.000 (0.84), c007 111,090 vs 111,000
  (0.57), c013 51.200 vs 51.300 (0.76), c014 281,445 vs 281,435 (0.62), c044
  30.900 vs 30.000 (**0.95**), c097 55.800 vs 55.000 (**0.97**), c098 250,670
  vs 250,690 (0.83).
- **Labels checked by eye** for c044 ("TOTAL 30.000") and c097 ("TOTAL
  55.000"): both are clearly printed. These are real model errors, some very
  confident.

**The errors depend on how small the change is [E].** False yeses by the
relative size of the changed total:

| change vs true total | questions | Flash wrongly yes | 27B wrongly yes |
|---|---|---|---|
| < 1% | 55 | 1 | 6 |
| 1–3% | 5 | 0 | 1 |
| 3–10% | 10 | 0 | 1 |
| ≥ 10% | 25 | 0 | 0 |

**Reading [C].** 27B behaves as if it checks the value *approximately*: a total
that is close to the printed one gets accepted, sometimes with high
confidence. Flash behaves more like a digit-by-digit check. For IDP this
matters: catching an OCR-style one-digit slip is exactly the job of a
verifier, and there **the smaller model is the safer one** on this set. 8 of
95 is a small count; the size-of-change pattern is the more telling part.

---

## 16. Harder content data set: `docvqa-v0` (DocVQA validation), built, not yet run

*2026-10-09. Data only; no API calls.*

**Why [U].** After receipt totals (E03/E04, ~99%), harder content from a
different dataset. DocVQA was chosen [U] over DocILE (invoices; needs a token,
later). Real business pages (letters, forms, tables, charts, handwriting),
human questions and answers, and each question tagged with the skill it needs.

**Source [E].** `lmms-lab-encoder/DocVQA`, validation split, revision
`539088ef…` (Apache-2.0 tag on HF; DocVQA's own terms apply). 5,349
questions on 1,286 pages; 6 parquet files (1.06 GB) in `data/_raw/docvqa/`,
git-ignored.

**Turning open answers into typed questions [C, design agreed with U].** For one
question per page, the wrong options are the human answers to *other*
questions about the *same page*, so every option is a real string from that
page:

- **choice:** the DocVQA question itself; options = true answer + 2–3
  same-page answers.
- **noul_true:** `Question about this page: "<q>" Is the answer "<true>"?`
- **noul_false:** the same, with a same-page answer.

Rules for wrong options: the same *kind* as the true answer (number / date /
code with digits / words), and for words a similar length (1, 2–3 or 4+
words). The length rule was added after the first build showed giveaways
like "g" next to "nutrient content of the supplements per cup". A wrong option
is never a near-duplicate of any accepted answer (normalised equality,
containment, or similarity > 0.8): one DocVQA chart page has "…mortality rate"
and "…mortality rate canada" as answers to different questions.

**Sample.** 20 questions for each of 6 skills (table/list, layout, form,
free_text, handwritten, figure/diagram), each from a different page; questions
need at least 2 same-kind wrong options. 120 pages, 360 questions; choices
have 3 options (63) or 4 (57). Answer kinds: number 44, words 63, code 11,
date 2.

**Images [C].** Pages are large greyscale scans (median longest side ~2,200
px, ~490 KB). Measured on 80 pages: PNG at 1024 px leaves 48/80 over the API
limit; JPEG q75 at 1024 px leaves 2/80; 1280 px q65 leaves 6/80 (sharper). A
dense table page checked by eye at 1280 px is readable; **chosen: greyscale,
longest side ≤ 1024 px, JPEG q75**, the same 1024 px as cord-v0. 6 sampled
pages were still over 180 KB and were replaced by another page of the same
skill (listed in the manifest). Small table text at 1024 px is the main risk
to readability.

**Planned.** E05 = Clef-flash, E06 = Clef 27B on docvqa-v0; 360 calls each.

---

## 17. E05: Clef-flash on DocVQA, ~98–99%; one construction bug found and excluded

*2026-10-09. `E05`: Clef-flash, `docvqa-v0` (questions d1), 120 pages; 5
first and then the rest [U]. 360 replies (one call retried, see below);
336,999 input tokens (≈ $0.030). p50 latency 0.55 s.*

**A second, smaller token limit [E].** One call (d6134 choice, 154 KB image)
returned 413: "estimated 52,939 tokens exceeded this model context window
limit (**24,576**)". The two yes/no calls on the same image succeeded, and
the retry succeeded. So the hosted service sometimes routes to an instance
with a smaller window; with the base64 estimate (entry 7), images over ~70 KB
can fail there. Resuming re-sends only failures, so it costs one retry.

**Construction bug [E], excluded via errata.** On d2389 the wrong option
"June 2, 1997" is the same date as the true answer "6-2-97": the
near-duplicate check compared text, not dates. Both affected items (choice,
noul_false) are listed in `data/docvqa-v0/errata.json` and excluded by
`evaluate.py` (reported under `excluded_errata`). A scan of all 360 items
found no other date clash. The data set stays frozen. `prepare_docvqa.py` now
compares dates by meaning for any future version.

**Results, errata excluded [E].**

| group | n | accuracy | NLL | Brier | ECE |
|---|---|---|---|---|---|
| choice (3–4 same-page options) | 119 | 0.992 | 0.055 | — | 0.032 |
| yes/no, true answer | 120 | 0.975 | 0.113 | — | 0.041 |
| yes/no, same-page wrong answer | 119 | 0.992 | 0.055 | — | 0.022 |

(Brier is in `experiments/E05/metrics.json`.) Every skill scored 0.90–1.00 on
every question kind. 118 of 120 choice answers had top p ≥ 0.85, and 99% of
those were right.

**The remaining 5 errors, all viewed [C].**

| page | skill | what happened | verdict |
|---|---|---|---|
| d763 | figure | said *no* (0.23) to "first step = 'Prioritization Process Steps'" | **label wrong**: that is the chart's title banner, not a step |
| d1911 ×2 | handwritten | "To whom is Terry Whitson delivering?" → picked Ronnie Hurd (0.87), label Phipps Bend | **question ambiguous**: the form lists Point of Delivery = Phipps Bend (a place), Grower = Ronnie Hurd (a person) |
| d12658 | layout | said yes (0.94) to "Other Topics" as what follows "Case Report Forms"; label Risk Analysis | **real error**: "Risk Analysis" is the next row; "Other Topics" is the next heading |
| d2012 | figure | said *no* (0.07) to "strongest brand = Camel Lights"; the choice for the same page was right | **real error**: Camel Lights sits on the Strength side of the chart; the two question styles disagree |

**Reading [C].** On harder, real business pages, with wrong options that are
other true strings from the same page, Clef-flash stays at ~98–99% and well
calibrated. Of 7 raw errors, 2 were our bug, 3 are label or question problems,
and 2 are real model errors (a layout slip and a chart reading). The 1024 px
downscale did not visibly hurt the table questions (table/list 1.00 on all
three kinds).

---

## 18. Findings written up as a blog draft; E06 dropped

*2026-10-09.*

- **E06 dropped [U].** Clef 27B on DocVQA stopped at 230 of 360 replies when the
  account's **daily free Workers AI allocation (10,000 neurons)** ran out:
  the account is on the free plan, so every run so far cost nothing; the
  journal's dollar figures are list prices. The user ran 27B only for symmetry,
  and Flash is already at 95%+ on content, so E06 is not finished. Its partial
  output stays in `outputs/E06.jsonl`. `run_clef.py` now stops at once on the
  daily-cap error (code 4006) instead of retrying.
- **Blog [U].** For general tech readers, as Markdown in the repo, using
  27B results from E02 and E04 only: `docs/blog/clef-on-documents.md`. Every
  quoted number and all three figures come from `results/make_blog.py`
  (writes `docs/blog/numbers.json` and `docs/blog/figures/*.png`), never typed by
  hand. Figure style follows the dataviz reference palette and was restyled
  "modern" at the user's request [U].
- **Story of the post [C, outline agreed].** (1) Content works: ~99% on
  receipts and business pages, calibrated. (2) The 57% page-type score is
  mostly labels; every answer at ≥70% confidence matched or fitted the page.
  (3) The bigger model accepts near-miss totals (8/95 vs 1/95). Plus practical
  gotchas and a fine-print section.

---

## 19. Blog rebuilt to the user's playbook, with a "surprise" framing

*2026-10-09.*

- **Playbook [U].** The user's `BLOG_PLAYBOOK.md` (from the earlier Jev and Laya
  blog) now governs the post: exploration framing, the obvious doubt raised,
  one storyline, question headings with a plain-topic first line, a hook and a
  short-version box, no dashes, Plotly charts on white, every number a build
  placeholder, PNG exports of charts and of "tables in disguise", `tables.md`.
- **Story [U].** Most people know Jev and have not noticed that a *vision*
  decision model has shipped: "hey, a vision decision model is here", told
  with surprise.
- **Contamination doubt [U].** The user doubts the ~99% content scores: CORD,
  DocVQA and RVL-CDIP are public, so Clef may have seen them. The post raises
  this openly ("Is 99% too good to be true?") and says it is untested. **Next
  experiment, after the free daily quota resets:** the same questions with a
  blank image and with another document's image.
- **Build.** `docs/blog/template.html` → `build.py` → `clef-docs.html` and
  `tables.md`; `export_png.py` (headless Chrome) → `docs/blog/images/` (3
  charts; boxes: short version, why it fits, request/response, near-miss
  receipt, cost). The receipt box is filled from CORD's own label for c044. The
  matplotlib draft (`results/make_blog.py`, `docs/blog/figures/`,
  `clef-on-documents.md`) is removed in favour of this single source.
- Totals over the reported experiments (E01–E05): 1,274 answered questions,
  $0.16 at list price, $0 paid (free tier).

---

## 20. Blog restructured around the two umbrellas

*2026-10-09. Restructure asked for by the user [U].*

- The post is now organised by the project's two kinds of question:
  **about the document** (what kind of page: RVL-CDIP, E01 Flash and E02
  27B) and **about the content** (what the page says: CORD receipts, E03 Flash
  and E04 27B; DocVQA business pages, E05 Flash). A test map table places each
  experiment under its umbrella, and the parts are labelled "Part 1 · About
  the document" and "Part 2 · About the content".
- **"Is 99% too good to be true?" moved to the end**, just before "What's
  next?", so the contamination test reads as the natural next step [U].
- Added: an at-a-glance chart (labelled by umbrella) in the setup, and a chart
  of business-page accuracy by skill in Part 2. PNG exports now: 4 charts and
  6 boxes (short version, why it fits, test map, request/response, near-miss
  receipt, cost).

---

## 21. E06 finished: Clef 27B on DocVQA matches Flash

*2026-10-10. The free allocation had not reset at 00:00 UTC (still 429 at
01:22 UTC); it was available again by 18:48 UTC, about 26 hours after the
cap was hit (16:26 UTC on Oct 9). The reset rule is still unclear. The
remaining 130 calls were sent [U]; E06 is complete: 360 replies, errata
excluded as for E05. 27B latency p50 1.16 s, p95 4.3 s, max 23.9 s.*

**Paired with E05 [E].**

| group | E05 Flash | E06 27B | ECE Flash → 27B |
|---|---|---|---|
| choice | 0.992 | 0.992 | 0.032 → 0.011 |
| yes/no, true answer | 0.975 | 0.967 | 0.041 → 0.024 |
| yes/no, same-page wrong answer | 0.992 | 0.992 | 0.022 → 0.032 |

By skill (right out of n, Flash / 27B): tables 60/60, running text 60/60,
layout 59/60, forms 58/57 (of 58), handwriting 58/57, charts 58/58.

**Where they differ.** 27B fixes 3 of Flash's misses (d12658 layout slip,
d2012 chart, d763 title banner) and makes 5 of its own. Four of those are a
*no* to the true answer (d14791 "PM SERVICE" 0.16, d14795 "research" 0.20,
d9095 "1.38" 0.37, d10680 "1" 0.06); one is a *yes* to a wrong answer (d7887,
0.61). Both models pick "Ronnie Hurd" on the ambiguous d1911.

**Reading [C].** On DocVQA the two models are equal in accuracy, and 27B's
choice probabilities are better calibrated. The receipts' "close enough"
weakness (entry 15) does not show here: DocVQA's wrong options are different
strings, not near-miss numbers. 27B's errors here lean towards being too
*cautious* (saying no to the right answer).

---

## 22. Contamination test planned: Flash only, blank and swapped images

*2026-10-10. Plan only; no calls yet.*

**Question [U].** Are the ~99% content scores reading or memory? CORD and
DocVQA are public, so Clef may have seen them.

**Decisions.**

- **Clef-flash only [U].** 27B is not tested for contamination; one model is
  enough to make the point.
- **Same questions, different image [C, agreed].** Twins of E03 and E05 where
  only the image changes. `run_clef.py` has a frozen `--image-mode`:

  | exp | data | image sent | twin of |
  |---|---|---|---|
  | E07 | cord-v0 | blank (white, same size and format) | E03 |
  | E08 | cord-v0 | another receipt | E03 |
  | E09 | docvqa-v0 | blank | E05 |
  | E10 | docvqa-v0 | another business page | E05 |

  Swaps rotate the page order by half; no page keeps its own image, and a
  swap is skipped forward when the other page shares a true answer with this
  page. Each output line records `image_mode` and `image_sent`.
- **Reading the result [C].** A reading model should fall towards the
  "options-only" level on blank and swapped pages; a remembering model should
  stay high. The options alone can carry signal (on receipts the total is
  often the largest amount), so the **blank run measures that options-only
  level**, and the drop is judged against it, not against chance.
- **Variant, later [proposed by C, U to confirm].** A "not shown on this page"
  option on choice questions, run on real, blank and swapped images so the
  option has its own baseline.
- About 1,274 calls; blank images are tiny, so it should fit one day's free
  allowance. Small batch first.

---

## 23. Contamination test: Clef-flash is reading the page, not remembering it

*2026-10-10. E07–E10 (Flash; twins of E03 and E05 with only the image
changed), 5 pages each first, then all [U]. 1,274 calls, all succeeded.
Comparison by `contamination.py` → `experiments/contamination.json`;
each run is also scored in `experiments/E07…E10/metrics.json`.*

**Results [E]** (accuracy; mean probability given to the correct answer):

| data | question | real | blank | swapped | chance |
|---|---|---|---|---|---|
| receipts | choice | 0.99 (0.95) | **0.18** (0.24) | **0.25** (0.26) | 0.28 |
| receipts | yes/no, true total | 1.00 (0.95) | **0.00** (0.03) | **0.00** (0.03) | |
| receipts | yes/no, changed total | 0.99 | 1.00 | 1.00 | |
| business pages | choice | 0.99 (0.96) | **0.55** (0.49) | **0.44** (0.45) | 0.29 |
| business pages | yes/no, true answer | 0.97 (0.92) | **0.07** (0.16) | **0.02** (0.06) | |
| business pages | yes/no, wrong answer | 0.99 | 0.99 | 0.99 | |

(Blank images are the same size as the real ones, so token counts are
identical: the model sees a page of the right size with nothing on it.)

**Reading [C].**

1. **No sign of memory.** Asked "is the answer X?" with the true X, Clef-flash
   says yes 95–100% of the time with the real page and almost never without
   it (0–7%). A model that remembered these public pages would keep saying yes.
   This is the cleanest test, because the question gives nothing away.
2. **Receipts:** without the page, the choice falls to *below* chance
   (0.18–0.25 vs 0.28). The "the total is the largest amount" shortcut is not
   used; the 99% is reading.
3. **Business pages: the question and options alone answer about half the
   choice questions** (0.55 blank vs 0.29 chance). This is common sense, not
   memory: e.g. "smoking or non-smoking room?" with options NON-SMOKING /
   united / delta, or "corrected *dinner* time?" with one evening time.
   Without the page it is right on 39/63 word answers but only 19/44 numbers;
   by skill most often on layout (15/20) and forms (14/19), least on charts
   (5/20). This is a weakness of our *test design*: the same-page wrong options
   are often of a different kind of thing than the question asks for. The
   choice score on DocVQA overstates reading; the yes/no-true score does not.
4. Yes/no on a *wrong* answer stays at ~99% in every run: without the page the
   model says no to everything, so that row says nothing about reading.

**Consequence for the blog [C, to propose].** The "Is 99% too good to be
true?" section can now be answered: mostly no. The receipt and yes/no results
are reading; part of the DocVQA multiple-choice score comes from guessable
options.
