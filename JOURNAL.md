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
