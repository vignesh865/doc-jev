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
