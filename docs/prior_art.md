# Prior art notes

Read from source code where possible (2026-09-27). These are inspirations, not
designs we must follow.

## Jev (TypeSafe AI, closed)
- API: `state` + dict of typed questions → one distribution per question, single pass.
- Types: `noul` (P(true)), `choice`, `score` (≤10 levels). Trained with "RLCD".
- **Text-only** (string / JSON / array); no image input. 64k context.
- Confidence shown to users = top prob − mean of the rest (a margin, not a probability).

## CLM (Contrastive-LM/CLM, Apache-2.0)
- Frozen Qwen3-8B, last-token pooling; two ~20M projection heads (state, action).
- Score = scale · cosine(state, option) → softmax. Bidirectional InfoNCE.
- Data: ~60M Q&A pairs → ~30M hard negatives → ~1M agent trajectories.
- The question is appended to the state (`context + "\n\n" + question`), so the
  state vector is question-conditioned. Options are embedded alone and cached.
- `noul` candidates: `"true: Yes. This is true: {q}"` / `"false: ..."`.
  `score` = plain choice over level texts; reports expected index; no ordinal loss.
- No fitted calibration (temperature default 1).

## Laya (NandhaKishorM/laya, Apache-2.0)
- ModernBERT-large, fully fine-tuned, cross-encoder input:
  `[CLS] <type> question [SEP] [MASK] opt0 [MASK] opt1 ... [SEP] state [SEP]`.
- Head: 2 transformer layers → score each `[MASK]`; plus an act/escalate head.
- Options share `head_max_len`=192 tokens → Banking77 (77 labels) 0.425 vs Jev 0.87.
- "RLCD" in the notebook: Gaussian noise on logits (group of G), advantage =
  group-normalised proper-score reward, **plus a weight-1.0 cross-entropy term**.
  Reward = log score + spherical score − RPS (RPS only for `score`, order-aware).
  Since the reward is differentiable, direct minimisation of the proper score is
  a simpler equivalent to try.
- Calibration: temperature per (type × option-count bucket {2, 3–5, 6–10, 11+}),
  clamped to [0.5, 5]. ECE 0.466 → 0.081.
- Code explicitly separates `max(p)` (calibrated) from entropy "confidence" (not).
- Base checkpoints are near random zero-shot on its typed-decision benchmark; fine-tuned reach 0.766.

## Visual Jev (arXiv 2609.25845) — closest prior art; read only via abstract/snippets
- Qwen3-VL-4B, vision tower frozen, LoRA on the language model.
- Encodes image + context once; questions run as batched suffixes; candidate
  probabilities read from the LM head.
- Macro accuracy 70.6 → 76.1 after answer-supervised post-training; 8.9× faster at 32 questions per image.
- Typed decision heads gave **no accuracy gain** over LM-head readout.

## Vision-JEV (arnodjiang/Vision-JEV, MIT)
- Qwen3.5-0.8B + rank-16 LoRA + candidate pointer head; extract/choice/boolean.
- Uncalibrated; no task-ready checkpoint or results released (v0.1 prototype).

## jev-visual (hr98w/jev-visual)
- Educational: Qwen3.5-0.8B (MLX, 4-bit), shared prefill, LM-head candidate
  log-probs, no training or calibration.
