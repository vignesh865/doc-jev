# DocJev — context for Claude sessions

Research / hobby project, open source (Apache-2.0). Claude acts as assistant PI.
Build from first principles; Jev, Laya, CLM, Visual Jev are inspirations only
(notes in docs/prior_art.md). The original spec was advice, not a contract.

Read first: docs/research_plan.md (motivation, RQs, experiment ladder, decision log).
Update its decision log and experiment table whenever a decision or result lands.

Goal: small model, document image + typed questions (noul / choice / score) ->
calibrated distribution per question, single pass, no generation. Zero-shot first;
per-domain tuning is the fallback.

Rules:
- Argmax decode only; calibrate max(p); fit temperature by NLL, report ECE.
- North-star metric: coverage at <=1% / <=5% risk, always with accuracy, NLL, Brier, ECE.
- Evaluate before building: benchmark + zero-shot baseline before any training.
- Test labels come from human-annotated datasets only, never from LLM-generated labels.
- Budget: MLX locally (Apple Silicon), then Kaggle (2xT4), then RunPod; Unsloth. Backbones <= ~4B.
- Model runners emit per-question logits as NumPy; data/metrics/calibration stay pure NumPy.

Status: E0 (src/docjev/metrics.py + tests) done. Next: E1 benchmark v0 builders
(RVL-CDIP, DocVQA, FUNSD/CORD/SROIE, held-out OOD dataset), then E2 zero-shot
VLM LM-head readout on MLX.

Dev: `pip install -e '.[dev]' && pytest`
