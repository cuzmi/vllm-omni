# vLLM-Omni performance evidence

This branch stores results and provenance only. Experiment drivers, source copies, patches and model weights are excluded.

- [MAGI-2 TurboVAE output offload](magi-vae-output-overlap/20261008T145022Z/REPORT.md): H100 correctness, matched pinned-memory control, decoder-stage latency and raw traces. No full-model E2E claim.

- [FLUX.2 request-local RoPE reuse](flux2-rope-reuse/20261010/REPORT.md): real-model correctness and reuse verified on one H200; ABBA E2E differences below repeated-run drift, no stable speedup established. Includes the failed probe-startup attempt and final evidence.
