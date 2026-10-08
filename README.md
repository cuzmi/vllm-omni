# vLLM-Omni performance evidence

This branch stores results and provenance only. Experiment drivers, source copies, patches and model weights are excluded.

- [MAGI-2 TurboVAE output offload](magi-vae-output-overlap/20261008T145022Z/REPORT.md): H100 correctness, matched pinned-memory control, decoder-stage latency and raw traces. No full-model E2E claim.
- [MAGI-2 offline E2E, interrupted run](magi-vae-output-overlap/20261008T155553Z-e2e/REPORT.md): 4 H200, 100-step T2VA/I2VA output parity; complete 272p ABBA and incomplete 540p coverage. No demonstrated E2E speedup.
