# MAGI-2 offline E2E evidence — interrupted run

Completed full 100-step generation and MP4 encoding for both source versions. All completed matched T2VA outputs at 272p/540p and the I2VA pair at 272p have identical returned video-frame and audio-sample hashes. The **overall planned run did not complete**: 14 of 16 timed requests finished before a platform interruption. The 272p ABBA sequence is complete; the final 540p baseline pair and both auxiliary diagnostic requests are missing.

**No E2E speedup is demonstrated.** In the complete 272p sequence, TurboVAE decode wall time decreased 5.29%, while offline E2E mean latency increased 1.14%. These are observations from a small, single-allocation experiment, not a claim of general speedup or proven absence of regressions.

## Setup and measurement boundary

- A: [baseline a46e9aae](https://github.com/cuzmi/vllm-omni/tree/a46e9aae7bb5065a27a69e34ba3f2a09af17a5ce); B: [candidate b1224568](https://github.com/cuzmi/vllm-omni/tree/b1224568a41a5774643ae7e0838ea849775d6983). Production code changes only the TurboVAE offload path; regression tests are also in B.
- `sand-ai/MAGI-2-preview`, revision `2dea51b64db47ee5b4402d36fd90829a0c58913b`; 273.10 GiB of required components cached. No 1080p refiner.
- One Modal allocation: 4 H200, 32 requested CPU cores, 512 GiB requested RAM. These are tested resources, not a minimum-resource claim. DiT Ulysses SP=4, TP=1, CFG parallel=1, **VAE patch parallel=1** so the optimized non-distributed decoder path is exercised.
- Pinned image digest `sha256:a61aed4961263c1ad4aa8108a78af151a0781eca98513db39354d3ecaee30f99`. Installed packages: vLLM 0.31.0, Torch 2.13.0+cu130, Diffusers 0.40.0, Transformers 5.14.1. The installed vLLM-Omni metadata is 0.31.0rc2.dev48+gc548a110a; the **executed source** is the pinned A/B checkout above, selected through PYTHONPATH.
- One fixed prompt and seed 42; 100 denoising steps; 125 frames at 12.5 fps; 448×256 and 896×512; stereo 44.1 kHz audio. Exact input and environment settings are in [workload.json](workload.json).
- Fresh engine for each arm, default regional compilation, shared compilation cache, deterministic settings. Each arm/resolution has one 100-step warmup followed by two measured requests. Warmups, model load, hashes and file writes are excluded.
- Offline E2E runs from immediately before `Omni.generate` through normalization, encoding and audio/video muxing to MP4 bytes. It includes text encoding, denoising and decoding. It is **not HTTP serving latency or a concurrency benchmark**.
- Both trees receive the same lightweight `perf_counter` wrapper around `TurboVAE.decode`, with no added GPU synchronization. Decode wall time includes host work and any dependency waits; it is not GPU-kernel-only time. The extra synchronized stage profiler is disabled in the main measurements; its planned auxiliary runs were not reached.

## Complete 272p ABBA result

Means over four measured requests per implementation; lower is better:

| Metric | Baseline A | Candidate B | B versus A |
|---|---:|---:|---:|
| Offline E2E | 93.730606 s | 94.795207 s | **+1.136%** |
| Omni.generate | 92.826190 s | 93.897984 s | +1.155% |
| TurboVAE.decode | 334.852 ms | 317.134 ms | **−5.291%** |

Per-arm means expose repetition effects rather than hiding them:

| Arm | Measured requests | Offline E2E | TurboVAE.decode |
|---|---:|---:|---:|
| 0A | 2 | 93.852537 s | 335.623 ms |
| 1B | 2 | 94.008144 s | 325.677 ms |
| 2B | 2 | 95.582270 s | 308.591 ms |
| 3A | 2 | 93.608676 s | 334.081 ms |

A-to-A E2E mean drift is −0.260%; B-to-B is +1.674%. Individual E2E sample ranges are 0.577 s for A and 3.219 s for B. The slower B sample, **97.132532 s**, is retained. Its cause was not isolated. The observed 17.718 ms decode saving is only about 0.019% of baseline E2E; it does not account for the observed whole-request latency increase. No formal significance or general regression-free claim is made from these four samples per implementation.

## Incomplete 540p sequence

| Arm | Measured requests | Offline E2E mean | TurboVAE.decode mean |
|---|---:|---:|---:|
| 0A | 2 | 377.247447 s | 1189.236 ms |
| 1B | 2 | 377.497898 s | 1163.070 ms |
| 2B | 2 | 376.646435 s | 1163.322 ms |
| 3A | **0 — interrupted** | — | — |

The first completed A/B pair observed +0.066% E2E latency and −2.200% decode time. The missing final A pair prevents the planned ABBA comparison. Unbalanced aggregates remain visible in `summary.json` with an explicit incomplete/descriptive-only flag; they must not be used as an E2E speedup claim.

## Correctness and artifacts

- All completed matched T2VA requests at each resolution have identical returned uint8 RGB frame hashes and float32 audio hashes across A and B. The 100-step 272p I2VA A/B pair also matches exactly, using the same saved [reference image](reference.png).
- Each returned video is 125×H×W×3; audio is 441000×2 at 44100 Hz. This checks returned outputs, not every internal intermediate tensor.
- All **17 recovered MP4 files** match their recorded payload hashes and fully decode without errors using local FFmpeg. See [media_validation.json](media_validation.json). Before interruption, remote ffprobe also verified 11 MP4s, including every representative file retained here: 10-second H.264 video at 12.5 fps and stereo 44.1 kHz AAC audio.
- MP4 payloads are not always byte-identical even when the pre-encoding video/audio arrays match. The correctness comparison therefore uses the returned arrays, not container hashes.
- Six representative MP4s are retained here: one A/B pair for each T2VA resolution and for I2VA. All measurements and logs are retained. Experiment Python, source archives, model weights, and private billing records are excluded.

## Interruption and evidence limits

The app stopped at 2026-10-08 17:44:47 UTC. The runner recorded `KeyboardInterrupt`; the subsequent compute-based download returned `workspace is disabled`. The specific administrative/billing reason is unconfirmed. Direct Volume reads recovered the saved files. All related apps were subsequently observed stopped with zero tasks; no remaining work was moved to another workspace.

Completed records: **14 measured + 8 warmup + 2 I2VA correctness + 1 smoke = 25**. Missing: the final two 540p measured requests and the two auxiliary diagnostics. The original failed [status.json](status.json) is preserved alongside [interruption.json](interruption.json). An earlier optional topology probe failure occurred before model loading and produced no inference samples; it is recorded separately in [preflight_failure.json](preflight_failure.json).

This is a single prompt/seed and one GPU allocation. There is no completed 540p ABBA, HTTP load test, long-duration stress test, or new full-model GPU overlap trace. The [earlier H100 decoder-only evidence](../20261008T145022Z/REPORT.md) uses a different device/workload and must not be pooled with these H200 results or presented as full-model E2E speedup.

Raw evidence: [measurements](measurements.jsonl), [summary](summary.json), [decode timings](vae_decode_timings.json), [provenance](PROVENANCE.json), [environment/source manifest](manifest.json), and arm logs.
