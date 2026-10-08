# MAGI-2 TurboVAE: bounded asynchronous output offload

On one H100 80 GB, the candidate reduced the measured **VAE decode + CPU materialization** latency by 7.52% at 448x256 and 9.79% at 896x512 for 125 output frames. All measured outputs were bit-identical. A matched pinned-memory synchronous control was slower than the candidate by 3.58% and 2.67%, respectively. This is a decoder-stage result with fixed synthetic latents and real pretrained VAE weights, **not full MAGI request performance**.

## Source and environment

- Baseline: `a46e9aae7bb5065a27a69e34ba3f2a09af17a5ce`.
- Candidate: `b1224568a41a5774643ae7e0838ea849775d6983`, branch `perf/magi-vae-output-overlap`.
- Source delta: `vllm_omni/diffusion/models/magi2/turbo_vae.py` plus focused repository regression tests. The measured candidate file hash matches the committed file. Publication status is recorded separately; a local commit does not imply a push.
- Model: `sand-ai/MAGI-2-preview`, revision `2dea51b64db47ee5b4402d36fd90829a0c58913b`; only the TurboVAE JSON and 1.87 GB checkpoint were downloaded. See `manifest.json` for file SHA256 values.
- Reused Modal image: `vllm-omni-dev:nightly-20261008-a61aed496126`; derived only to ensure pytest is installed. Framework source was mounted from the local baseline archive, not taken from the image's newer Omni package.
- Hardware: one NVIDIA H100 80GB HBM3; 8 CPU threads; PyTorch 2.13.0+cu130; cuDNN 92000; CUDA 13.0. See `environment.json` and `PROVENANCE.json`.
- [Modal run](https://modal.com/apps/xinyuj2/main/ap-IzUAL3ZkSfYYhSbaJYYoUC).

## What changed

For non-distributed CUDA decoding with multiple chunks and output offload enabled, enqueue D2H on a side stream and defer the host wait. Retain each GPU source until its completion event. Keep at most two live pinned staging buffers and two pending copies. Retire completed chunks into ordinary CPU output storage, reuse staging, and drain the copy stream on exceptions before releasing buffers. Preserve the original chunk ordering, cropping and CPU merge. CPU, single-chunk, non-offload and distributed paths retain their original behavior.

The allocator may cache freed pinned blocks; the tested bound is on live staging buffers, not total reserved host allocator memory. The CPU staging-to-output copy and final merge are included in every candidate timing.

## Correctness and lifecycle

- 24 tests passed on H100: the existing TurboVAE tests plus FP32/FP16/BF16 offload parity, temporal padding/cropping, batch size 2, non-contiguous input, previous-output ownership, failure on a later decode, CPU fallback and a two-live-buffer bound.
- Real checkpoint BF16 parity: latent temporal lengths 1, 7, 8, 14 and 32, with finite and exactly equal results for A/B/C.
- All 72 timed outputs and warmups matched the baseline output hash, including both native output sizes. See `correctness.json`, `measurements.jsonl` and `tests.log`.
- Peak GPU allocated bytes were identical across A/B/C: 2,260,431,872 for 448x256 and 8,376,900,608 for 896x512. This does not establish equal reserved memory or behavior at other shapes.

## Unprofiled timing

A is the original synchronous pageable `.cpu()` path. B is the candidate. C has exactly the candidate's pinning, packing, staging pool and CPU merge, with one additional `ready.synchronize()` immediately after each copy event is recorded. Thus C prevents producer progress during each D2H without changing the data transformation. The ablation source hash and definition are in `ablation.json`.

Both source methods use the same loaded pretrained decoder and fixed GPU latent in one process on one GPU. Each method receives two warmups. Order is `ABCCBAABCCBA`, three measured calls per arm, twelve calls per method per shape. Host wall time includes `decode`, D2H, staging copies and CPU merge; explicit device synchronization brackets each sample. Hashing, garbage collection and trace collection are outside the timed region. cuDNN benchmark is disabled and deterministic mode enabled.

| Output size, 125 frames | A original, ms | C pinned synchronous, ms | B asynchronous, ms | B vs A latency | B vs C latency |
| --- | ---: | ---: | ---: | ---: | ---: |
| 448x256 | 360.432 | 345.683 | 333.313 | -7.52% | -3.58% |
| 896x512 | 1401.956 | 1299.368 | 1264.723 | -9.79% | -2.67% |

The four baseline arm means ranged 357.244–364.235 ms (1.94% of the mean) at 448x256 and 1400.238–1404.112 ms (0.28%) at 896x512. All four B arm means were below all C arm means in each workload. These ranges are within-run drift context, not confidence intervals or independent deployment repetitions. The earlier exploratory run is retained locally and is not pooled with this run.

## Trace mechanism

Trace timings are diagnostic and separate from the unprofiled latency results. Intersections use actual `gpu_memcpy` DtoH intervals and kernel intervals on the dominant compute stream; kernels on the copy stream are excluded from the compute overlap calculation. D2H counts include small scalar copies as well as the five large output copies. Raw Chrome traces and exact copy events are retained.

| Output | Method | D2H union, ms | D2H intersecting compute kernels, ms | Kernel count |
| --- | --- | ---: | ---: | ---: |
| 448x256 | A | 7.383 | 0.000 | 3340 |
| 448x256 | B | 3.417 | 2.238 | 3340 |
| 448x256 | C | 3.491 | 0.000 | 3340 |
| 896x512 | A | 139.636 | 0.000 | 3321 |
| 896x512 | B | 13.861 | 9.251 | 3321 |
| 896x512 | C | 13.846 | 0.000 | 3321 |

| Attribution | Observation | Confidence |
| --- | --- | --- |
| Decoder compute | Same total kernel counts; comparable compute busy times across A/B/C | High for these captures |
| Output transfer | A copies large output to pageable host memory; B/C copy to pinned host memory on a separate stream | High, actual GPU copy events |
| Deferred wait | B overlaps transfer with compute; C's immediate event wait removes that overlap | High for these captures |

| Overlap opportunity | Implemented boundary | Remaining exposure |
| --- | --- | --- |
| Chunk N output vs chunk N+1 decode | Copy stream waits for producer stream; CPU consumer waits for per-copy event | Last chunk tail, CPU staging copy and merge |

| Fusion pattern | Change |
| --- | --- |
| None | No decoder kernel replacement or fusion; five packing kernels move to the copy stream |

The overall benefit combines a faster pinned transfer path and delayed host waiting. B-vs-C wall-time differences must not be equated numerically with D2H/kernel intersection: delaying the host wait also changes CPU launch progress and CPU/GPU overlap. No kernel-only or bandwidth-universal speed claim is made.

## Limits

No full DiT generation, text/image encoding, video encoding, server scheduling, distributed VAE, CUDA Graph or non-CUDA platform benchmark was run. Latents are synthetic fixed-seed inputs, not DiT-generated samples. Only BF16 real-weight workloads at two resolutions were timed. There is no full-MAGI E2E claim.

The original probe reports `ModuleNotFoundError: No module named 'vllm._C'`; that probe uses an obsolete Python module name for this CUDA build. A subsequent H100 smoke test in the same named base image successfully loaded `vllm._C_stable_libtorch` and `vllm._moe_C_stable_libtorch`, and executed `torch.ops._C.rms_norm` with a passing numerical comparison (max absolute error 2.384185791015625e-07). The Python import name changed while the Torch operator namespace remains `_C`. Original logs are preserved. See `native_extension_gpu_smoke.json`, `native_extension_diagnosis.json` and `PROVENANCE.json`. This native-operator smoke does not establish full vLLM serving correctness.

`environment.json`'s `candidate_module` field resolves the `torch.inference_mode` wrapper; it is not the model source path. The source archive, replacement hashes, extraction location and exact commit mapping are recorded in `manifest.json` and `PROVENANCE.json`.

## Artifact policy

This results branch contains only reports, measurement JSON/JSONL, logs, compressed traces, configuration/provenance and checksums. No Python scripts, patches, source archives, model weights or PR-body drafts are included. Executable experiment files remain in the local RemoteFiles experiment directory. Source/script hashes in the runtime manifest identify external inputs and are not local-file links. `SHA256SUMS.json` verifies files actually shipped with the evidence.
