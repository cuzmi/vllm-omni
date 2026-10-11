# FLUX.2 请求内 RoPE 复用：正确性通过，未证实稳定的端到端收益

2026-10-10（America/New_York）。这是一次保留负结果的性能实验：请求内复用确实减少了 RoPE 准备次数，真实模型输出一致，但本轮端到端耗时差异低于同版本重复运行漂移。不能据此宣称稳定加速，也不能据此证明所有负载都没有收益。

## 对照和结果

使用真实 `black-forest-labs/FLUX.2-dev`，固定模型 revision `26afe3a78bb242c0a8bb181dcc8937bb16e5c66c`。单张 H200，BF16，eager，TP=SP=CFG parallel=1，并发 1；关闭量化、CPU offload、额外 diffusion cache 和计时阶段 profiler。

通过原生 `Omni.generate` 测量，包含文本编码、去噪、VAE 和返回图片的提取；不包含模型加载、预热、哈希、PNG 编码或写盘。两类负载为文生图，以及固定合成参考图的单图编辑 + CFG。每类两个固定提示词/seed，1024×1024，32 步。同一 GPU 容器内按 A→B→B→A 启动独立进程；每轮每场景一次预热、两次正式请求，共 8 次预热和 16 次计时请求。

| 场景 | A 中位耗时，秒 | B 中位耗时，秒 | 描述性耗时减少 | A 后轮相对前轮漂移 | B 后轮相对前轮漂移 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 文生图 | 15.695072 | 15.676275 | 0.1198% | -0.7202% | -0.9604% |
| 编辑 + CFG | 38.634802 | 38.619467 | 0.0397% | -0.2347% | -0.1619% |

每个版本、每场景仅四个正式观测；上表是描述性统计，不是置信区间或显著性结论。全部 16 个正式请求成功，对应提示词/seed 的 RGB 哈希跨四轮完全一致。原始逐请求记录保存在 `results/20261010T233514Z/measurements.jsonl`。

## 正确性与机制

正式计时前，512×512、4 步的两类负载在 A/B 中各重复两次，共 8 次真实模型请求。RoPE 输出、每一步 scheduler 后的 latent 和最终 RGB 均逐位一致，且中间值有限。

- 文生图每请求 `rope_prepare` 调用次数：A 4 次，B 1 次。
- 编辑 + CFG：A 8 次，B 2 次。
- 原生 pipeline / transformer 的实际导入路径及 dtype 已记录，确认使用源码覆盖层，而非镜像内另一版本。
- 512² 文生图诊断中，B 峰值 allocated 比 A 增加 1.5 MiB；编辑诊断主峰相同。这不代表其他尺寸的峰值或普遍内存上界。

诊断探针与正式计时分离。没有采集 GPU trace，因此本实验验证了调用次数变化，没有给出 RoPE kernel 耗时或重叠归因。

## 四格复盘

| 省掉什么成本 | 在什么条件下成立 | 付出什么代价 | 用什么对照验证 |
| --- | --- | --- | --- |
| 请求内重复生成 RoPE，以及编辑位置 ID 的重复拼接 | 请求内位置 ID 不变；本轮仅验证单卡 SP=1 | RoPE 张量存活到 pipeline 返回；增加临时存活内存和一个显式传参路径 | 真实中间值逐位比较、实际调用计数、同卡 ABBA、A/A 与 B/B 漂移 |

这次尝试证实了可删除的重复工作，但未建立值得优先推进的端到端收益。下一次应先确认目标操作占据的时间，再决定是否扩展实验；本次归档不触发额外 GPU 运行。

## 来源、失败记录与边界

- [基线源码](https://github.com/cuzmi/vllm-omni/tree/a46e9aae7bb5065a27a69e34ba3f2a09af17a5ce)。
- [实验源码与单元测试](https://github.com/cuzmi/vllm-omni/tree/4ed516bfc510528c0b5bd6e45e8fd2b3ce6b92f2)：运行时尚未提交；事后提交的两个生产文件哈希与执行快照一致。分支为 `perf/flux2-request-rope-reuse`。
- 镜像 digest：`vllm/vllm-omni@sha256:a61aed4961263c1ad4aa8108a78af151a0781eca98513db39354d3ecaee30f99`。PyTorch 2.13.0+cu130、vLLM 0.31.0、diffusers 0.40.0、transformers 5.14.1。镜像内 vLLM-Omni distribution metadata 是 `0.31.0rc2.dev48+gc548a110a`，实际运行源码是上述覆盖层。
- `20261010T233109Z` 首次 GPU 尝试因诊断探针在初始化 dummy run 中读取尚不存在的请求编号而失败。修复后跳过初始化诊断、在每次新进程前清理请求元数据。失败日志保留，不计入性能比较；它不是模型正确性失败。首次失败探针的源码哈希不同于最终脚本，归档不声称最终脚本逐字复现该失败。
- 成功运行：`20261010T233514Z`。累计记录 GPU 执行约 24.5 分钟（含失败尝试），按请求资源费率估算计算费用约 3 美元，不是实际账单；不含长期存储。完成后 Modal 容器列表为空。
- `cpu_evidence/results.json` 是更早的 CPU 源码提取/替身消费者验证；其中“未测 GPU”等字段描述当时状态。它不是原生项目 pytest 或真实 attention 结果，原生 pytest 当时受本地缺少 vLLM 阻塞。
- 未验证多卡 SP、NPU、编译模式、并发服务或图片质量数据集；固定小样本输出一致不能替代这些验证。

原始执行清单中的“未提交”状态和本地路径按历史事实保留；后续发布位置由 `PUBLICATION.json` 补充。脚本和测试位于 RemoteFiles，PR results 分支只保存证据与来源。

完整脚本、单元测试副本及复现说明：[RemoteFiles 固定归档](https://github.com/cuzmi/RemoteFiles/tree/5afc61aaffeeb164adf620fd009b37886e7bab10/modal_vllm_omni/flux2_rope_reuse)。
