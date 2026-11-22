[![ci](https://github.com/shuaiminjun/single-gpu-ai-stack/actions/workflows/ci.yml/badge.svg)](https://github.com/shuaiminjun/single-gpu-ai-stack/actions/workflows/ci.yml) [![python](https://img.shields.io/badge/python-3.10%2B-blue)]() [![license](https://img.shields.io/badge/license-MIT-green)](LICENSE)

# single-gpu-ai-stack

> **Status** — early-stage, actively developed. Feedback, corrections and "you got this wrong" are all welcome: **shuaiminjun541@gmail.com**
>
> Built with heavy AI assistance. The design decisions (and the mistakes that led to them) are written down in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) and [`docs/CASE_STUDY.md`](docs/CASE_STUDY.md) — read those before assuming the code is right.


**一张 20 GB 的显卡，对外一套 OpenAI 兼容接口，同时提供 chat / 图像 / 语音三种能力。**

核心只有一件事：**显存预算调度**——谁在用卡、能不能装下、装不下淘汰谁、装不下又没法淘汰时怎么拒绝。协议层（SSE、chunked WAV、b64 PNG）用标准库手写，零运行时依赖。

```bash
python -m core check --config examples/config.json    # 先看显存预算
python -m core serve --config examples/config.json --port 9881
```

```bash
# 1) 流式 chat
curl -N localhost:9881/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"chat-7b","stream":true,"messages":[{"role":"user","content":"hi"}]}'

# 2) 出图
curl -s localhost:9881/v1/images/generations -H 'Content-Type: application/json' \
  -d '{"model":"image-sdxl","prompt":"a red fox","size":"512x512"}' | jq -r '.data[0].b64_json' | base64 -d > out.png

# 3) 语音
curl -s localhost:9881/v1/audio/speech -H 'Content-Type: application/json' \
  -d '{"model":"tts-a","input":"独立寒秋，湘江北去"}' -o out.wav
```

---

## 它解决什么问题

在一张卡上堆 LLM + 扩散 + TTS，"全部加载"必然失败，而且失败得很脏：

1. **驱动直接 abort** —— ROCm 上是 `LLVM ERROR: Can't get available size`，进程被 SIGABRT，没有 Python 堆栈；
2. **不崩但慢 4 倍** —— 三个模型互相换入换出（实测 1024² 出图从 48 s 掉到 240 s/步）。

所以问题不是"怎么调模型 API"，而是"**怎么让几个不同框架的模型共用一张卡，而调用方无感**"。这个仓库只做这件事，并让它可测。

---

## 设计

```
        ┌────────────────────────────── HTTP (stdlib) ─────────────────────────────┐
        │ GET  /healthz  /v1/models  /metrics                                      │
        │ POST /v1/chat/completions   JSON | SSE                                   │
        │ POST /v1/images/generations OpenAI 形状（b64 或 data-url）                │
        │ POST /v1/audio/speech       WAV（默认正确长度 / stream 时分块）           │
        └───────────────────────────────────┬──────────────────────────────────────┘
                                            │
                                     ┌──────▼──────┐
                                     │  Scheduler  │  ← 唯一持有 GPU
                                     │ 预算/队列/LRU│
                                     └──────┬──────┘
                    ┌───────────────┬───────┴───────┬───────────────┐
              ┌─────▼─────┐   ┌─────▼─────┐   ┌─────▼─────┐   ┌─────▼─────┐
              │ chat-7b   │   │ chat-27b  │   │ image-xl  │   │ tts-a     │
              │ 8.0 GB    │   │ 15.5 GB   │   │ 11.0 GB   │   │ 3.5 GB    │
              └───────────┘   └───────────┘   └───────────┘   └───────────┘
```

四条硬规则：

| 规则 | 为什么 |
|---|---|
| **单飞（single-flight）** | 单卡并发不提高吞吐，只会让延迟不可预测。同一时刻只有一个 job 持有 GPU。 |
| **只淘汰空闲模型** | 正在服务的永不卸载；job 结束时刷新 LRU，热模型不会被踢。 |
| **准入控制** | 单个模型就超过整卡容量时**立刻拒绝**（`DoesNotFit`），不让调用方无限等待。 |
| **队列有上限** | 过载返回 `429`，而不是堆到内存里。 |

细节与取舍见 [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)。

---

## 实测（AMD RX 7900 XT 20 GB / ROCm 7.2.1）

| 指标 | 数值 | 说明 |
|---|---|---|
| chat 流式**首包** | **0.20 s**（整包 0.47 s） | 增量到达，不是缓冲后一次性返回 |
| 语音首包 | **0.20 s**（分块）/ 默认非流式给正确长度头 | 播放器直接能放 |
| 淘汰行为 | 驻留 8 + 3.5 GB，再来 12 GB → 淘汰最旧的，**不踢正在跑的** | `/healthz` 里能看到 `evictions` 计数 |
| 队列可观测 | `X-Queue-Wait-Ms`、`X-Gen-Ms` 响应头；`/metrics` 给 Prometheus | 排队/生成时间都可归因 |

> 上表是 **fake backend** 在本机跑出来的协议层延迟（不含模型前向）。接真模型的方法见 `integrations/`。

---

## 与 OpenAI 接口的差异（诚实清单）

| 已兼容 | 未实现 |
|---|---|
| `/v1/models`（`data[].id/object/owned_by`） | function calling / tool_calls |
| `/v1/chat/completions` 的 SSE 帧与 `[DONE]` | `logprobs`、`n>1`、`response_format=json` |
| `/v1/images/generations` 的 `b64_json` / `url`、`size` | `quality`、`style`、图生图 |
| `/v1/audio/speech` 的 `audio/wav`（含分块） | `audio/mpeg`、`speed`、音色列表 |
| `usage`（fake backend 里是词数估算） | 精确 tokenizer |

**故意没做**：多卡、分布式、模型热更新。加了会稀释这个仓库要证明的东西——单卡上的资源调度与协议正确性。

---

## 测试（20 passed）

```bash
pip install pytest && pytest -q
```

覆盖的是**协议与调度**，不是模型质量：

- 调度：准入、LRU 淘汰、禁用淘汰时排队、两线程验证严格单飞、队列上限、等待时间统计
- chat：SSE 首帧早于整体完成、`[DONE]`、404/429/504 分支
- image：返回的是能被 `struct` 解析的真 PNG（尺寸对得上）、`url` 形式、缺 prompt 400
- audio：WAV 能被标准库 `wave` 打开、分块模式头字段正确

`FakeBackend` 让以上全部在**没有 GPU 的 CI** 上跑。

---

## 目录

```
core/
  config.py      # 模型清单 + 显存预算（JSON/YAML）
  scheduler.py   # 准入 / 排队 / LRU 淘汰 / 单飞
  backends.py    # FakeBackend / ComfyUIBackend / SubprocessBackend
  pngio.py       # 无依赖 PNG 编码（仅供 fake backend 与测试）
  server.py      # 标准库 HTTP：SSE / chunked WAV / b64 PNG / 指标
  cli.py         # serve / check
integrations/    # 接真模型的最小示例（comfyui 工作流、tts 命令行）
docs/
  ARCHITECTURE.md   # 调度算法与取舍
  CASE_STUDY.md     # 真实排障：语音"闷/杂音"如何被量化
tests/
```

---

## 路线图

- [ ] `integrations/` 补一个能直接跑的 ComfyUI 工作流示例（含 `--cache-none` 说明）
- [ ] 请求取消：客户端断开 → 立刻释放 GPU
- [ ] 按 `kind` 分池（chat / image / audio 各自排队，避免一个长出图卡住对话）
- [ ] GitHub Actions：pytest + 三个接口的 smoke

## License

MIT
