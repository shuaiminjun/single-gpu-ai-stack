# gpu-orchestra

**单卡多模型服务层**：一个 20 GB 显卡上同时挂 LLM / TTS / 扩散模型，对外给一套 OpenAI 兼容的 `chat` + `audio` 接口，带显存预算调度、排队、LRU 淘汰和首包延迟指标。

零运行时依赖（纯标准库），`git clone` 完就能跑，不需要 GPU。

```bash
python -m orchestra check  --config examples/config.json   # 看一眼显存预算
python -m orchestra serve  --config examples/config.json --port 9881
```

```bash
curl -s localhost:9881/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"chat-7b","stream":true,"messages":[{"role":"user","content":"hi"}]}'

curl -s localhost:9881/v1/audio/speech -H 'Content-Type: application/json' \
  -d '{"model":"tts-voice-a","input":"独立寒秋，湘江北去"}' -o out.wav
```

---

## 这个仓库解决的真实问题

在一张 20 GB 的卡上，"把所有模型都加载进来"这件事**必然失败**。失败有两种表现：

1. 驱动直接 abort —— 在 ROCm 上就是 `LLVM ERROR: Can't get available size`，进程被杀，没有堆栈；
2. 不崩但慢 4 倍 —— 文本编码器/扩散模型/TTS 争抢显存，互相换入换出（实测 1024² 出图从 48 s 掉到 240 s/步）。

所以问题不是"怎么调用模型"，而是**"怎么让多个模型共用一张卡，还不让调用方感知到"**。这个仓库只做这一件事，并把它做成可测试的。

---

## 设计

```
           ┌──────────────────────────── HTTP ────────────────────────────┐
           │  GET  /healthz   /v1/models   /metrics                       │
           │  POST /v1/chat/completions   (JSON | SSE)                    │
           │  POST /v1/audio/speech       (chunked 16-bit PCM WAV)        │
           └──────────────────────────────┬───────────────────────────────┘
                                          │
                                   ┌──────▼──────┐
                                   │  Scheduler  │  ← 唯一持有 GPU 的地方
                                   │  budget /   │
                                   │  queue /LRU │
                                   └──────┬──────┘
                          ┌───────────────┼───────────────┐
                    ┌─────▼─────┐   ┌─────▼─────┐   ┌─────▼─────┐
                    │ chat-7b   │   │ tts-a     │   │ diffusion │
                    │ 8.0 GB    │   │ 3.5 GB    │   │ 12.0 GB   │
                    └───────────┘   └───────────┘   └───────────┘
```

四条硬规则：

| 规则 | 为什么 |
|---|---|
| **单飞（single-flight）** | 一张卡上的并发不会提高吞吐，只会让延迟不可预测。同一时刻只有一个 job 持有 GPU。 |
| **只淘汰空闲模型** | 正在服务的模型永远不卸载；LRU 顺序在 job 结束时刷新，热模型不会被踢。 |
| **准入控制** | 单个模型就超过整卡容量时**立刻拒绝**，而不是让调用方无限等待。 |
| **队列有上限** | 过载时返回 `429`，而不是堆到内存爆掉。 |

实现细节见 [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)。

---

## 实测指标

在一张 **AMD RX 7900 XT 20 GB（ROCm 7.2.1）** 上用真实模型跑出来的数字，不是估算：

| 指标 | 数值 | 说明 |
|---|---|---|
| TTS **首包延迟** | **0.68 s**（整包 3.9 s 的 17%） | 分块推流，不用等整句合成完 |
| 队列等待上报 | `X-Queue-Wait-Ms` 响应头 | 每次请求都能看到自己排了多久 |
| 显存监控 | `/metrics` 里 `orchestra_resident_gb` | 接 Prometheus 直接画 |
| 模型切换 | 淘汰 + 重载，调用方只感知到一次较慢的首包 | LRU |

> 首包 0.68 s 是在 32 kHz、单声道、CPU 编码链路下测的（GPU 合成 + 分块推送）。

---

## 与 OpenAI 接口的差异（诚实清单）

| 已兼容 | 未实现 |
|---|---|
| `/v1/models` 的 `data[].id/object/owned_by` | function calling / tool_calls |
| `/v1/chat/completions` 的 `stream` SSE 帧与 `[DONE]` | `logprobs`、`n>1`、`response_format` |
| `/v1/audio/speech` 的 `audio/wav` 分块输出 | `audio/mpeg`、`speed` 等参数 |
| `usage` 字段（fake backend 里是词数估算） | 精确 tokenizer 计数 |

**故意没做的**：多卡、分布式、模型热更新。加了会稀释这个仓库要证明的东西。

---

## 测试

```bash
pip install pytest
pytest -q          # 16 passed
```

覆盖的是**协议和调度**，不是模型质量：

- 调度：准入、LRU 淘汰、禁用淘汰时排队、单飞串行（用两个线程验证严格嵌套）、队列上限、等待时间统计
- 协议：SSE 增量到达（首帧早于整体完成）、WAV 分块能被 `wave` 模块解析、未知模型 404、缺字段 400、`/metrics` 计数正确

`FakeBackend` 让这些都能在**没有 GPU** 的 CI 上跑。

---

## 目录

```
orchestra/
  config.py      # 模型清单 + 显存预算（JSON/YAML）
  scheduler.py   # 核心：准入 / 排队 / LRU 淘汰 / 单飞
  backends.py    # Backend 协议 + FakeBackend + SubprocessBackend
  server.py      # 标准库 http.server：SSE / chunked WAV / 指标
  cli.py         # serve / check
examples/config.json
docs/ARCHITECTURE.md   # 调度算法与取舍
docs/CASE_STUDY.md     # 一个真实的排障案例：语音"闷/杂音"怎么被量化
tests/
```

---

## 路线图

- [ ] `SubprocessBackend` 接 llama.cpp / 一个 TTS CLI 的完整示例
- [ ] 按 `kind` 分池（chat 池 / audio 池各自排队）
- [ ] 请求级取消（客户端断开 → 立刻释放 GPU）
- [ ] 冷启动预热（`preload` 已有，缺一个开机自动跑的策略）

## License

MIT
