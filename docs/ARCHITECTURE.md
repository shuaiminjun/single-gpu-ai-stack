# 架构与取舍

## 为什么用标准库写 HTTP

三个原因，按重要性排：

1. **可验证**：`pip install` 都不需要，评审者 `clone` 完直接 `python -m orchestra serve` 就能打接口。带一堆依赖的 demo 经常在别人机器上起不来。
2. **协议是重点**：SSE 帧、chunked WAV、响应头——"OpenAI 兼容"最容易翻车的地方就在这里，用标准库写反而把这些字节暴露得很清楚。
3. **反正是内部服务**：真正吃性能的是模型前向，不是 HTTP 层。真要上生产，把 `server.py` 换成 FastAPI 是半小时的事（`Handler` 不依赖框架）。

代价：没有自动 OpenAPI 文档、没有中间件生态、流式背压要自己管。都写在 README 的"未实现"里了。

---

## 调度算法

```
acquire(model):
    if model.vram > card.vram:            # 单个就装不下
        raise DoesNotFit                  # 立刻拒绝，不让它排队
    if waiting >= max_queue:
        raise QueueFull(429)
    waiting += 1
    while True:
        if not busy and fits(model):      # 空闲 + 装得下
            break
        if not busy and policy == "lru":
            victims = plan_eviction(model) # 纯函数，不改状态
            if victims is not None:
                evict(victims); break
        wait()                             # 或超时
    busy = True
    resident[model] = model.vram; resident.move_to_end(model)
    return Lease(wait_s)
```

三个细节值得说：

**① `plan_eviction` 是纯函数。** 它只返回"该淘汰谁"，不修改状态；修改在调用方持锁时统一做。这样淘汰策略可以单独测，也让"淘汰到一半失败"不可能发生。

**② LRU 在 job 结束时刷新。** `release()` 里会 `move_to_end(model)`，所以"刚用完的模型"是最不容易被淘汰的。否则一个长请求刚结束，它的模型立刻被踢掉。

**③ 单飞用 `Condition` 而不是信号量。** 需要的语义是"等一个条件成立"（空闲 + 装得下），不是"计数"。`Condition.wait(0.05)` 的轮询间隔是为了让超时可控——纯 `wait()` 在 eviction 场景会漏掉唤醒。

---

## 状态机

一个模型在一个时刻只有两种状态：**resident**（占着显存）或 **absent**。没有"加载中"——加载发生在持有锁的临界区外（真实后端里），期间 `busy=True`，别的请求进不来。

请求的生命周期：

```
queued → running → released
   │         │
   │         └─ release(): busy=False, LRU 刷新, notify_all
   └─ 超时/队列满 → 拒绝（不计入 admitted）
```

`/metrics` 里的计数直接映射这三个转移，便于接监控。

---

## 为什么不做多卡 / 分布式

能做，但会把仓库变成一个"半成品调度框架"，而这正是市场上已经饱和的东西（Ray Serve、KServe、vLLM 的 tensor parallel）。

这个仓库要证明的是：**在资源受限的单机上，把多模型共存这件事想清楚、写对、测到。** 所以有意停在单卡、单进程、单飞。

---

## 可扩展点（留给后续）

| 方向 | 改动位置 | 难度 |
|---|---|---|
| 按 `kind` 分池（chat 池 / audio 池各自排队） | `Scheduler` 拆成 `dict[kind] -> Scheduler` | 小 |
| 请求取消（客户端断开即释放） | `Handler` 捕获 `BrokenPipeError` → `lease.release()` | 小 |
| 真实后端 | 实现 `Backend` 协议，注册到 `make_backend` | 小 |
| 多卡 | 每个 `Scheduler` 绑一个 `device`，外层按显存做路由 | 中 |
| 抢占（高优先级踢低优先级） | `Lease` 加优先级 + 可中断检查点 | 大 |
