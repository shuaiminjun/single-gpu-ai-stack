# 接真实模型（最小示例）

`core/backends.py` 里的 `FakeBackend` 只是为了让协议可测。真正上线要换 backend，两种方式：

## 1. 图像：ComfyUI（`ComfyUIBackend`）

`ComfyUIBackend` 已经在 `core/backends.py` 里实现：提交工作流 → 轮询 `/history` → 从 `/view` 取图。

`config.json` 里把模型指向一个工作流 JSON：

```json
{
  "id": "image-krea2",
  "kind": "image",
  "vram_gb": 12.5,
  "backend": "comfyui",
  "target": "integrations/comfyui/krea2.json"
}
```

工作流里用 `{prompt}` 占位，例如：

```json
{
  "6": {"class_type": "CLIPTextEncode",
        "inputs": {"text": "{prompt}", "clip": ["4", 1]}}
}
```

启动时 backend 的名字由 CLI 决定（`--backend comfyui`），也可以给每个模型写死 `"backend": "comfyui"`。

**两个真机坑（踩过）**：

- 大模型（12B 级）在 20 GB 卡上必须给 ComfyUI 加 `--cache-none`，否则文本编码器赖在显存里，1024² 会从 48 s 掉到 240 s/步；
- 生成完不要立刻重启 ComfyUI：驱动上下文要 6~10 s 才释放，否则直接 abort（ROCm 上表现为 `LLVM ERROR`）。

## 2. 语音 / 对话：命令行（`SubprocessBackend`）

`target` 是一行 argv 模板，`{prompt}` 会被替换：

```json
{
  "id": "tts-a",
  "kind": "audio",
  "vram_gb": 3.5,
  "backend": "subprocess",
  "target": "python bench/tts_server_client.py --text {prompt} --out -"
}
```

只要你的脚本把 **16-bit PCM** 打到 stdout，`/v1/audio/speech` 就能直接用它；调度器负责排队和淘汰。

## 3. 自己写一个 backend

实现三个方法即可（没用到的方法抛 `NotImplementedError`）：

```python
class MyBackend:
    def chat_stream(self, spec, prompt, **kw) -> Iterator[str]: ...
    def audio_stream(self, spec, text, **kw) -> Iterator[bytes]: ...
    def image(self, spec, prompt, **kw) -> bytes: ...
```

然后注册到 `make_backend()`。调度器完全不关心你怎么加载模型，只关心 `vram_gb` 是否装得下。

---

## 附带的两个样板 + 一个运维脚本

| 文件 | 用途 |
|---|---|
| `comfyui/esrgan-refine.json` | 出图 → ESRGAN 超分 → 回炉精修。**细节最好**的一档（832×1216 出图，成品 1248×1824） |
| `comfyui/latent-hires.json` | 只做 latent 放大。快一倍，细节不如上面那条 |
| `autoroute.json` | 一句话出图的关键词路由表：写"写实/水墨/猫娘"，自动选到对应模型 |

`esrgan-refine.json` / `latent-hires.json` 里的 `_meta.refine`（或 `_meta.hires`）声明了
"链怎么接"：谁出图、放大到多大、二次采样去噪多少。**这两个字段是给调用方看的**，
不填也能跑（就退化成一次出图）。

用 `ops/comfy_watchdog.py` 兜住崩溃：

```bash
python ops/comfy_watchdog.py --comfy-dir /path/to/ComfyUI --port 8188 --interval 60
```

ComfyUI 在 ROCm 上显存分配失败时是**直接 abort 进程**，不会抛异常。看门狗每 60 秒探一次
`/system_stats`，挂了就杀掉残留进程、等 12 秒（让 GPU 上下文释放，否则会连撞两次）、
再拉起来。**训练模型的时候记得先停它**，否则它会把 ComfyUI 拉起来抢显存。
