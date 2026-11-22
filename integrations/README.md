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
