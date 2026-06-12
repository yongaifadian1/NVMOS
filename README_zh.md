# NVMOS

[English](README.md) | [中文](README_zh.md)

NVMOS 用于预测语音中目标非语言发声事件（non-verbal vocalization, NV）的感知质量。输入是一段音频及其对应文本，文本中需要包含一个显式 NV 标签，例如 `[laugh]`、`[sigh]` 或 `[cough]`。输出是针对该标记 NV 事件的 0-5 分 MOS 风格质量分数。

本仓库提供基于 SPEAR 第 9 层特征的 NVMOS 推理代码。评分器权重发布在 Hugging Face：[`maimai11/NVMOS`](https://huggingface.co/maimai11/NVMOS)。

## 模型

当前发布的推理流程使用：

- 音频表征：SPEAR Large Speech-Audio，第 9 层
- 文本查询表征：XLM-R Large 中 `[tag]` 内目标标签 span 对应 hidden states 的平均
- 下游评分器：2 层、8 头文本查询交叉注意力模块和回归头

首次运行时，音频编码器、文本编码器和 NVMOS 评分器权重会自动从 Hugging Face 下载。

## 环境安装

创建干净的 conda 环境：

```bash
conda create -n nvmos python=3.10 -y
conda activate nvmos
pip install --upgrade pip
pip install -r requirements.txt
```

## 推理

使用音频文件和对应文本进行推理。文本中必须在实际位置包含且只包含一个 NV 标签：

```bash
python infer.py \
  --audio /path/to/audio.wav \
  --text "I tried to explain the situation, but honestly it was just too awkward. [laugh]"
```

输出示例：

```json
{
  "score": 3.72,
  "raw_score": 3.72,
  "target_tag": "laugh",
  "text": "I tried to explain the situation, but honestly it was just too awkward. [laugh]",
  "audio_path": "/path/to/audio.wav",
  "audio_frames": 248,
  "attention_peak_frame": 137
}
```

## 手动下载模型文件

默认推理命令会自动下载所需文件。如果只想提前下载 NVMOS 评分器文件：

```bash
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("maimai11/NVMOS")
PY
```

上游编码器在 `model/config.json` 中指定：

- `marcoyang/spear-large-speech-audio`
- `FacebookAI/xlm-roberta-large`

## 注意事项

- 默认情况下，音频会被截断为 12 秒，与训练特征提取设置相匹配。
- 该分数评估的是标记的 NV 事件的质量，而不是整体话语质量。

## 引用

如果使用 NVMOS，请引用对应论文和模型发布。
