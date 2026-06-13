# NVMOS: Non-Verbal Vocalization Quality Assessment in Speech

<p>
  <a href="https://huggingface.co/maimai11/NVMOS"><img src="https://img.shields.io/badge/Hugging%20Face-NVMOS-yellow" alt="Hugging Face"></a>
  <a href="https://yongaifadian1.github.io/NVMOS/"><img src="https://img.shields.io/badge/Demo-Page-green" alt="Demo Page"></a>
  <a href="#english"><img src="https://img.shields.io/badge/Language-English-blue" alt="English"></a>
  <a href="#chinese"><img src="https://img.shields.io/badge/Language-%E4%B8%AD%E6%96%87-red" alt="中文"></a>
</p>

<a id="english"></a>

NVMOS predicts the perceptual quality of a target non-verbal vocalization (NV) in speech. The input is an audio file and its text containing one explicit NV tag such as `[laugh]`, `[sigh]`, or `[cough]`. The output is a MOS-like quality score in the range 0-5 for the marked NV event.

This repository contains inference code for the released SPEAR-L9 NVMOS model. The scorer weights are hosted on Hugging Face at [`maimai11/NVMOS`](https://huggingface.co/maimai11/NVMOS).

## Model

The released inference pipeline uses:

- audio representation: SPEAR Large Speech-Audio, layer 9
- text query representation: XLM-R Large hidden states averaged over the target tag span inside `[tag]`
- downstream scorer: 2-layer, 8-head text-query cross-attention with a regression head

The audio and text encoders are downloaded automatically from Hugging Face on first use. The NVMOS scorer checkpoint is downloaded from `maimai11/NVMOS`.

## Supported NV Categories

| Category | Tag |
| --- | --- |
| Ahem | `[ahem]` |
| Chuckle | `[chuckle]` |
| Cough | `[cough]` |
| Cry | `[cry]` |
| Exhale | `[exhale]` |
| Hiss | `[hiss]` |
| Hum | `[hum]` |
| Inhale | `[inhale]` |
| Laugh | `[laugh]` |
| Moan | `[moan]` |
| Pant | `[pant]` |
| Sigh | `[sigh]` |
| Smack | `[smack]` |
| Sneeze | `[sneeze]` |
| Sniffle | `[sniffle]` |
| Snore | `[snore]` |

## Installation

Create a clean conda environment:

```bash
conda create -n nvmos python=3.10 -y
conda activate nvmos
pip install --upgrade pip
pip install -r requirements.txt
```

## Inference

First run the two provided examples to verify the environment and model download. They correspond to representative low- and high-quality NV events from the demo page:

```bash
python infer.py \
  --audio examples/example_low_cry_pred0.82.wav \
  --text "$(cat examples/example_low_cry_pred0.82.txt)"

python infer.py \
  --audio examples/example_high_ahem_pred4.10.mp3 \
  --text "$(cat examples/example_high_ahem_pred4.10.txt)"
```

Then run prediction with your own audio file and the corresponding text. The text must contain exactly one NV tag at its actual position:

```bash
python infer.py \
  --audio /path/to/your_audio.wav \
  --text "I tried to explain the situation, but honestly it was just too awkward. [laugh]"
```

Example output:

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

## Notes

- By default, audio is truncated to 12 seconds, matching the training feature extraction setting.
- The score estimates the quality of the marked NV event, not the overall utterance quality.

## Citation

If you use NVMOS, please cite the corresponding paper and model release.

---

<a id="chinese"></a>

# NVMOS 中文说明

<p>
  <a href="#english"><img src="https://img.shields.io/badge/Language-English-blue" alt="English"></a>
  <a href="https://huggingface.co/maimai11/NVMOS"><img src="https://img.shields.io/badge/Hugging%20Face-NVMOS-yellow" alt="Hugging Face"></a>
  <a href="https://yongaifadian1.github.io/NVMOS/"><img src="https://img.shields.io/badge/Demo-%E9%A1%B5%E9%9D%A2-green" alt="Demo 页面"></a>
</p>

NVMOS 用于预测语音中目标非语言发声事件（non-verbal vocalization, NV）的感知质量。输入是一段音频及其对应文本，文本中需要包含一个显式 NV 标签，例如 `[laugh]`、`[sigh]` 或 `[cough]`。输出是针对该标记 NV 事件的 0-5 分 MOS 风格质量分数。

本仓库提供基于 SPEAR 第 9 层特征的 NVMOS 推理代码。评分器权重发布在 Hugging Face：[`maimai11/NVMOS`](https://huggingface.co/maimai11/NVMOS)。

## 模型

当前发布的推理流程使用：

- 音频表征：SPEAR Large Speech-Audio，第 9 层
- 文本查询表征：XLM-R Large 中 `[tag]` 内目标标签 span 对应 hidden states 的平均
- 下游评分器：2 层、8 头文本查询交叉注意力模块和回归头

首次运行时，音频编码器、文本编码器和 NVMOS 评分器权重会自动从 Hugging Face 下载。

## 支持的 NV 类别

| 类别 | 标签 |
| --- | --- |
| 清嗓 / Ahem | `[ahem]` |
| 轻笑 / Chuckle | `[chuckle]` |
| 咳嗽 / Cough | `[cough]` |
| 哭声 / Cry | `[cry]` |
| 呼气 / Exhale | `[exhale]` |
| 嘶声 / Hiss | `[hiss]` |
| 哼声 / Hum | `[hum]` |
| 吸气 / Inhale | `[inhale]` |
| 笑声 / Laugh | `[laugh]` |
| 呻吟 / Moan | `[moan]` |
| 喘息 / Pant | `[pant]` |
| 叹气 / Sigh | `[sigh]` |
| 咂嘴 / Smack | `[smack]` |
| 喷嚏 / Sneeze | `[sneeze]` |
| 抽鼻 / Sniffle | `[sniffle]` |
| 鼾声 / Snore | `[snore]` |

## 环境安装

创建干净的 conda 环境：

```bash
conda create -n nvmos python=3.10 -y
conda activate nvmos
pip install --upgrade pip
pip install -r requirements.txt
```

## 推理

先运行仓库中提供的两个示例，确认环境和模型下载可以正常工作。这两个示例对应 demo 页面中的低质量和高质量 NV 事件：

```bash
python infer.py \
  --audio examples/example_low_cry_pred0.82.wav \
  --text "$(cat examples/example_low_cry_pred0.82.txt)"

python infer.py \
  --audio examples/example_high_ahem_pred4.10.mp3 \
  --text "$(cat examples/example_high_ahem_pred4.10.txt)"
```

然后再替换为自己的音频和对应文本进行评估。文本中必须在实际位置包含且只包含一个 NV 标签：

```bash
python infer.py \
  --audio /path/to/your_audio.wav \
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

## 注意事项

- 默认情况下，音频会被截断为 12 秒，与训练特征提取设置相匹配。
- 该分数评估的是标记的 NV 事件的质量，而不是整体话语质量。

## 引用

如果使用 NVMOS，请引用对应论文和模型发布。
