# NVMOS

[English](README.md) | [中文](README_zh.md)

NVMOS predicts the perceptual quality of a target non-verbal vocalization (NV) in speech. The input is an audio file and its text containing one explicit NV tag such as `[laugh]`, `[sigh]`, or `[cough]`. The output is a MOS-like quality score in the range 0-5 for the marked NV event.

This repository contains inference code for the released SPEAR-L9 NVMOS model. The scorer weights are hosted on Hugging Face at [`maimai11/NVMOS`](https://huggingface.co/maimai11/NVMOS).

## Model

The released inference pipeline uses:

- audio representation: SPEAR Large Speech-Audio, layer 9
- text query representation: XLM-R Large hidden states averaged over the target tag span inside `[tag]`
- downstream scorer: 2-layer, 8-head text-query cross-attention with a regression head

The audio and text encoders are downloaded automatically from Hugging Face on first use. The NVMOS scorer checkpoint is downloaded from `maimai11/NVMOS`.

## Installation

Create a clean conda environment:

```bash
conda create -n nvmos python=3.10 -y
conda activate nvmos
pip install --upgrade pip
pip install -r requirements.txt
```

## Inference

Run prediction with an audio file and the corresponding text containing exactly one NV tag at its actual position:

```bash
python infer.py \
  --audio /path/to/audio.wav \
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

## Downloading Model Files Manually

The default command downloads all required files automatically. To pre-download only the NVMOS scorer files:

```bash
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("maimai11/NVMOS")
PY
```

The upstream encoders are specified in `model/config.json` in the model repo:

- `marcoyang/spear-large-speech-audio`
- `FacebookAI/xlm-roberta-large`

## Notes

- By default, audio is truncated to 12 seconds, matching the training feature extraction setting.
- The score estimates the quality of the marked NV event, not the overall utterance quality.

## Citation

If you use NVMOS, please cite the corresponding paper and model release.
