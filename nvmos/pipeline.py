from __future__ import annotations

import json
import re
import wave
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchaudio
from huggingface_hub import hf_hub_download
from transformers import AutoModel, AutoTokenizer

from .model import TextQueryNVMOS


def normalize_nv_tag_text(text: str, tag: str) -> tuple[str, tuple[int, int]]:
    if not text:
        text = f"[{tag}]"
        return text, (1, 1 + len(tag))
    pattern = re.compile(r"\[" + re.escape(tag) + r"\]", re.IGNORECASE)
    match = pattern.search(text)
    if match:
        return text, (match.start() + 1, match.end() - 1)
    text = f"{text} [{tag}]"
    start = len(text) - len(tag) - 1
    return text, (start, start + len(tag))


def infer_tag(text: str, tag: str | None = None) -> str:
    if tag:
        return tag.strip().strip("[]")
    match = re.search(r"\[([^\[\]]+)\]", text)
    if not match:
        raise ValueError("No NV tag found. Provide text containing e.g. '[laugh]' or pass --tag laugh.")
    return match.group(1).strip()


class NVMOSPipeline:
    def __init__(
        self,
        model_id: str = "maimai11/NVMOS",
        device: str | None = None,
        audio_encoder: str | None = None,
        text_encoder: str | None = None,
        local_files_only: bool = False,
    ):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        config_path = hf_hub_download(model_id, "config.json", local_files_only=local_files_only)
        self.config: dict[str, Any] = json.loads(Path(config_path).read_text(encoding="utf-8"))
        self.audio_encoder_id = audio_encoder or self.config["audio_encoder"]
        self.text_encoder_id = text_encoder or self.config["text_encoder"]

        self.audio_model = AutoModel.from_pretrained(
            self.audio_encoder_id,
            trust_remote_code=True,
            local_files_only=local_files_only,
        ).to(self.device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(self.text_encoder_id, local_files_only=local_files_only)
        self.text_model = AutoModel.from_pretrained(self.text_encoder_id, local_files_only=local_files_only).to(self.device).eval()

        self.scorer = TextQueryNVMOS(
            audio_dim=int(self.config.get("audio_dim", 1024)),
            text_dim=int(self.config.get("text_dim", 1024)),
            hidden_dim=int(self.config.get("hidden_dim", 256)),
            heads=int(self.config.get("attention_heads", 8)),
            layers=int(self.config.get("cross_layers", 2)),
            ffn_dim=int(self.config.get("ffn_dim", 1024)),
            dropout=float(self.config.get("dropout", 0.1)),
        ).to(self.device).eval()
        weight_path = hf_hub_download(model_id, "nvmos_spear_l9.pt", local_files_only=local_files_only)
        state = torch.load(weight_path, map_location=self.device)
        self.scorer.load_state_dict(state)

    def load_audio(self, audio_path: str | Path) -> torch.Tensor:
        audio_path = Path(audio_path)
        try:
            wav, sr = torchaudio.load(str(audio_path))
        except RuntimeError:
            if audio_path.suffix.lower() != ".wav":
                raise
            wav, sr = self._load_pcm_wav(audio_path)
        wav = wav.mean(dim=0)
        target_sr = int(self.config.get("sample_rate", 16000))
        if sr != target_sr:
            wav = torchaudio.functional.resample(wav, sr, target_sr)
        max_audio_sec = float(self.config.get("max_audio_sec", 12.0))
        if max_audio_sec > 0:
            wav = wav[: int(max_audio_sec * target_sr)]
        return wav

    @staticmethod
    def _load_pcm_wav(audio_path: Path) -> tuple[torch.Tensor, int]:
        with wave.open(str(audio_path), "rb") as wav_file:
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            sr = wav_file.getframerate()
            frames = wav_file.readframes(wav_file.getnframes())
        if sample_width == 1:
            data = np.frombuffer(frames, dtype=np.uint8).astype(np.float32)
            data = (data - 128.0) / 128.0
        elif sample_width == 2:
            data = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
        elif sample_width == 4:
            data = np.frombuffer(frames, dtype="<i4").astype(np.float32) / 2147483648.0
        else:
            raise RuntimeError(f"Unsupported PCM WAV sample width: {sample_width} bytes")
        wav = torch.from_numpy(data.reshape(-1, channels).T.copy())
        return wav, sr

    @torch.no_grad()
    def extract_audio_features(self, audio_path: str | Path) -> tuple[torch.Tensor, torch.Tensor]:
        wav = self.load_audio(audio_path).to(self.device)
        audio = wav.unsqueeze(0)
        audio_len = torch.tensor([wav.numel()], device=self.device)
        out = self.audio_model(audio, audio_len)
        hidden_states = out["hidden_states"]
        layer = int(self.config.get("audio_layer", 9))
        if layer >= len(hidden_states):
            raise IndexError(f"SPEAR returned {len(hidden_states)} hidden states, requested layer {layer}")
        frames = hidden_states[layer]
        lens = out.get("encoder_out_lens")
        valid_len = int(lens[0].item()) if lens is not None else int(frames.shape[1])
        frames = frames[:, :valid_len].float()
        key_padding_mask = torch.zeros(1, frames.shape[1], dtype=torch.bool, device=self.device)
        return frames, key_padding_mask

    @torch.no_grad()
    def extract_tag_query(self, text: str, tag: str) -> tuple[torch.Tensor, str]:
        normalized_text, span = normalize_nv_tag_text(text, tag)
        enc = self.tokenizer(
            normalized_text,
            return_tensors="pt",
            truncation=True,
            max_length=int(self.config.get("max_text_len", 192)),
            return_offsets_mapping=True,
        )
        offsets = enc.pop("offset_mapping")[0].tolist()
        enc = {k: v.to(self.device) for k, v in enc.items()}
        hidden = self.text_model(**enc).last_hidden_state[0]
        mask = [start < span[1] and end > span[0] and end > start for start, end in offsets]
        idx = torch.tensor(mask, device=self.device, dtype=torch.bool)
        if not idx.any():
            idx[0] = True
        return hidden[idx].mean(dim=0, keepdim=True).float(), normalized_text

    @torch.no_grad()
    def predict(self, audio_path: str | Path, text: str, tag: str | None = None) -> dict[str, Any]:
        tag = infer_tag(text, tag)
        frames, key_padding_mask = self.extract_audio_features(audio_path)
        tag_ctx, normalized_text = self.extract_tag_query(text, tag)
        score, attn = self.scorer(frames, tag_ctx, key_padding_mask)
        raw_score = float(score.item())
        lo, hi = self.config.get("score_range", [0, 5])
        clipped = max(float(lo), min(float(hi), raw_score))
        return {
            "score": clipped,
            "raw_score": raw_score,
            "target_tag": tag,
            "text": normalized_text,
            "audio_path": str(audio_path),
            "audio_frames": int(frames.shape[1]),
            "attention_peak_frame": int(attn[0, 0].argmax().item()) if attn is not None else None,
        }
