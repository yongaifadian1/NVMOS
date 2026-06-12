from __future__ import annotations

import json
import re
import struct
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchaudio
from huggingface_hub import hf_hub_download, snapshot_download
from transformers import AutoModel, AutoTokenizer

from .model import TextQueryNVMOS


def parse_single_nv_tag(text: str, tag: str | None = None) -> tuple[str, tuple[int, int]]:
    matches = list(re.finditer(r"\[([^\[\]]+)\]", text))
    if not matches:
        raise ValueError("No bracketed NV tag found. Provide text containing exactly one target tag at its actual position, e.g. '[laugh]'.")
    if len(matches) > 1:
        raise ValueError("Multiple bracketed NV tags found. NVMOS inference expects exactly one target tag in the text.")
    match = matches[0]
    parsed_tag = match.group(1).strip()
    if tag and tag.strip().strip("[]").lower() != parsed_tag.lower():
        raise ValueError(f"Provided tag '{tag}' does not match the single text tag '[{parsed_tag}]'.")
    return parsed_tag, (match.start() + 1, match.end() - 1)


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
        text_encoder_source = self._resolve_model_source(self.text_encoder_id, local_files_only)

        self.audio_model = AutoModel.from_pretrained(
            self.audio_encoder_id,
            trust_remote_code=True,
            local_files_only=local_files_only,
        ).to(self.device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(text_encoder_source, local_files_only=local_files_only)
        self.text_model = AutoModel.from_pretrained(text_encoder_source, local_files_only=local_files_only).to(self.device).eval()

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
        state = torch.load(weight_path, map_location=self.device, weights_only=True)
        self.scorer.load_state_dict(state)

    @staticmethod
    def _resolve_model_source(model_id_or_path: str, local_files_only: bool) -> str:
        if Path(model_id_or_path).exists() or not local_files_only:
            return model_id_or_path
        return snapshot_download(model_id_or_path, local_files_only=True)

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
        data = audio_path.read_bytes()
        if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
            raise RuntimeError(f"Unsupported WAV container: {audio_path}")

        fmt: tuple[int, int, int, int] | None = None
        frames = None
        pos = 12
        while pos + 8 <= len(data):
            chunk_id = data[pos : pos + 4]
            chunk_size = struct.unpack_from("<I", data, pos + 4)[0]
            chunk_start = pos + 8
            chunk_end = chunk_start + chunk_size
            if chunk_id == b"fmt ":
                audio_format, channels, sr, _, _, bits = struct.unpack_from("<HHIIHH", data, chunk_start)
                fmt = (audio_format, channels, sr, bits)
            elif chunk_id == b"data":
                frames = data[chunk_start:chunk_end]
            pos = chunk_end + (chunk_size % 2)

        if fmt is None or frames is None:
            raise RuntimeError(f"Malformed WAV file: {audio_path}")

        audio_format, channels, sr, bits = fmt
        if audio_format == 1 and bits == 8:
            data = np.frombuffer(frames, dtype=np.uint8).astype(np.float32)
            data = (data - 128.0) / 128.0
        elif audio_format == 1 and bits == 16:
            data = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
        elif audio_format == 1 and bits == 32:
            data = np.frombuffer(frames, dtype="<i4").astype(np.float32) / 2147483648.0
        elif audio_format == 3 and bits == 32:
            data = np.frombuffer(frames, dtype="<f4").astype(np.float32)
        elif audio_format == 3 and bits == 64:
            data = np.frombuffer(frames, dtype="<f8").astype(np.float32)
        else:
            raise RuntimeError(f"Unsupported WAV format tag={audio_format}, bits={bits}")
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
        _, span = parse_single_nv_tag(text, tag)
        enc = self.tokenizer(
            text,
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
        return hidden[idx].mean(dim=0, keepdim=True).float(), text

    @torch.no_grad()
    def predict(self, audio_path: str | Path, text: str, tag: str | None = None) -> dict[str, Any]:
        tag, _ = parse_single_nv_tag(text, tag)
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
