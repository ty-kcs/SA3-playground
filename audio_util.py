"""audio helpers"""

from __future__ import annotations

import io
from typing import Tuple

import torch
import torchaudio

TARGET_SR = 44100


def parse_file_upload(value) -> tuple[str | None, bytes | None]:
    """Return (filename, bytes) from ipywidgets FileUpload.value, or (None, None)."""
    if not value:
        return None, None
    if isinstance(value, (list, tuple)):
        entry = value[0]
        content = entry["content"]
        if hasattr(content, "tobytes"):
            content = content.tobytes()
        return entry.get("name", "upload.wav"), content
    name = next(iter(value))
    content = value[name]["content"]
    if hasattr(content, "tobytes"):
        content = content.tobytes()
    return name, content


def load_uploaded_wav(data: bytes, target_sr: int = TARGET_SR) -> tuple[torch.Tensor, int]:
    """Load WAV bytes as float stereo tensor [channels, samples]."""
    wav, sr = torchaudio.load(io.BytesIO(data))
    return _to_stereo_resample(wav, sr, target_sr)


def load_path(path: str, target_sr: int = TARGET_SR) -> tuple[torch.Tensor, int]:
    wav, sr = torchaudio.load(path)
    return _to_stereo_resample(wav, sr, target_sr)


def _to_stereo_resample(
    wav: torch.Tensor, sr: int, target_sr: int
) -> tuple[torch.Tensor, int]:
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
        sr = target_sr
    if wav.shape[0] == 1:
        wav = wav.repeat(2, 1)
    elif wav.shape[0] > 2:
        wav = wav[:2]
    return wav, sr


def normalize(wav: torch.Tensor) -> torch.Tensor:
    peak = wav.abs().max()
    if peak > 0:
        wav = wav / peak
    return wav.clamp(-1, 1)


def to_init_tuple(wav: torch.Tensor, sr: int = TARGET_SR) -> Tuple[int, torch.Tensor]:
    """SA3 generate() expects (sample_rate, tensor [C, T])."""
    if wav.dim() == 3:
        wav = wav[0]
    return sr, wav


def tensor_for_audio_widget(wav: torch.Tensor, sr: int = TARGET_SR) -> tuple[torch.Tensor, int]:
    """Mono float32 for IPython.display.Audio."""
    if wav.dim() == 3:
        wav = wav[0]
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    return wav.cpu().float().clamp(-1, 1), sr
