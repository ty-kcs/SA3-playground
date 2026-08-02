"""Thin wrappers for SA3 Colab playground."""

from __future__ import annotations

import torch

from audio_util import TARGET_SR, normalize, to_init_tuple
from rf_inversion_sa3 import invert_and_edit_sa3

DEFAULT_SEED = 1234

_MODEL_CACHE: dict[str, object] = {}


def load_model(name: str, device: str = "cuda"):
    """Load SA3 model with simple in-session cache."""
    if name not in _MODEL_CACHE:
        from stable_audio_3 import StableAudioModel

        _MODEL_CACHE[name] = StableAudioModel.from_pretrained(name, device=device)
    return _MODEL_CACHE[name]


def _duration_from_wav(wav: torch.Tensor, sr: int = TARGET_SR) -> float:
    return wav.shape[-1] / sr


def run_ia(
    model,
    wav: torch.Tensor,
    *,
    sigma: float,
    prompt: str,
    steps: int,
    cfg: float,
    seed: int = DEFAULT_SEED,
) -> torch.Tensor:
    wav = normalize(wav)
    duration = _duration_from_wav(wav)
    init = to_init_tuple(wav)
    out = model.generate(
        init_audio=init,
        init_noise_level=sigma,
        prompt=prompt,
        duration=duration,
        steps=steps,
        cfg_scale=cfg,
        seed=seed,
    )
    return out


def run_inpaint(
    model,
    wav: torch.Tensor,
    *,
    mask_start: float,
    mask_end: float,
    prompt: str,
    duration: float,
    steps: int,
    cfg: float,
    seed: int = DEFAULT_SEED,
) -> torch.Tensor:
    wav = normalize(wav)
    init = to_init_tuple(wav)
    out = model.generate(
        inpaint_audio=init,
        inpaint_mask_start_seconds=mask_start,
        inpaint_mask_end_seconds=mask_end,
        prompt=prompt,
        duration=duration,
        steps=steps,
        cfg_scale=cfg,
        seed=seed,
    )
    return out


def _encode_latent(model, wav: torch.Tensor):
    wav = normalize(wav)
    duration = _duration_from_wav(wav)
    sample_size = wav.shape[-1]
    init = to_init_tuple(wav)
    latent, _ = model._encode_audio_input(init, sample_size)
    return latent.to(model.device), duration


def _decode_latent(model, latent: torch.Tensor, max_samples: int) -> torch.Tensor:
    pre = model.model.pretransform
    with torch.inference_mode():
        dtype = next(pre.parameters()).dtype
        wav = pre.decode(latent.to(dtype))
    if wav.shape[-1] > max_samples:
        wav = wav[..., :max_samples]
    return wav


def _build_sigmas(model, latent, steps: int) -> torch.Tensor:
    from stable_audio_3.inference.sampling import build_schedule

    return build_schedule(
        steps=steps,
        sigma_max=1.0,
        dist_shift=model.model.sampling_dist_shift,
        effective_seq_len=latent.shape[-1],
        fallback_seq_len=latent.shape[-1],
        include_endpoint=True,
        device=str(model.device),
    )


def run_inversion_sanity(model, wav: torch.Tensor, *, seed: int = DEFAULT_SEED) -> torch.Tensor:
    """Fixed-params round-trip: empty prompt, gamma=0, eta=0."""
    latent, _duration = _encode_latent(model, wav)
    steps = 50
    sigmas = _build_sigmas(model, latent, steps)
    edited, _ = invert_and_edit_sa3(
        model,
        latent,
        "",
        inversion_steps=steps,
        sampling_steps=steps,
        gamma=0.0,
        eta=0.0,
        cfg_scale=1.0,
        device=str(model.device),
        disable_tqdm=True,
        seed=seed,
        sigmas=sigmas,
    )
    max_samples = wav.shape[-1]
    return _decode_latent(model, edited, max_samples)


def run_inversion_edit(
    model,
    wav: torch.Tensor,
    *,
    prompt: str,
    gamma: float,
    eta: float,
    s: float,
    tau: float,
    cfg: float,
    inv_steps: int,
    samp_steps: int,
    seed: int = DEFAULT_SEED,
) -> torch.Tensor:
    latent, _duration = _encode_latent(model, wav)
    sigmas = _build_sigmas(model, latent, inv_steps)
    edited, _ = invert_and_edit_sa3(
        model,
        latent,
        prompt,
        inversion_steps=inv_steps,
        sampling_steps=samp_steps,
        gamma=gamma,
        eta=eta,
        s=s,
        tau=tau,
        cfg_scale=cfg,
        device=str(model.device),
        disable_tqdm=True,
        seed=seed,
        sigmas=sigmas,
    )
    max_samples = wav.shape[-1]
    return _decode_latent(model, edited, max_samples)
