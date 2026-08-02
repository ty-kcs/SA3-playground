"""
RF-Inversion for Stable Audio 3 (rectified_flow models).

Same structure as rf_inversion.py. SA3-only differences marked with comments:
  - t=0 data, t=1 noise (distribution_shift.py, sample_discrete_euler)
  - Phase A walks 0→1; sampling uses model(x, t) without negate
  - schedule default "linear" (+ optional sigmas= from build_schedule)
  - invert_and_edit_sa3 wires SA3 inpaint_mask / inpaint_masked_input
"""

import torch
from tqdm import tqdm


def _apply_dist_shift(dist_shift, t, seq_len):
    if dist_shift is None:
        return t
    if hasattr(dist_shift, "shift"):
        return dist_shift.shift(t, seq_len)
    if hasattr(dist_shift, "time_shift"):
        return dist_shift.time_shift(t, seq_len)
    raise AttributeError("dist_shift must provide shift() or time_shift()")


def inversion_rsde(
    model_fn,
    noise,
    init_data,
    steps=8,
    gamma=0.2,
    device="cuda",
    schedule="linear",
    dist_shift=None,
    disable_tqdm=False,
    callback=None,
    sigmas=None,
    **extra_args,
):
    batch_size = init_data.shape[0]
    Y_t = init_data.to(device)
    noise = noise.to(device)

    # SA3: 0→1 via flip of standard 1→0 schedule (open-small walks 1→0 here)
    if sigmas is not None:
        t = torch.flip(sigmas.to(device), dims=[0])
    else:
        if schedule == "log":
            log_vals = torch.linspace(-6, 2, steps + 1, device=device)
            t = torch.sigmoid(-log_vals)
            t[0] = 1.0
            t[-1] = 0.0
        else:  # linear
            t = torch.linspace(1.0, 0.0, steps + 1, device=device)

        # Apply dist_shift if provided (like build_schedule / sample_diffusion)
        if dist_shift is not None:
            t = _apply_dist_shift(dist_shift, t, init_data.shape[-1])
            t = t.clone()
            t[0] = 1.0
            t[-1] = 0.0

        t = torch.flip(t, dims=[0])

    for i, (t_curr, t_next) in enumerate(
        tqdm(zip(t[:-1], t[1:]), disable=disable_tqdm, desc="Inversion")
    ):
        dt = t_next - t_curr  # SA3: positive along 0→1 (open-small: t_curr - t_next)

        ts = t_curr.expand(batch_size)

        u_uncond = model_fn(Y_t, ts, **extra_args)
        u_cond = (noise - Y_t) / torch.clamp(1.0 - t_curr, min=1e-6)

        # --- Norm matching (prevents one field from dominating) ---
        # Compute L2 norms over (channels, time); keep batch dims
        uncond_norm = u_uncond.norm(dim=(1, 2), keepdim=True) + 1e-8
        cond_norm = u_cond.norm(dim=(1, 2), keepdim=True) + 1e-8
        scale = (uncond_norm / cond_norm).clamp(0.0, 3.0)
        u_cond_nm = scale * u_cond

        # Linear blend in the velocity space
        u = u_uncond + gamma * (u_cond_nm - u_uncond)

        # --- Output rescale (CFG-style) to preserve the base field's scale ---
        out_std = u.std(dim=(1, 2), keepdim=True) + 1e-8
        uncond_std = u_uncond.std(dim=(1, 2), keepdim=True) + 1e-8
        u = u * (uncond_std / out_std)

        # u = u_uncond + gamma * (u_cond - u_uncond)

        Y_t = Y_t + dt * u

        # Call callback if provided
        if callback is not None:
            callback(
                {
                    "x": Y_t,
                    "t": t_curr,
                    "step": i + 1,
                    "total_steps": len(t) - 1,
                    "v_uncond": u_uncond,
                    "v_cond": u_cond,
                    "v": u,
                }
            )

    return Y_t


def sample_rsde(
    model_fn,
    noise,
    init_data,
    steps=8,
    eta=0.4,
    s=0.0,
    tau=1.0,
    device="cuda",
    schedule="linear",
    dist_shift=None,
    disable_tqdm=False,
    callback=None,
    sigmas=None,
    **extra_args,
):
    batch_size = noise.shape[0]
    X_t = noise.to(device)

    if sigmas is not None:
        t = sigmas.to(device)
    elif schedule == "log":
        log_vals = torch.linspace(-6, 2, steps + 1, device=device)
        t = torch.sigmoid(-log_vals)
        t[0] = 1.0
        t[-1] = 0.0
    else:  # linear
        t = torch.linspace(1.0, 0.0, steps + 1, device=device)

    tau_step = int(tau * (len(t) - 1))
    s_step = int(s * (len(t) - 1))

    # Apply dist_shift if provided (like build_schedule / sample_diffusion)
    if sigmas is None and dist_shift is not None:
        t = _apply_dist_shift(dist_shift, t, noise.shape[-1])
        t = t.clone()
        t[0] = 1.0
        t[-1] = 0.0

    for i, (t_curr, t_next) in enumerate(
        tqdm(zip(t[:-1], t[1:]), disable=disable_tqdm, desc="Sampling")
    ):
        dt = t_next - t_curr  # SA3: negative along 1→0 
        ts = t_curr.expand(batch_size)  

        u_uncond = model_fn(X_t, ts, **extra_args)  # SA3 (open-small: -model_fn(...))

        # Editing window (same logic as your script; just uses correct t/dt now)
        in_window = (i >= s_step) and (i <= tau_step)
        u_cond = None
        if eta != 0.0 and in_window:
            u_cond = (init_data - X_t) / torch.clamp(t_curr, min=1e-6)  # SA3 (open-small: 1-t)

            # --- Norm matching (prevents one field from dominating) ---
            # Compute L2 norms over (channels, time); keep batch dims
            uncond_norm = u_uncond.norm(dim=(1, 2), keepdim=True) + 1e-8
            cond_norm = u_cond.norm(dim=(1, 2), keepdim=True) + 1e-8
            scale = (uncond_norm / cond_norm).clamp(0.0, 3.0)
            u_cond_nm = scale * u_cond

            # Linear blend in the velocity space
            u = u_uncond + eta * (u_cond_nm - u_uncond)

            # --- Output rescale (CFG-style) to preserve the base field's scale ---
            out_std = u.std(dim=(1, 2), keepdim=True) + 1e-8
            uncond_std = u_uncond.std(dim=(1, 2), keepdim=True) + 1e-8
            u = u * (uncond_std / out_std)

            # u = u_uncond + eta * (u_cond - u_uncond)
        else:
            u = u_uncond

        denoised = X_t - t_curr * u_uncond  # SA3: I-B-style preview (sample_discrete_euler)

        X_t = X_t + dt * u

        # Call callback if provided
        if callback is not None:
            callback(
                {
                    "x": X_t,
                    "t": t_curr,
                    "step": i + 1,
                    "total_steps": len(t) - 1,
                    "u_uncond": u_uncond,
                    "u_cond": u_cond,
                    "u": u,
                    "denoised": denoised,
                }
            )

    return X_t


def invert_and_edit_sa3(
    model,
    init_audio,
    target_prompt,
    inversion_steps=8,
    sampling_steps=8,
    gamma=0.5,
    eta=0.4,
    s=0,
    tau=1.0,
    cfg_scale=1.0,
    device="cuda",
    disable_tqdm=False,
    seed=None,
    sigmas=None,
):
    """
    Complete inversion and editing pipeline

    Args:
        model: StableAudioModel (SA3)
        init_audio: Input audio latent
        target_prompt: Text prompt for editing
        inversion_steps: Steps for inversion phase
        sampling_steps: Steps for sampling phase
        gamma: Forward controller strength
        eta: Reverse controller strength
        s: Start step for reverse controller
        tau: End fraction for reverse controller
        device: Device to run on
        seed: Random seed for reproducible noise (None for random)
        sigmas: Optional build_schedule() tensor (SA3 standard inference timesteps)

    Returns:
        edited_audio: Final edited audio latent
        inverted_noise: Intermediate structured noise (for debugging)
    """
    # Generate noise for inversion
    if seed is not None:
        torch.manual_seed(seed)
    noise = torch.randn_like(init_audio)

    # Create empty conditioning for inversion
    audio_length_sec = (
        init_audio.shape[-1] * model.model.pretransform.downsampling_ratio
    ) / model.model.sample_rate
    print(f"Audio length (sec): {audio_length_sec:.2f}")
    empty_conditioning = [
        {"prompt": "", "seconds_start": 0, "seconds_total": audio_length_sec}
    ]
    empty_conditioning_tensors = model.model.conditioner(empty_conditioning, device)
    # SA3: match generate() — zero inpaint when not inpainting
    batch_size = init_audio.shape[0]
    latent_seq_len = init_audio.shape[-1]
    empty_conditioning_tensors["inpaint_mask"] = [
        torch.zeros((batch_size, 1, latent_seq_len), device=device)
    ]
    empty_conditioning_tensors["inpaint_masked_input"] = [
        torch.zeros(
            (batch_size, model.model.io_channels, latent_seq_len), device=device
        )
    ]
    empty_conditioning_inputs = model.model.get_conditioning_inputs(
        empty_conditioning_tensors
    )

    # Phase A: Inversion
    inverted_noise = inversion_rsde(
        model.model.model,
        noise,
        init_audio,
        steps=inversion_steps,
        gamma=gamma,
        device=device,
        disable_tqdm=disable_tqdm,
        dist_shift=model.model.sampling_dist_shift,
        sigmas=sigmas,
        **empty_conditioning_inputs,
    )

    # Create target conditioning for sampling
    target_conditioning = [
        {"prompt": target_prompt, "seconds_start": 0, "seconds_total": audio_length_sec}
    ]
    target_conditioning_tensors = model.model.conditioner(target_conditioning, device)
    target_conditioning_tensors["inpaint_mask"] = [
        torch.zeros((batch_size, 1, latent_seq_len), device=device)
    ]
    target_conditioning_tensors["inpaint_masked_input"] = [
        torch.zeros(
            (batch_size, model.model.io_channels, latent_seq_len), device=device
        )
    ]
    target_conditioning_inputs = model.model.get_conditioning_inputs(
        target_conditioning_tensors
    )

    # Phase B: Editing
    edited_audio = sample_rsde(
        model.model.model,
        inverted_noise,
        init_audio,
        steps=sampling_steps,
        eta=eta,
        s=s,
        tau=tau,
        device=device,
        disable_tqdm=disable_tqdm,
        dist_shift=model.model.sampling_dist_shift,
        sigmas=sigmas,
        **target_conditioning_inputs,
        cfg_scale=cfg_scale,
    )

    return edited_audio, inverted_noise
