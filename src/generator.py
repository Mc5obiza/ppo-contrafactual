"""
src/generator.py
================
Pretrained Stable Diffusion v1.5 generator with LoRA fine-tuning support.

Responsibilities
----------------
- Encode images to latent space (VAE encoder)
- Decode latents back to pixel space (VAE decoder)
- Run gradient-guided DDIM denoising to produce counterfactual images
- Compute denoising training loss (L_diff) for LoRA fine-tuning

"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler
from peft import LoraConfig, get_peft_model

import config


class PretrainedLatentDiffusionGenerator(nn.Module):
    """
    Latent diffusion generator wrapping a pretrained SD1.5 VAE + UNet.

    A lightweight ``cond_proj`` MLP maps (target_class, encoded_latent)
    to cross-attention conditioning understood by the UNet.

    LoRA adapters are applied to the UNet attention layers and trained
    alongside ``cond_proj`` while the rest of the network stays frozen.

    Parameters
    ----------
    model_id : str
        HuggingFace model ID for Stable Diffusion v1.5.
    device : str
    dtype : torch.dtype
    classifier : nn.Module
        Frozen binary classifier f(·).
    seg_net : nn.Module
        Frozen lung segmentation network S(·).
    lung_channel_idx : list[int]
        Indices of the left/right lung channels in seg_net output.
    obj_det : nn.Module
        Frozen YOLOv8 feature extractor O(·).
    """

    def __init__(
        self,
        model_id: str,
        device: str,
        dtype: torch.dtype,
        classifier: nn.Module,
        seg_net: nn.Module,
        lung_channel_idx: list[int],
        obj_det: nn.Module,
    ):
        super().__init__()
        self.device = device
        self.dtype  = dtype

        print("[generator] loading VAE ...")
        self.vae = AutoencoderKL.from_pretrained(
            model_id, subfolder="vae", torch_dtype=dtype
        ).to(device)
        self.vae.eval()
        for p in self.vae.parameters():
            p.requires_grad_(False)

        print("[generator] loading UNet ...")
        self.unet = UNet2DConditionModel.from_pretrained(
            model_id, subfolder="unet", torch_dtype=dtype
        ).to(device)
        self.unet.eval()
        for p in self.unet.parameters():
            p.requires_grad_(False)

        self.scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")
        self.scheduler.set_timesteps(config.T_DIFF)
        self.scheduler.timesteps = self.scheduler.timesteps.to(device)

        self.cross_attn_dim = self.unet.config.cross_attention_dim
        cond_in_dim = (
            1 + config.LATENT_CHANNELS * config.LATENT_SIZE * config.LATENT_SIZE // 64
        )
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_in_dim, 256),
            nn.SiLU(),
            nn.Linear(256, self.cross_attn_dim),
        ).to(device=device, dtype=dtype)

        self.classifier      = classifier
        self.seg_net         = seg_net
        self.lung_channel_idx = lung_channel_idx
        self.obj_det         = obj_det

        self._to_rgb  = lambda x: x.repeat(1, 3, 1, 1) if x.shape[1] == 1 else x
        self._to_gray = lambda x: (
            0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Encode / decode
    # ──────────────────────────────────────────────────────────────────────────

    def encoder(self, x: torch.Tensor) -> torch.Tensor:
        """Map an image (1, H, W) → flat latent vector."""
        x_rgb     = self._to_rgb(x).to(self.dtype)
        posterior = self.vae.encode(x_rgb * 2.0 - 1.0)
        z = posterior.latent_dist.sample() * self.vae.config.scaling_factor
        return z.flatten(start_dim=1)

    def decoder(self, z_flat: torch.Tensor) -> torch.Tensor:
        """Map flat latent vector → grayscale image (1, H, W)."""
        B = z_flat.shape[0]
        z     = z_flat.view(
            B, config.LATENT_CHANNELS, config.LATENT_SIZE, config.LATENT_SIZE
        ).to(self.dtype)
        x_rgb = self.vae.decode(z / self.vae.config.scaling_factor).sample
        x_rgb = (x_rgb.clamp(-1, 1) + 1.0) / 2.0
        return self._to_gray(x_rgb).float()

    # ──────────────────────────────────────────────────────────────────────────
    # Conditioning
    # ──────────────────────────────────────────────────────────────────────────

    def _build_condition(
        self, c_scalar: torch.Tensor, psi_x_flat: torch.Tensor
    ) -> torch.Tensor:
        psi_small = F.adaptive_avg_pool1d(
            psi_x_flat.unsqueeze(1),
            config.LATENT_CHANNELS * config.LATENT_SIZE * config.LATENT_SIZE // 64,
        ).squeeze(1)
        cond_dtype = next(self.cond_proj.parameters()).dtype
        inp  = torch.cat([c_scalar, psi_small], dim=-1).to(cond_dtype)
        cond = self.cond_proj(inp)
        return cond.to(self.dtype).unsqueeze(1)

    # ──────────────────────────────────────────────────────────────────────────
    # Segmentation mask helper
    # ──────────────────────────────────────────────────────────────────────────

    def _get_seg_mask(self, x: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x.shape
        x_xrv = x * 2048.0 - 1024.0
        if (H, W) != (config.SEG_INPUT_SIZE, config.SEG_INPUT_SIZE):
            x_xrv = F.interpolate(
                x_xrv,
                size=(config.SEG_INPUT_SIZE, config.SEG_INPUT_SIZE),
                mode="bilinear",
                align_corners=False,
            )
        out        = self.seg_net(x_xrv.float())
        lung_logits = out[:, self.lung_channel_idx, :, :]
        lung_mask  = torch.sigmoid(lung_logits).amax(dim=1, keepdim=True)
        if (H, W) != (config.SEG_INPUT_SIZE, config.SEG_INPUT_SIZE):
            lung_mask = F.interpolate(lung_mask, size=(H, W), mode="bilinear", align_corners=False)
        return lung_mask.to(x.dtype)

    # ──────────────────────────────────────────────────────────────────────────
    # Edit-mask shaping (dilation + feathering) for background locking
    # ──────────────────────────────────────────────────────────────────────────

    def _gaussian_blur(self, m: torch.Tensor, sigma: float) -> torch.Tensor:
        """
        Separable Gaussian blur on a (B, 1, H, W) tensor. Used to feather a
        hard mask edge so the locked-background composite has no visible seam
        (the core trick Blended Latent Diffusion adds over naive RePaint).
        """
        if sigma is None or sigma <= 0.0:
            return m
        radius = max(1, int(3.0 * sigma))
        coords = torch.arange(-radius, radius + 1, device=m.device, dtype=torch.float32)
        g = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
        g = (g / g.sum()).to(m.dtype)
        kx = g.view(1, 1, 1, -1)
        ky = g.view(1, 1, -1, 1)
        m = F.conv2d(m, kx, padding=(0, radius))
        m = F.conv2d(m, ky, padding=(radius, 0))
        return m

    def _build_edit_mask(
        self, latent_mask: torch.Tensor, dilation: int, feather_sigma: float
    ) -> torch.Tensor:
        """
        Turn the raw lung latent mask into a soft *edit* mask in [0, 1]:
          - dilate (morphological, via max-pool) so opacities sitting on the
            lung boundary aren't clipped out of the editable region;
          - feather (Gaussian) so the background-lock composite blends
            smoothly instead of showing a hard cut at the lung border.
        Operates at latent resolution (LATENT_SIZE x LATENT_SIZE); a dilation
        of `d` there corresponds to ~d * IMG_SIZE/LATENT_SIZE pixels.
        """
        m = latent_mask
        if dilation and dilation > 0:
            k = 2 * dilation + 1
            m = F.max_pool2d(m, kernel_size=k, stride=1, padding=dilation)
        m = self._gaussian_blur(m, feather_sigma)
        return m.clamp(0.0, 1.0).to(latent_mask.dtype)

    # ──────────────────────────────────────────────────────────────────────────
    # Training loss (LoRA fine-tuning)
    # ──────────────────────────────────────────────────────────────────────────

    def compute_training_losses(
        self, x: torch.Tensor, c_scalar: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """
        Compute the denoising diffusion loss L_diff.

        Parameters
        ----------
        x : torch.Tensor  (B, 1, H, W)
        c_scalar : torch.Tensor  (B, 1)  — binary class labels

        Returns
        -------
        dict with key 'L_diff'
        """
        B  = x.size(0)
        z0 = self.encoder(x)
        psi_x = z0.detach()

        t_idx     = torch.randint(0, config.T_DIFF, (B,), device=self.device)
        timesteps = self.scheduler.timesteps[t_idx]
        z0_4d     = z0.view(B, config.LATENT_CHANNELS, config.LATENT_SIZE, config.LATENT_SIZE)
        noise     = torch.randn_like(z0_4d)
        z_t_4d    = self.scheduler.add_noise(z0_4d, noise, timesteps)

        cond     = self._build_condition(c_scalar, psi_x)
        eps_pred = self.unet(z_t_4d.to(self.dtype), timesteps, encoder_hidden_states=cond).sample
        L_diff   = F.mse_loss(eps_pred.float(), noise.float())
        return {"L_diff": L_diff}

    # ──────────────────────────────────────────────────────────────────────────
    # Counterfactual generation
    # ──────────────────────────────────────────────────────────────────────────

    def generate_counterfactual(
        self,
        x: torch.Tensor,
        c_scalar: float,
        M: int = 5,
        strength: float = 0.55,
        cls_guidance_scale: float = 1.0,
        pres_guidance_scale: float = 1.0,
        guidance_ramp_start_frac: float = 0.25,
        lock_background: bool = config.LOCK_BACKGROUND,
        bg_mask_dilation: int = config.BG_MASK_DILATION,
        bg_mask_feather: float = config.BG_MASK_FEATHER,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Produce a counterfactual image x_cf targeting class c_scalar.

        Parameters
        ----------
        x : torch.Tensor  (1, 1, H, W)
        c_scalar : float  — target class value (e.g. 1.0 for pneumonia)
        M : int  — number of trajectory checkpoints to save
        strength : float  — what fraction of the full T timesteps to use
        cls_guidance_scale : float  — extra multiplier for classifier guidance
        pres_guidance_scale : float  — extra multiplier for preservation guidance
        guidance_ramp_start_frac : float
            Fraction of the trajectory (0-1) at which guidance starts
            ramping up from 0. Guidance reaches full strength at the end
            of the trajectory. Replaces the old hard on/off switch at the
            trajectory midpoint with a smooth linear ramp, to avoid the
            discontinuity that could kick a step into a noisy collapse.
        lock_background : bool
            If True, apply RePaint / Blended-Latent-Diffusion style masked
            latent blending at EVERY denoising step: the region OUTSIDE the
            (dilated, feathered) lung mask is overwritten with the original
            latent re-noised to the current timestep, so it can only follow
            the identity trajectory. This is the hard spatial lock that the
            per-step guidance masking (below) does NOT provide on its own —
            masked guidance only gates the *correction gradient*, while the
            free SDEdit denoising of a high-`strength` init still repaints
            the whole thorax. With the lock on, out-of-lung change collapses
            to the VAE round-trip floor by construction. Set False to
            reproduce the old unconstrained behaviour for A/B comparison.
        bg_mask_dilation : int
            Morphological dilation (in latent cells) applied to the lung mask
            before it's used as the edit region, so boundary opacities aren't
            clipped. ~1 latent cell ≈ IMG_SIZE/LATENT_SIZE pixels.
        bg_mask_feather : float
            Gaussian sigma (in latent cells) used to feather the edit-mask
            edge so the locked-background composite has no visible seam.

        Returns
        -------
        (x_cf, trajectory) : (Tensor, list[Tensor])
        """
        device   = x.device
        c_tensor = torch.tensor([[c_scalar]], dtype=torch.float32, device=device)

        with torch.no_grad():
            z0_orig_flat = self.encoder(x)
            psi_x        = z0_orig_flat.clone()
            cond         = self._build_condition(c_tensor, psi_x)
            z0_orig_4d   = z0_orig_flat.view(
                -1, config.LATENT_CHANNELS, config.LATENT_SIZE, config.LATENT_SIZE
            )
            full_timesteps = self.scheduler.timesteps
            n_keep     = max(1, int(len(full_timesteps) * strength))
            start_idx  = len(full_timesteps) - n_keep
            timesteps  = full_timesteps[start_idx:]
            init_noise = torch.randn_like(z0_orig_4d)
            z_t = self.scheduler.add_noise(
                z0_orig_4d, init_noise, full_timesteps[start_idx : start_idx + 1]
            )

            # Precompute anchors on the ORIGINAL fixed image once. These never
            # change across the guidance loop below, so recomputing them per
            # step (as the original code did) was pure wasted compute.
            S_x = self._get_seg_mask(x)
            latent_mask = F.interpolate(
                S_x,
                size=(config.LATENT_SIZE, config.LATENT_SIZE),
                mode="bilinear",
                align_corners=False,
            )
            # Soft edit mask (dilated + feathered) used BOTH to gate the
            # guidance gradient and to composite the locked background. Using
            # one mask for both keeps the region PPO's guidance can push and
            # the region generation is allowed to change perfectly consistent.
            edit_mask = self._build_edit_mask(
                latent_mask, bg_mask_dilation, bg_mask_feather
            ).to(z_t.dtype)
            O_x = self.obj_det(self._to_rgb(x))

        trajectory     = []
        checkpoint_idx = set(np.linspace(0, len(timesteps) - 1, M, dtype=int).tolist())

        n_steps        = len(timesteps)
        ramp_start_idx = int(n_steps * guidance_ramp_start_frac)
        ramp_span      = max(1, n_steps - ramp_start_idx)  # avoid div-by-zero

        for i, t in enumerate(timesteps):
            with torch.no_grad():
                eps_hat  = self.unet(z_t, t, encoder_hidden_states=cond).sample
                step_out = self.scheduler.step(eps_hat, t, z_t)
                z0_hat   = step_out.pred_original_sample

            with torch.no_grad():
                z_t = step_out.prev_sample

            # Smooth 0 -> 1 ramp instead of a hard switch at the midpoint.
            guidance_weight = max(0.0, min(1.0, (i - ramp_start_idx) / ramp_span))

            if guidance_weight > 0.0:
                z0_hat_g = z0_hat.detach().requires_grad_(True)
                with torch.enable_grad():
                    x_hat_g  = self.decoder(z0_hat_g.flatten(start_dim=1))
                    f_out    = self.classifier(self._to_rgb(x_hat_g))
                    L_cls    = F.binary_cross_entropy(f_out, c_tensor)
                    l1_map   = torch.abs(x_hat_g - x)
                    seg_t    = (S_x * l1_map).sum() / (S_x.sum() + 1e-8)
                    O_hat    = self.obj_det(self._to_rgb(x_hat_g))
                    obj_t    = F.mse_loss(O_hat, O_x.detach())
                    L_pres_g = seg_t + config.ETA_OBJ * obj_t
                    grad_cls,  = torch.autograd.grad(L_cls,    z0_hat_g, retain_graph=True)
                    grad_pres, = torch.autograd.grad(L_pres_g, z0_hat_g)

                with torch.no_grad():
                    action = (
                        - guidance_weight * (config.ALPHA_CLS  * 1.15 * cls_guidance_scale) * grad_cls.to(self.dtype)
                        - guidance_weight * (config.ALPHA_PRES * pres_guidance_scale) * grad_pres.to(self.dtype)
                    )
                    action = action * edit_mask.to(action.dtype)
                    z_t = z_t + action

            # ── background lock (RePaint / Blended Latent Diffusion) ──────────
            # Overwrite everything outside the edit mask with the ORIGINAL
            # latent forward-diffused to the next timestep, so only the lung
            # region is free to change. On the final step the original latent
            # is used clean, pinning the out-of-lung output to x's VAE
            # reconstruction. This is the hard spatial constraint; masked
            # guidance above only steers, it does not lock.
            if lock_background:
                with torch.no_grad():
                    if i < n_steps - 1:
                        t_next   = timesteps[i + 1].reshape(1)
                        noise_bg = torch.randn_like(z0_orig_4d)
                        z_bg     = self.scheduler.add_noise(z0_orig_4d, noise_bg, t_next)
                    else:
                        z_bg = z0_orig_4d
                    z_t = edit_mask * z_t + (1.0 - edit_mask) * z_bg.to(z_t.dtype)

            if i in checkpoint_idx:
                with torch.no_grad():
                    trajectory.append(self.decoder(z_t.flatten(start_dim=1)).detach().cpu())

        with torch.no_grad():
            x_cf = self.decoder(z_t.flatten(start_dim=1))
        return x_cf, trajectory

    # ──────────────────────────────────────────────────────────────────────────
    # Utilities
    # ──────────────────────────────────────────────────────────────────────────

    def vram_estimate_mb(self) -> dict[str, float]:
        def param_mb(m):
            return sum(p.numel() * p.element_size() for p in m.parameters()) / (1024 ** 2)
        return dict(
            vae_mb=param_mb(self.vae),
            unet_mb=param_mb(self.unet),
            cond_proj_mb=param_mb(self.cond_proj),
            total_mb=param_mb(self.vae) + param_mb(self.unet) + param_mb(self.cond_proj),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Factory helpers
# ─────────────────────────────────────────────────────────────────────────────

def build_generator(
    classifier: nn.Module,
    seg_net: nn.Module,
    lung_channel_idx: list[int],
    obj_det: nn.Module,
    model_id: str = config.DIFFUSION_MODEL_ID,
    device: str = config.DEVICE,
    dtype: torch.dtype = config.DTYPE,
) -> PretrainedLatentDiffusionGenerator:
    """Instantiate generator, apply LoRA, upcast trainable params to fp32."""
    generator = PretrainedLatentDiffusionGenerator(
        model_id=model_id,
        device=device,
        dtype=dtype,
        classifier=classifier,
        seg_net=seg_net,
        lung_channel_idx=lung_channel_idx,
        obj_det=obj_det,
    )

    lora_config = LoraConfig(
        r=config.LORA_R,
        lora_alpha=config.LORA_ALPHA,
        lora_dropout=config.LORA_DROPOUT,
        target_modules=config.LORA_TARGETS,
    )
    generator.unet = get_peft_model(generator.unet, lora_config)
    generator.unet.print_trainable_parameters()

    # Upcast trainable params to fp32 for numerical stability
    for p in generator.unet.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    for p in generator.cond_proj.parameters():
        p.requires_grad_(True)
        p.data = p.data.float()

    vram = generator.vram_estimate_mb()
    print(
        f"[generator] VRAM estimate — VAE={vram['vae_mb']:.0f} MB, "
        f"UNet={vram['unet_mb']:.0f} MB, total={vram['total_mb']:.0f} MB"
    )
    return generator