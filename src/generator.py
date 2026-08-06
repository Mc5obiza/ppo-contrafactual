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
from src.gradcam import (
    DegenerateCAMError,
    cached_target_layer,
    ensemble_grad_cam,
    grad_cam,
)


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

        # Episodes where SOME (not all) ensemble members gave a degenerate CAM,
        # so the mask averaged over fewer members than intended. Distinct from
        # the env's _cam_episodes_degenerate, which counts total failures that
        # fall back to the lung-only mask. Both matter when judging whether a
        # run is a clean "+gradcam_mask" arm.
        self._cam_partial_degenerate = 0
        # Episodes where EVERY member was degenerate, so there was no
        # localisation signal at all and the episode ran on the lung-only mask.
        # Counted here (in addition to the env's own counter) so the generator
        # can rate-limit its own warning and still report an accurate total.
        self._cam_full_degenerate = 0
        # Total compute_gradcam_latent_mask calls, i.e. episodes with the mask
        # enabled — the denominator for the two counters above.
        self._cam_calls = 0

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
    # Grad-CAM edit-budget mask (hard action-space constraint)
    # ──────────────────────────────────────────────────────────────────────────

    def compute_gradcam_latent_mask(
        self,
        x: torch.Tensor,
        target_layer_name: str = config.GRADCAM_TARGET_LAYER,
        floor: float      = config.GRADCAM_MASK_FLOOR,
        gamma: float      = config.GRADCAM_MASK_GAMMA,
        blur_sigma: float = config.GRADCAM_MASK_BLUR,
        norm_pct: float   = config.GRADCAM_NORM_PCT,
    ) -> torch.Tensor | None:
        """
        Grad-CAM of the FROZEN classifier on x, reduced to a latent-resolution
        multiplicative edit-budget mask in [floor, 1].

        Returns None if the Grad-CAM is degenerate (spatially constant, i.e.
        no localisation signal — see gradcam.DegenerateCAMError, typically a
        saturated classifier output). None means "no CAM constraint this
        episode", which falls back to the lung-only mask. Callers must COUNT
        these; a run with many degenerate episodes is not a valid
        "+gradcam_mask" arm.

        Passed to generate_counterfactual as `cam_scale`, where it multiplies
        the lung edit mask and therefore bounds the per-location output latent
        deviation (see config.USE_GRADCAM_MASK for the exact bound).

        CACHING CONTRACT — call this ONCE PER EPISODE on the episode's original
        image and reuse the result (see
        MedicalCounterfactualEnv._init_episode_state). One densenet121
        forward+backward is negligible against the ~1250-4500 UNet evaluations
        an episode runs, but recomputing it per PPO step or per refinement pass
        would be pure waste — and would let the mask drift off the original
        image, which defeats the point.

        The classifier is NOT modified: parameters stay frozen and only
        input-side gradients are taken. Note that the CAM is always taken
        w.r.t. the pneumonia output, which is the correct target for BOTH
        directions: on a positive image it highlights the existing opacity
        (where p→n should remove evidence), on a negative image it highlights
        where the classifier's pneumonia detector has support (where n→p should
        add it). Expect a lower-contrast, less focal mask in the n→p direction,
        since attention on a negative image is weaker — this is one reason
        `floor` matters.

        Safe to call from inside a torch.no_grad() block: the Grad-CAM itself
        is wrapped in torch.enable_grad() below. Both the PPO rollout
        (env.step) and generate_via_policy run under no_grad.
        """
        self._cam_calls += 1
        x_in  = self._to_rgb(x.detach().clone()).requires_grad_(True)
        try:
            with torch.enable_grad():
                if hasattr(self.classifier, "members"):
                    # ENSEMBLE: CAM every member against its own forward and
                    # average. Passing the ensemble to resolve_target_layer
                    # instead would silently return members.0's `features` and
                    # log success — a one-member CAM labelled as the ensemble's.
                    # See gradcam.ensemble_grad_cam.
                    cam, degenerate = ensemble_grad_cam(
                        self.classifier, x_in, target_layer_name
                    )
                    if degenerate:
                        self._cam_partial_degenerate += 1
                        # Rate-limited: this fires on essentially every episode
                        # (87/87 in eval job 17665348), and 400 identical lines
                        # bury the progress output in a multi-hour log. The
                        # counter is what matters, and callers report it via
                        # cam_degeneracy_summary() at the end of a run.
                        n = self._cam_partial_degenerate
                        if n <= 3 or n % 50 == 0:
                            print(f"[generator] NOTE: Grad-CAM degenerate for "
                                  f"{len(degenerate)}/{len(self.classifier.members)} "
                                  f"ensemble members {[i for i, _ in degenerate]}; "
                                  f"averaging over the rest. "
                                  f"(occurrence {n}; further ones suppressed)")
                else:
                    layer = cached_target_layer(self.classifier, target_layer_name)
                    cam = grad_cam(self.classifier, layer, x_in)   # (h, w) in [0, 1]
        except DegenerateCAMError as e:
            # Degrade to "no CAM constraint" (caller sees None -> lung-only
            # mask) rather than returning a uniform mask at `floor`, which
            # would look like a working constraint while carrying no
            # localisation signal at all. The caller is responsible for
            # COUNTING these — see MedicalCounterfactualEnv._init_episode_state
            # — because a run where this fires often is not a valid
            # "+gradcam_mask" ablation arm and must not be reported as one.
            self._cam_full_degenerate += 1
            n = self._cam_full_degenerate
            if n <= 3 or n % 50 == 0:
                print(f"[generator] WARNING: Grad-CAM degenerate for this image, "
                      f"falling back to lung-only mask for this episode. {e} "
                      f"(occurrence {n}; further ones suppressed)")
            return None

        cam = cam.detach()[None, None].float()
        cam = F.interpolate(
            cam,
            size=(config.LATENT_SIZE, config.LATENT_SIZE),
            mode="bilinear",
            align_corners=False,
        )
        # Percentile normalisation AFTER upsampling, so the percentile is taken
        # over LATENT_SIZE^2 cells rather than the CAM's handful of native ones.
        hi  = torch.quantile(cam.flatten(), norm_pct).clamp(min=1e-6)
        cam = (cam / hi).clamp(0.0, 1.0)
        if gamma != 1.0:
            cam = cam.pow(gamma)
        cam = self._gaussian_blur(cam, blur_sigma).clamp(0.0, 1.0)
        return (floor + (1.0 - floor) * cam).to(self.dtype)

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
        cam_scale: torch.Tensor | None = None,
        x_anchor: torch.Tensor | None = None,
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
        cam_scale : torch.Tensor | None
            Optional (1, 1, LATENT_SIZE, LATENT_SIZE) per-location edit budget
            in [0, 1] from compute_gradcam_latent_mask, MULTIPLIED into the
            lung edit mask. This is the hard Grad-CAM action-space constraint:
            because the resulting mask gates both the guidance gradient and
            the background-lock composite, and the final step composites
            against the clean anchor latent, the output latent deviation per
            location is bounded by the mask value there. None -> no CAM
            constraint, behaviour identical to before this kwarg existed.
            Must be precomputed and cached per episode by the caller; do NOT
            compute it here, this method runs 2-4x per PPO step.
        x_anchor : torch.Tensor | None
            (1, 1, H, W) image that the background lock, the segmentation /
            CAM mask, and the preservation targets (L1 map, object-detector
            features) are all held against. Defaults to `x`, which reproduces
            the previous behaviour exactly.
            Passing the EPISODE's original image here matters when this method
            is chained (environment.step feeds each pass the previous pass's
            output): with x_anchor=x the lock only bounds drift relative to
            one call ago, so ~100 chained calls per episode accumulate diffuse
            change that no mask constrains. `x` remains the SDEdit
            initialisation either way — only the reference changes.
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
        # `x` is the SDEdit initialisation (the previous pass's output when
        # chained); `x_anchor` is what everything is held AGAINST. Defaulting
        # x_anchor to x keeps behaviour byte-identical to the pre-anchor code.
        anchor_is_init = x_anchor is None or x_anchor is x
        x_anchor       = x if x_anchor is None else x_anchor

        with torch.no_grad():
            z0_orig_flat = self.encoder(x)
            psi_x        = z0_orig_flat.clone()
            cond         = self._build_condition(c_tensor, psi_x)
            z0_orig_4d   = z0_orig_flat.view(
                -1, config.LATENT_CHANNELS, config.LATENT_SIZE, config.LATENT_SIZE
            )
            # Lock reference, kept separate from z0_orig_4d (the trajectory
            # init) so the lock can pin to the episode's original image while
            # the trajectory still starts from the chained pass input.
            z_anchor_4d = (
                z0_orig_4d if anchor_is_init
                else self.encoder(x_anchor).view_as(z0_orig_4d)
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
            S_x = self._get_seg_mask(x_anchor)
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
            # HARD action-space restriction: scale the per-location edit budget
            # by classifier attention. Not a post-hoc penalty — this mask gates
            # the guidance gradient AND the lock composite below, and the final
            # step's composite against the clean anchor makes the output latent
            # deviation per location proportional to the mask value there.
            # Multiplied INTO the lung mask rather than replacing it: the lung
            # mask is anatomical evidence, the CAM is a coarse saliency
            # estimate that leaks onto mediastinum/diaphragm/border, so
            # intersecting can only tighten and preserves the existing
            # out-of-lung guarantee.
            if cam_scale is not None:
                edit_mask = edit_mask * cam_scale.to(edit_mask.dtype)
            O_x = self.obj_det(self._to_rgb(x_anchor))

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
                    l1_map   = torch.abs(x_hat_g - x_anchor)
                    seg_t    = (S_x * l1_map).sum() / (S_x.sum() + 1e-8)
                    O_hat    = self.obj_det(self._to_rgb(x_hat_g))
                    obj_t    = F.mse_loss(O_hat, O_x.detach())
                    L_pres_g = seg_t + config.ETA_OBJ * obj_t
                    # TWO backwards, deliberately. This is the single most
                    # expensive thing in the pipeline — it runs on 60-80% of the
                    # DDIM steps of every refinement pass of every PPO step, and
                    # retain_graph=True means the VAE decoder is walked backward
                    # twice per guided step.
                    #
                    # It LOOKS trivially fusable: `action` below is a fixed
                    # linear combination of these two gradients whose
                    # coefficients are plain Python floats known before the
                    # backward, so differentiating (w_cls*L_cls + w_pres*L_pres)
                    # once is the same vector in exact arithmetic, and saves one
                    # full decoder backward (~1.5x on the whole eval).
                    #
                    # It is NOT the same vector in fp16, which is what this
                    # actually runs in. z0_hat_g is fp16 (z_t is fp16 all the
                    # way from the encoder), so the decoder backward is fp16
                    # too. Folding the weights into the loss scales the whole
                    # backward chain BEFORE those fp16 intermediates instead of
                    # after — and with ALPHA_PRES=0.05 the preservation branch
                    # loses several bits of headroom against fp16's subnormal
                    # floor. Measured against the real SD1.5 VAE decoder + the
                    # rsna DenseNet on realistic encoded latents: 0.6%-4.7% max
                    # relative deviation in `action` across cls/pres scale
                    # settings, versus an fp16 epsilon of 0.1%. That is a real
                    # trajectory change, not rounding.
                    #
                    # So: do not fuse these without either moving the guidance
                    # block to fp32 or adding a gradient-scaling factor, and
                    # re-validating CV/FID against a previous run either way.
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
                        noise_bg = torch.randn_like(z_anchor_4d)
                        z_bg     = self.scheduler.add_noise(z_anchor_4d, noise_bg, t_next)
                    else:
                        z_bg = z_anchor_4d
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

    def cam_degeneracy_summary(self) -> str:
        """
        One-line summary of how often the Grad-CAM edit-budget mask actually
        carried a localisation signal, for printing at the end of a run.

        The per-episode prints are rate-limited (they fired on 87/87 episodes in
        eval job 17665348 and buried the log), so this is the accurate record.
        A high rate here means the run is NOT a valid "+gradcam_mask" arm — see
        compute_gradcam_latent_mask's docstring.
        """
        if self._cam_calls == 0:
            return "[generator] Grad-CAM mask: not used (USE_GRADCAM_MASK off)"
        pct = lambda n: 100.0 * n / self._cam_calls
        return (
            f"[generator] Grad-CAM mask over {self._cam_calls} episodes: "
            f"{self._cam_partial_degenerate} partial "
            f"({pct(self._cam_partial_degenerate):.1f}%), "
            f"{self._cam_full_degenerate} total-degenerate -> lung-only mask "
            f"({pct(self._cam_full_degenerate):.1f}%)"
        )

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