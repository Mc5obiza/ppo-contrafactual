"""
src/gradcam.py
==============
Single shared Grad-CAM implementation for the frozen classifier.

Extracted verbatim (in logic) from check_classifier_localization.py so that
the Grad-CAM used to *constrain* the PPO action space
(PretrainedLatentDiffusionGenerator.compute_gradcam_latent_mask) is the exact
same code path as the Grad-CAM used to *validate* classifier localization in
the diagnostic script. Two subtly-different implementations would mean the
diagnostic no longer tells you anything about the mask.

Both hard-won fixes from the diagnostic are preserved here — do not
"simplify" either of them:

  1. find_target_layer() prefers a submodule literally named "features".
     For DenseNet, a single internal dense-layer Conv2d only contributes a
     narrow slice (e.g. 32 of ~1024) of the channels that actually reach the
     classifier, because DenseNet concatenates every dense layer's output
     onto a growing stack. Hooking one internal conv produces a near-uniform
     map regardless of prediction confidence.

  2. grad_cam() uses a forward hook + retain_grad(), NOT
     register_full_backward_hook. TorchXRayVision's DenseNet.forward runs
     F.relu(features, inplace=True) on the exact tensor a backward hook
     would wrap, which PyTorch forbids. retain_grad() has no
     custom-autograd-Function wrapping, so the downstream in-place op is
     handled by ordinary autograd versioning.

The only behavioural change vs. the original: grad_cam returns a torch
Tensor rather than a numpy array (callers wanting numpy call
.cpu().numpy()), since the generator consumes it as a tensor on-device.
"""

import torch


class DegenerateCAMError(RuntimeError):
    """
    Raised when Grad-CAM produces a spatially CONSTANT map, which carries no
    localisation information at all.

    WHY THIS NEEDS ITS OWN EXCEPTION: a constant CAM is not a loud failure. Fed
    through compute_gradcam_latent_mask's `floor + (1 - floor) * cam`, an
    all-zero CAM becomes a UNIFORM mask at `floor` — every location gets an
    identical edit budget, so the locality constraint is silently disabled
    while the run completes normally and the ablation arm still looks valid.
    That is the single most damaging way this can fail, so it is detected
    explicitly rather than left to produce quietly meaningless results.

    The realistic cause is a saturated output: the classifier returns a
    probability, and once it rounds to exactly 1.0 (or 0.0) in fp32 — logit
    magnitude beyond ~17 — d(prob)/d(activations) is IDENTICALLY zero, so the
    CAM is all zeros.

    Note that backpropagating logit(p) instead of p does NOT fix this. The
    p*(1-p) factor is a spatial constant and cancels under CAM normalisation
    anyway, and once p has rounded to 1.0 the gradient information is already
    destroyed — logit(clamp(p)) differentiates a constant and also yields
    exactly zero. Verified, not assumed. Recovering it would require the
    pre-sigmoid logit, i.e. hooking the classifier's final Linear.
    """


def find_target_layer(model: torch.nn.Module) -> torch.nn.Module:
    """
    Find the best Grad-CAM hook target.

    Prefers a submodule named "features" (the full concatenated feature stack
    right before global pooling + classifier — the standard DenseNet Grad-CAM
    target). Falls back to the last spatial Conv2d heuristic for
    architectures with no "features" submodule (e.g. ResNet-18, where
    layer4's conv works fine).
    """
    named = dict(model.named_modules())
    for name, m in named.items():
        if name.split(".")[-1] == "features" and isinstance(m, torch.nn.Module):
            print(f"[gradcam] found dedicated feature-extractor submodule: '{name}' "
                  f"— hooking its full output (standard DenseNet Grad-CAM target)")
            return m

    print("[gradcam] no 'features' submodule found — falling back to last "
          "spatial Conv2d heuristic (expected path for ResNet-18)")
    last_spatial_conv = None
    last_any_conv     = None
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            last_any_conv = m
            if m.kernel_size != (1, 1):
                last_spatial_conv = m

    if last_spatial_conv is not None:
        return last_spatial_conv
    if last_any_conv is not None:
        print("[gradcam] WARNING: only found 1x1 conv layers — Grad-CAM will "
              "have poor/no spatial resolution.")
        return last_any_conv
    raise RuntimeError("No usable layer found for Grad-CAM.")


def resolve_target_layer(model: torch.nn.Module, name: str) -> torch.nn.Module:
    """
    Resolve config.GRADCAM_TARGET_LAYER to an actual module.

    Parameters
    ----------
    model : the frozen classifier (e.g. XRVPneumoniaClassifier, which wraps
            the xrv DenseNet as ``.model`` — so named_modules() keys look
            like "model.features.denseblock3").
    name  : either "features" (delegates to find_target_layer's auto-detect,
            the proven default, 7x7 output at 224 input) or a dotted module
            path matched as a SUFFIX of a named_modules() key, e.g.
            "features.denseblock3" (14x14 output — finer localisation).

    Spatial resolution for the xrv DenseNet-121 at 224 input:
        features.denseblock2 -> 28x28   (~9px cells at IMG_SIZE=256)
        features.denseblock3 -> 14x14   (~18px cells)
        features             ->  7x7    (~37px cells)

    Note: hooking an internal denseblock means the *next* module
    (transition3's ReLU) will in-place-modify the captured tensor. That is
    the same situation fix #2 above already handles, so it works — but if it
    ever raises, "features" is the safe fallback.
    """
    if name == "features":
        return find_target_layer(model)

    named = dict(model.named_modules())
    if name in named:
        print(f"[gradcam] hooking requested layer '{name}'")
        return named[name]

    matches = [k for k in named if k == name or k.endswith("." + name)]
    if len(matches) == 1:
        print(f"[gradcam] hooking requested layer '{name}' (resolved to '{matches[0]}')")
        return named[matches[0]]
    if len(matches) > 1:
        raise ValueError(
            f"GRADCAM_TARGET_LAYER='{name}' is ambiguous — matches {matches}. "
            f"Use a fully-qualified name."
        )

    spatial_hint = [k for k in named if "denseblock" in k or "layer" in k]
    raise ValueError(
        f"GRADCAM_TARGET_LAYER='{name}' not found in the classifier. "
        f"Use 'features' for the default target, or one of these spatial "
        f"blocks: {spatial_hint[:12]}{' ...' if len(spatial_hint) > 12 else ''}"
    )


def cached_target_layer(model: torch.nn.Module, name: str) -> torch.nn.Module:
    """
    resolve_target_layer(), memoised on the model instance.

    WHY: resolve_target_layer / find_target_layer print a line every call, and
    compute_gradcam_latent_mask runs once per PPO episode — so an 800-episode
    run emitted 800 identical "[gradcam] found dedicated feature-extractor
    submodule" lines per model. With an ensemble that doubles. The resolution
    is a pure function of (model, name), so cache it and log once.

    The cache is a plain dict attribute: nn.Module.__setattr__ only intercepts
    Parameters/Modules/Tensors, so this does not get registered as state.
    """
    cache = getattr(model, "_gradcam_layer_cache", None)
    if cache is None:
        cache = {}
        model._gradcam_layer_cache = cache
    if name not in cache:
        cache[name] = resolve_target_layer(model, name)
    return cache[name]


def ensemble_grad_cam(
    ensemble: torch.nn.Module,
    x: torch.Tensor,
    target_layer_name: str,
) -> tuple[torch.Tensor, list[tuple[int, str]]]:
    """
    Mean Grad-CAM across the members of an EnsemblePneumoniaClassifier.

    WHY THIS IS NEEDED
    ------------------
    Passing the ensemble straight to find_target_layer does NOT fail — it walks
    named_modules() and returns the first submodule named "features", which on
    an ensemble is `members.0.model.features`. It then prints its usual success
    message. The result is a single-member CAM silently presented as the
    ensemble's, and member 0 is rsna: the 2-head single-corpus model most prone
    to the very shortcut the CAM mask exists to constrain. Verified behaviour,
    not a hypothetical — see the plan's Phase 2a.

    Each member is CAM'd against its OWN forward pass and its own `features`
    layer, then the maps are averaged. Averaging happens AFTER grad_cam's
    per-map [0,1] normalisation, so a member with larger raw activation
    magnitudes cannot dominate the average.

    Degenerate members (saturated output -> spatially constant CAM, see
    DegenerateCAMError) are SKIPPED rather than allowed to contribute a
    zero map that would drag the mean toward uniform. Only if EVERY member is
    degenerate does this raise — which the caller already handles by falling
    back to the lung-only mask.

    Returns
    -------
    (cam, degenerate) : cam is (h, w) in [0, 1] on x's device;
                        degenerate is a list of (member_index, reason) for
                        members that were skipped, so callers can report the
                        rate rather than hiding it.

    Must be called with grad enabled (see grad_cam).
    """
    cams: list[torch.Tensor] = []
    degenerate: list[tuple[int, str]] = []

    for i, member in enumerate(ensemble.members):
        layer = cached_target_layer(member, target_layer_name)
        try:
            cams.append(grad_cam(member, layer, x))
        except DegenerateCAMError as e:
            degenerate.append((i, str(e).split(".")[0]))

    if not cams:
        raise DegenerateCAMError(
            f"Grad-CAM was degenerate for ALL {len(ensemble.members)} ensemble "
            f"members, so there is no localisation signal to average. Reasons: "
            f"{[r for _i, r in degenerate]}"
        )

    # Members are all densenet121-res224-* so their `features` maps are the same
    # size, but do not assume it — a mismatched member would otherwise fail deep
    # inside torch.stack with an unhelpful message.
    ref_shape = cams[0].shape
    if any(c.shape != ref_shape for c in cams):
        cams = [
            c if c.shape == ref_shape else
            torch.nn.functional.interpolate(
                c[None, None], size=ref_shape, mode="bilinear", align_corners=False
            )[0, 0]
            for c in cams
        ]

    return torch.stack(cams, dim=0).mean(dim=0), degenerate


def grad_cam(
    model: torch.nn.Module,
    target_layer: torch.nn.Module,
    x: torch.Tensor,
) -> torch.Tensor:
    """
    Parameters
    ----------
    model        : classifier, already in eval mode
    target_layer : module whose output to hook (see resolve_target_layer)
    x            : (1, C, H, W) — the exact input the model expects, with
                   requires_grad=True

    Returns
    -------
    torch.Tensor (h, w) heatmap normalised to [0, 1], detached, on x's device.

    Must be called with grad enabled. Callers inside a torch.no_grad() block
    (e.g. the PPO rollout) are responsible for wrapping this in
    torch.enable_grad() — see compute_gradcam_latent_mask.
    """
    activations = {}

    def fwd_hook(module, inp, out):
        out.retain_grad()
        activations["value"] = out

    h1 = target_layer.register_forward_hook(fwd_hook)
    try:
        model.zero_grad(set_to_none=True)
        out = model(x)
        out.sum().backward()
    finally:
        h1.remove()

    act_tensor = activations.get("value")
    if act_tensor is None:
        raise RuntimeError(
            "forward hook never fired — target_layer is not used in this "
            "model's forward pass for this input."
        )
    if act_tensor.grad is None:
        raise RuntimeError(
            "activations.grad is None after backward() — the hooked layer's "
            "output may not be part of the path from input to the model's "
            "final output. Check that target_layer is actually used in the "
            "forward pass for this input."
        )

    act  = act_tensor[0]         # (C, h, w) — reflects the value AFTER any
                                 # downstream in-place ops (e.g. the ReLU),
                                 # which is standard/expected for Grad-CAM
    grad = act_tensor.grad[0]    # (C, h, w)
    weights = grad.mean(dim=(1, 2))                              # (C,)
    cam = torch.relu((weights[:, None, None] * act).sum(dim=0))  # (h, w)

    cam = cam.detach().float()
    if not torch.isfinite(cam).all():
        raise DegenerateCAMError("Grad-CAM produced non-finite values.")

    # Detect degeneracy BEFORE normalising — normalisation is exactly what
    # turns a constant map into a silent all-zeros mask. See DegenerateCAMError.
    cam = cam - cam.min()
    cam_max = cam.max()
    if cam_max <= 0:
        raise DegenerateCAMError(
            f"Grad-CAM is spatially constant (post-ReLU span = {float(cam_max):.3e}), "
            f"so it carries no localisation signal. Most likely the classifier's "
            f"output has saturated to exactly 0.0 or 1.0, making "
            f"d(output)/d(activations) identically zero. Check f(x) for this image: "
            f"if it is 1.0 or 0.0 to full fp32 precision, that is the cause. "
            f"Backpropagating logit(p) does NOT help (see DegenerateCAMError); "
            f"recovering the gradient needs the pre-sigmoid logit."
        )
    return cam / cam_max
