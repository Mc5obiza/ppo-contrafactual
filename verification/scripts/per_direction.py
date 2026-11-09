"""
Per-direction generator adapter for the eval code at 49e6050.

The committed eval loads ONE generator for both directions:
_build_pipeline -> pipeline.load_generator_weights(ctx.generator, CKPT_DIR),
which takes the first of CKPT_DIR/{n_to_p,p_to_n}/ holding unet_finetuned.pt +
cond_proj.pt. This module makes each direction generate with its own adapter
by reloading the generator, through that same loader, right before the
direction generates:

    ADAPTER_NP_VIEW  view whose n_to_p/ holds the n->p adapter
                     (loader resolves n_to_p/)
    ADAPTER_PN_VIEW  view whose n_to_p/ has NO unet, so the loader resolves
                     p_to_n/ (the p->n adapter)
    ADAPTER_EXPECT_NP_SHA / ADAPTER_EXPECT_PN_SHA   optional full sha256 of the
                     unet file each direction must load; abort otherwise

Every policy env holds the same ctx.generator object, and the generator keeps
no state derived from the adapter (cond_proj is applied per call), so
reloading the weights in place switches the adapter for everything that
generates afterwards.

After each load it prints one line per direction, e.g.
    [adapter] p_to_n <- /…/checkpoints/p_to_n/unet_finetuned.pt sha256=6ae44230… |
              generator == file: unet LoRA 256/256, cond_proj 4/4
and raises if the weights now in the generator are not exactly the file's.
"""
import hashlib
import os

import torch

import pipeline

LABELS = {1.0: "n_to_p", 0.0: "p_to_n"}
_sha_cache: dict = {}


def enabled() -> bool:
    return bool(os.environ.get("ADAPTER_PN_VIEW"))


def _sha256(path: str) -> str:
    if path not in _sha_cache:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 24), b""):
                h.update(chunk)
        _sha_cache[path] = h.hexdigest()
    return _sha_cache[path]


def _n_equal(module, file_path: str, only_lora: bool) -> tuple[int, int]:
    want = torch.load(file_path, map_location="cpu", weights_only=True, mmap=True)
    have = module.state_dict()
    keys = [k for k in want if ("lora" in k.lower() or not only_lora)]
    same = sum(int(k in have and torch.equal(have[k].detach().cpu(), want[k].to(have[k].dtype)))
               for k in keys)
    return same, len(keys)


def use(generator, label: str) -> str:
    view = os.environ["ADAPTER_NP_VIEW" if label == "n_to_p" else "ADAPTER_PN_VIEW"]
    path = pipeline.load_generator_weights(generator, ckpt_dir=view)
    got = os.path.basename(os.path.dirname(path))
    if got != label:
        raise RuntimeError(f"[adapter] {label}: loader resolved '{path}' ({got}/), not {label}/")
    unet = os.path.realpath(path)
    cond = os.path.realpath(os.path.join(os.path.dirname(path), "cond_proj.pt"))
    sha = _sha256(unet)
    expect = os.environ.get(f"ADAPTER_EXPECT_{'NP' if label == 'n_to_p' else 'PN'}_SHA")
    lu, nu = _n_equal(generator.unet, unet, only_lora=True)
    lc, nc = _n_equal(generator.cond_proj, cond, only_lora=False)
    print(f"[adapter] {label} <- {unet} sha256={sha} | cond_proj <- {cond} | "
          f"generator == file: unet LoRA {lu}/{nu}, cond_proj {lc}/{nc}"
          + (f" | expected sha256 {'MATCH' if sha == expect else 'MISMATCH'}" if expect else ""),
          flush=True)
    if lu != nu or nu == 0 or lc != nc:
        raise RuntimeError(f"[adapter] {label}: generator does not hold the weights of {unet}")
    if expect and sha != expect:
        raise RuntimeError(f"[adapter] {label}: loaded {sha}, expected {expect}")
    return unet


def install_eval_main(eval_main) -> None:
    """Switch the adapter before every shard direction and every diff plot."""
    import src.diff_viz as dv

    orig_gen = eval_main._generate_shard

    def _generate_shard(x_shard, policy, target_c, ctx, *a, **kw):
        use(ctx.generator, LABELS[float(target_c)])
        return orig_gen(x_shard, policy, target_c, ctx, *a, **kw)

    orig_plot = dv.generate_and_plot_diff

    def generate_and_plot_diff(*a, **kw):
        use(kw["generator"], LABELS[float(kw["target_c"])])
        return orig_plot(*a, **kw)

    eval_main._generate_shard = _generate_shard
    dv.generate_and_plot_diff = generate_and_plot_diff
    print(f"[adapter] per-direction mode: n_to_p <- {os.environ['ADAPTER_NP_VIEW']}, "
          f"p_to_n <- {os.environ['ADAPTER_PN_VIEW']}", flush=True)
