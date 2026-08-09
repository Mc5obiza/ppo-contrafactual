"""
pipeline.py
===========
Shared construction code for the two pipeline entry points:

    main.py       — Steps 1-5: ingestion -> pretrained models -> generator
                    -> LoRA fine-tuning -> PPO training -> checkpoints
    eval_main.py  — Steps 6-6B: reload those checkpoints -> metric suite
                    -> counterfactual diff plots

WHY THE SPLIT
-------------
main.py previously ran training and evaluation in a single process. PPO
training costs ~12-20h on the GPUs this job actually lands on (Quadro
RTX 5000 / Tesla P100 on `publicgpu`) and evaluation is a further ~2.5-5h,
so one job that walltimes out during Step 6 throws the finished policies
away with it.

That is not hypothetical — it is what happened to job 17644171: both
directional policies were written to disk (22:21 and 11:05 respectively),
then the evaluation phase was killed at the 24h boundary having produced
nothing. The 20h of PPO training was recoverable in principle but there
was no entry point that could reload it.

Splitting the entry points means evaluation reloads saved policies rather
than depending on the training process still being alive, and can be
re-run — different n_eval, regenerated plots, metric fixes — without ever
repeating a PPO run.

WHAT LIVES HERE
---------------
Only the construction/loading that BOTH entry points need. Training logic
stays in main.py; evaluation logic stays in eval_main.py.
"""

import os
from dataclasses import dataclass

import torch

import config
import domains
from ingestion import build_dataset          # re-exported: smoke tests patch this
from src.models import build_segmentation_net, build_object_detector
from src.generator import build_generator
from src.xrv_classifier import build_ensemble_classifier, build_xrv_classifier
from factory.environment import MedicalCounterfactualEnv
from factory.ppo import PPOActorCritic
from src.inference import DirectionalPolicy


# ─────────────────────────────────────────────────────────────────────────────
# Shared context
# ─────────────────────────────────────────────────────────────────────────────

# Set ONLY by the smoke tests, via use_dataset_override(). None in every real
# run. Module-level rather than a build_common() argument so smoke_test.py can
# install it without every intermediate caller having to forward a flag.
_DATASET_OVERRIDE = None


def use_dataset_override(factory) -> None:
    """
    Install a (img_size) -> Dataset factory that build_common() will use
    INSTEAD of the active domain's.

    For smoke tests only: they need a handful of images, and DomainProfile is
    frozen so its build_dataset field cannot be patched in place. Pass None to
    clear.

    Works for every domain, which the previous `pipeline.build_dataset`
    monkeypatch did not — see the comment in build_common().
    """
    global _DATASET_OVERRIDE
    _DATASET_OVERRIDE = factory


@dataclass
class PipelineContext:
    """
    Everything Steps 1-3 build, bundled so the two entry points construct
    it identically. Deliberately a plain container — no behaviour — so it
    stays obvious that train and eval start from the same objects.
    """
    dataset: object
    classifier: object
    seg_net: object
    lung_channel_idx: list
    obj_det: object
    generator: object
    # The DomainProfile everything above was built from. Carried so downstream
    # consumers (metrics, held-out validation) read the domain's bindings
    # instead of re-deriving them from config and drifting apart.
    domain: object = None
    # Lesion segmenter for the clinical metrics, or None when the domain has
    # no such thing (both chest domains). Built here rather than inside
    # run_full_evaluation so training and evaluation cannot end up with
    # different ones, and so a missing checkpoint fails at construction.
    #
    # NOT passed to the generator or the env — it must stay outside the
    # guidance loop, or the metrics computed from it become circular. See
    # DomainProfile.build_lesion_net.
    lesion_net: object = None


def build_common() -> PipelineContext:
    """
    Steps 1-3, shared verbatim by training and evaluation.

    Note the generator returned here has FRESHLY INITIALISED LoRA adapters.
    - main.py trains them in Step 4 (run_lora_finetuning, which resumes
      from checkpoints/finetuning_latest.pt if present).
    - eval_main.py must call load_generator_weights() to restore them.
    """
    print("=" * 70)
    print("STEP 1 — Dataset ingestion")
    print("=" * 70)
    domain = domains.active()
    print(domain.summary())
    if domain.notes:
        print(f"[domain] {domain.notes}")

    # Dataset construction, in override-first order.
    #
    # The smoke tests need to shrink the dataset, and DomainProfile is a frozen
    # dataclass so its build_dataset field cannot be patched. The old route was
    # for smoke_test.py to patch the `pipeline.build_dataset` module global —
    # but that global is only consulted on the rsna_pneumonia branch below, so
    # patching it did NOTHING for any other domain, and a smoke test on
    # node21_nodule or cbis_ddsm_breast silently loaded the FULL dataset. That
    # defeats the point of a smoke test twice over: it is slow, and it fails
    # for lack of data rather than fast for a shape error.
    #
    # _DATASET_OVERRIDE fixes it for every domain at once. See
    # use_dataset_override() below.
    if _DATASET_OVERRIDE is not None:
        print("[pipeline] using the test dataset override "
              "(smoke test — NOT a real run)")
        dataset = _DATASET_OVERRIDE(config.IMG_SIZE)
    elif domain.name == "rsna_pneumonia":
        # Kept so the baseline stays byte-identical: the rsna_pneumonia
        # profile's build_dataset IS ingestion.build_dataset, and any existing
        # caller patching the module global still works.
        dataset = build_dataset(img_size=config.IMG_SIZE)
    else:
        dataset = domain.build_dataset(config.IMG_SIZE)

    print("\n" + "=" * 70)
    print("STEP 2 — Loading pretrained models")
    print("=" * 70)
    # f(·) is either an ENSEMBLE (config.GUIDANCE_ENSEMBLE_WEIGHTS) or the
    # single v1 model, per config.USE_GUIDANCE_ENSEMBLE. Whichever it is, this
    # ONE object flows to the generator (guidance gradient), the env
    # (termination + reward), the far-pool scan, the Xn/Xp split and CV — so the
    # choice changes all of them at once. See
    # src/xrv_classifier.EnsemblePneumoniaClassifier for why the ensemble exists
    # and what its CV numbers are (not) comparable to.
    #
    # A domain may bring its OWN classifier backend instead (build_guidance).
    # cbis_ddsm_breast does: torchxrayvision has no breast head, so f(·) there
    # is a set of backbones we trained. The branch is on the profile, not on
    # the domain name, so adding another non-chest domain needs no edit here.
    if domain.build_guidance is not None:
        classifier = domain.build_guidance()
    elif config.USE_GUIDANCE_ENSEMBLE:
        classifier = build_ensemble_classifier(
            weights_list=list(domain.ensemble_weights), pathology=domain.pathology
        )
    else:
        classifier = build_xrv_classifier(
            weights=domain.guidance_weights, pathology=domain.pathology
        )

    # Same story for S: the xrv PSPNet segments lungs, which a mammogram does
    # not have. A replacement must match its call contract — see
    # src/breast_segmentation.BreastAnatomySegmenter.
    if domain.build_masker is not None:
        seg_net, lung_channel_idx = domain.build_masker()
    else:
        seg_net, lung_channel_idx = build_segmentation_net()

    obj_det = build_object_detector()

    # Optional, and deliberately built AFTER the guidance objects so it is
    # obvious it is not one of them: the lesion segmenter measures the
    # counterfactuals, it never steers them.
    lesion_net = domain.build_lesion_net() if domain.build_lesion_net else None

    print("\n" + "=" * 70)
    print("STEP 3 — Building generator with LoRA")
    print("=" * 70)
    generator = build_generator(
        classifier=classifier,
        seg_net=seg_net,
        lung_channel_idx=lung_channel_idx,
        obj_det=obj_det,
    )

    return PipelineContext(
        dataset=dataset,
        classifier=classifier,
        seg_net=seg_net,
        lung_channel_idx=lung_channel_idx,
        obj_det=obj_det,
        generator=generator,
        domain=domain,
        lesion_net=lesion_net,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint loading (evaluation side)
# ─────────────────────────────────────────────────────────────────────────────

# Hashing two 1.7 GB files off a network filesystem is not something to repeat:
# load_generator_weights asks once per generator it loads and _build_pipeline
# asks again to decide whether a second generator is needed. Keyed by ckpt_dir.
_GENERATOR_DIGESTS: dict[str, tuple[str | None, str | None]] = {}


def _generator_digests(ckpt_dir: str) -> tuple[str | None, str | None]:
    """
    (n_to_p, p_to_n) content hashes of the saved generators, None where absent.

    Every direction's UNet has the same architecture and therefore the same
    file size, so size cannot distinguish them and the bytes have to be read.
    """
    if ckpt_dir in _GENERATOR_DIGESTS:
        return _GENERATOR_DIGESTS[ckpt_dir]

    import hashlib

    digests: list[str | None] = []
    for label in ("n_to_p", "p_to_n"):
        path = os.path.join(ckpt_dir, label, "unet_finetuned.pt")
        if not os.path.exists(path):
            digests.append(None)
            continue
        h = hashlib.blake2b(digest_size=16)
        with open(path, "rb") as fh:
            while chunk := fh.read(1 << 24):
                h.update(chunk)
        digests.append(h.hexdigest())

    result = (digests[0], digests[1])
    _GENERATOR_DIGESTS[ckpt_dir] = result
    return result


def generators_differ(ckpt_dir: str = config.CKPT_DIR) -> bool:
    """
    True when the two directions hold DIFFERENT fine-tuned generators.

    They are supposed to be identical copies of one shared LoRA generator.
    They stop being that as soon as the two run_ppo.slurm array tasks both run
    Step 4 — neither finds a checkpoint, each trains its own, and nothing in
    the PPO path is seeded to make them agree. eval_main._build_pipeline uses
    this to decide whether one generator is enough or each policy needs its
    own.

    False when either file is missing: nothing to contradict yet.
    """
    a, b = _generator_digests(ckpt_dir)
    return a is not None and b is not None and a != b


def _warn_if_generators_differ(ckpt_dir: str) -> None:
    """Say so, loudly, when the two directions disagree — see generators_differ."""
    if not generators_differ(ckpt_dir):
        return
    a, b = _generator_digests(ckpt_dir)
    print(
        f"\n⚠ [pipeline] n_to_p/ and p_to_n/ hold DIFFERENT fine-tuned "
        f"generators\n"
        f"    n_to_p: {a}\n"
        f"    p_to_n: {b}\n"
        f"  They are meant to be identical copies of one shared generator; two "
        f"run_ppo.slurm\n"
        f"  array tasks that both ran Step 4 produce this. Each policy is "
        f"paired with its own\n"
        f"  generator below, so the metrics stay honest — but the two "
        f"directions are then no\n"
        f"  longer sharing a generator, which is worth knowing when you "
        f"compare them.\n"
    )


def load_generator_weights(
    generator,
    ckpt_dir: str = config.CKPT_DIR,
    prefer_label: str = "n_to_p",
) -> str:
    """
    Restore the fine-tuned UNet + cond_proj weights into a freshly built
    generator.

    Prefers the full `unet_finetuned.pt` that save_checkpoints() writes,
    because that is the artefact main.py explicitly declares as "the
    trained generator". It is 1.7 GB and takes ~30s to load; the smaller
    PEFT-format `finetuning_latest.pt` (22 MB) holds the same LoRA weights
    and is used as a fallback.

    Those two ARE equivalent today — PPOTrainer optimises only
    policy.parameters(), so the generator is unchanged between the end of
    Step 4 and the end of Step 5. Preferring the post-training artefact
    means that stays true by construction if anyone later makes PPO touch
    the generator, instead of silently loading stale adapters.

    `prefer_label` decides which direction's copy wins. When the two are the
    identical copies they are meant to be, it changes nothing. When they are
    not — see _warn_if_generators_differ — it at least makes the choice
    explicit and reproducible instead of "whichever directory got written
    first".

    Returns
    -------
    str : the path actually loaded from, for logging.
    """
    _warn_if_generators_differ(ckpt_dir)

    others = [l for l in ("n_to_p", "p_to_n") if l != prefer_label]
    for label in [prefer_label, *others]:
        unet_path = os.path.join(ckpt_dir, label, "unet_finetuned.pt")
        cond_path = os.path.join(ckpt_dir, label, "cond_proj.pt")
        if os.path.exists(unet_path) and os.path.exists(cond_path):
            print(f"[pipeline] loading fine-tuned generator from '{ckpt_dir}/{label}/'")
            try:
                generator.unet.load_state_dict(
                    torch.load(unet_path, map_location=config.DEVICE)
                )
            except RuntimeError as err:
                # Overwhelmingly the cause is a LoRA geometry change: the
                # adapter A/B matrices are shaped (r, in) / (out, r), so a
                # LORA_R or LORA_TARGETS edit makes every previously saved
                # generator unloadable. Raw torch shape errors bury that.
                raise RuntimeError(
                    f"[pipeline] failed to load '{unet_path}' into the generator "
                    f"this config builds (LORA_R={config.LORA_R}, "
                    f"LORA_TARGETS={config.LORA_TARGETS}).\n"
                    f"This checkpoint was almost certainly written under a "
                    f"different LoRA rank or target set — those are baked into "
                    f"the adapter tensor shapes, so the two are not "
                    f"interchangeable.\n"
                    f"Either restore the config that produced it (see "
                    f"'{ckpt_dir}/archive_v1_rank8/README.md' for the rank-8 "
                    f"settings) or re-run the LoRA fine-tuning phase to produce "
                    f"a matching generator.\nOriginal error:\n{err}"
                ) from err
            generator.cond_proj.load_state_dict(
                torch.load(cond_path, map_location=config.DEVICE)
            )
            print(f"[pipeline]   → unet_finetuned.pt + cond_proj.pt")
            return unet_path

    # Fallback: the per-epoch LoRA checkpoint written by run_lora_finetuning.
    from peft import set_peft_model_state_dict

    ft_path = os.path.join(ckpt_dir, "finetuning_latest.pt")
    if os.path.exists(ft_path):
        print(f"[pipeline] no unet_finetuned.pt under {ckpt_dir}/{{n_to_p,p_to_n}}/ — "
              f"falling back to the LoRA checkpoint '{ft_path}'")
        ckpt = torch.load(ft_path, map_location=config.DEVICE)
        set_peft_model_state_dict(generator.unet, ckpt["lora_state_dict"])
        generator.cond_proj.load_state_dict(ckpt["cond_proj_state_dict"])
        print(f"[pipeline]   → resumed LoRA adapters from epoch {ckpt['epoch']}")
        return ft_path

    raise FileNotFoundError(
        f"No fine-tuned generator weights found. Looked for "
        f"'{ckpt_dir}/n_to_p/unet_finetuned.pt', "
        f"'{ckpt_dir}/p_to_n/unet_finetuned.pt', and '{ft_path}'. "
        f"Run the training pipeline (sbatch run_ppo.slurm) first."
    )


def load_directional_policy(
    target_c: float,
    label: str,
    ctx: PipelineContext,
    ckpt_dir: str = config.CKPT_DIR,
    generator=None,
) -> DirectionalPolicy:
    """
    Rebuild one direction's environment and reload its trained policy from
    ckpt_dir/<label>/ppo_policy.pt.

    Mirrors _train_directional_policy() in main.py so the env the policy is
    evaluated in matches the one it was trained in. In particular the env's
    far-pool holdout is a deterministic slice (_far_indices[:n_holdout], see
    MedicalCounterfactualEnv.__init__), so this reproduces the same split
    training used rather than resampling it.

    obs_dim is read off a live env.reset() rather than hardcoded, exactly as
    main.py does — a shape change then fails loudly at load_state_dict
    instead of silently mis-loading.

    Parameters
    ----------
    target_c : 1.0 for n→p, 0.0 for p→n
    label    : "n_to_p" / "p_to_n" — also the checkpoint subdirectory name
    generator : the generator this direction's policy was trained against.
             Defaults to ctx.generator, which is correct whenever both
             directions share one Step 4 generator. They do NOT share one when
             the two run_ppo.slurm array tasks each ran Step 4 (each writes
             CKPT_DIR/<label>/finetuning_latest.pt now, see main.py), and a
             policy driven by the wrong generator is being scored on dynamics
             it never saw. Every PPO-guided path in evaluation reaches the
             generator through this env, so passing it here is what makes the
             pairing hold end to end.
    """
    print("\n" + "-" * 70)
    print(f"Loading trained policy  [{label}, target_c={target_c}]")
    print("-" * 70)

    ckpt_path = os.path.join(ckpt_dir, label, "ppo_policy.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"No trained policy at '{ckpt_path}'. Run the training pipeline "
            f"(sbatch run_ppo.slurm) before evaluating."
        )

    env = MedicalCounterfactualEnv(
        dataset=ctx.dataset,
        classifier=ctx.classifier,
        target_c=target_c,
        generator=ctx.generator if generator is None else generator,
        seg_net=ctx.seg_net,
        obj_det=ctx.obj_det,
        budget_T=config.BUDGET_T,
        device=config.DEVICE,
    )

    obs     = env.reset()
    obs_dim = obs.shape[0]
    print(f"[{label}] obs_dim = {obs_dim}")

    policy = PPOActorCritic(obs_dim=obs_dim, action_dim=config.ACTION_DIM).to(config.DEVICE)
    state  = torch.load(ckpt_path, map_location=config.DEVICE)
    try:
        policy.load_state_dict(state)
    except RuntimeError as err:
        raise RuntimeError(
            f"[{label}] '{ckpt_path}' does not match the policy this config "
            f"builds (obs_dim={obs_dim}, action_dim={config.ACTION_DIM}). The "
            f"checkpoint was probably trained under a different config — "
            f"observation layout and ACTION_DIM must match. Original error:\n"
            f"{err}"
        ) from err

    # No dropout/batchnorm in PPOActorCritic, so this is a formality — but it
    # keeps the intent explicit: evaluation never trains this policy.
    policy.eval()
    print(f"[{label}] loaded policy from '{ckpt_path}' "
          f"({sum(p.numel() for p in policy.parameters()):,} params)")

    return DirectionalPolicy(env=env, policy=policy, target_c=target_c)
