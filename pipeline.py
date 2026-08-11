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
import splits
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
    # The persisted patient-level train/test split (splits.Split), or None when
    # config.USE_SPLIT is False. Carried here so every stage reads ONE
    # assignment: training must draw only from `split.train` and evaluation only
    # from `split.test`, and the whole point is lost if any stage recomputes it.
    split: object = None

    # ── convenience views ─────────────────────────────────────────────────────
    # Return dataset indices for a fold, or all indices when splitting is off,
    # so callers need no `if config.USE_SPLIT` of their own.

    def train_indices(self):
        import numpy as np
        if self.split is None:
            return np.arange(len(self.dataset), dtype=np.int64)
        return self.split.train

    def eval_indices(self):
        import numpy as np
        if self.split is None:
            return np.arange(len(self.dataset), dtype=np.int64)
        return self.split.test


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

    # The rsna_pneumonia profile's build_dataset IS ingestion.build_dataset, so
    # this stays byte-identical for the baseline domain. It is routed through
    # the module global rather than the profile attribute because smoke_test.py
    # and smoke_test_eval.py patch `pipeline.build_dataset` to shrink the
    # dataset — patching the profile's frozen field is not possible.
    if domain.name == "rsna_pneumonia":
        dataset = build_dataset(img_size=config.IMG_SIZE)
    else:
        dataset = domain.build_dataset(config.IMG_SIZE)

    # Resolve the train/test split HERE, once, before anything can touch the
    # dataset. Every stage then reads ctx.split rather than deriving its own —
    # see splits.py for why a recomputed split would not hold across the 8
    # generate-stage processes.
    split = None
    if config.USE_SPLIT:
        split = splits.load_or_create(dataset, domain=domain.name)
    else:
        print(
            "[split] USE_SPLIT=False — training and evaluation share every image. "
            "Reproduces the pre-split behaviour; NOT valid for a reported result."
        )

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
    if config.USE_GUIDANCE_ENSEMBLE:
        classifier = build_ensemble_classifier(
            weights_list=list(domain.ensemble_weights), pathology=domain.pathology
        )
    else:
        classifier = build_xrv_classifier(
            weights=domain.guidance_weights, pathology=domain.pathology
        )
    seg_net, lung_channel_idx = build_segmentation_net()
    obj_det                   = build_object_detector()

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
        split=split,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint loading (evaluation side)
# ─────────────────────────────────────────────────────────────────────────────

def load_generator_weights(generator, ckpt_dir: str = config.CKPT_DIR) -> str:
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

    Both direction subdirectories hold byte-identical copies (one shared
    LoRA generator, two policies), so whichever exists first is fine.

    Returns
    -------
    str : the path actually loaded from, for logging.
    """
    for label in ("n_to_p", "p_to_n"):
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
        generator=ctx.generator,
        seg_net=ctx.seg_net,
        obj_det=ctx.obj_det,
        budget_T=config.BUDGET_T,
        device=config.DEVICE,
        # Evaluation-time env: the EVAL fold. At eval the far pool is only used
        # by reset() to size the observation — the counterfactuals themselves are
        # generated for images scan_groups selected — but pointing it at the
        # training fold here would make the one env object in the eval process
        # disagree with the fold everything else uses.
        allowed_indices=ctx.eval_indices(),
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
