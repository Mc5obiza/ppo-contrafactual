"""
src/ppo.py
==========
PPO components:
  - Transition  — named storage for a single env step
  - PPOActorCritic — shared-trunk policy + value + auxiliary heads
  - compute_gae  — Generalised Advantage Estimation with reward shaping
  - PPOTrainer   — rollout collection and policy update loop

CHANGE LOG
----------
- PPOTrainer.train(): added RESUMABLE CHECKPOINTING and a wall-clock guard.
  * State goes to ckpt_dir/<label>/ppo_latest.pt every
    config.PPO_CKPT_EVERY_N_UPDATES updates, and once more when training
    completes. Separate from evaluate.save_checkpoints()'s ppo_policy.pt,
    which is the final artefact eval_main.py reloads and only appears once a
    direction has fully finished.
  * WHY: publicgpu caps --time at 24h and a direction does not reliably fit.
    Job 17653310 was SIGKILLed at the walltime ~30 of 100 updates into n_to_p
    with no PPO checkpoint written yet, so all 24h were unrecoverable. The
    train/eval split that preceded this change protected the boundary BETWEEN
    phases; this protects the inside of the longest phase.
  * The saved state includes cv_pre_ramp / lambda_frozen / lambda_last_ok /
    ramp_history, not just policy+optimizer. Restoring only the weights would
    re-baseline the CV probe against a mid-ramp policy and release a latched
    freeze, i.e. reintroduce the ramp oscillation _lambda_for_update rules out.
  * PPO_WALLCLOCK_BUDGET_S (env var, set by run_ppo.slurm) makes train() stop
    BEFORE beginning an update that would overrun, save, and return False.
    The remaining-time estimate uses the SLOWEST update seen, not the mean,
    because CV-probe updates cost ~3x a plain one and a mean under-predicts
    exactly the updates most at risk of overrunning.
  * train() now returns bool: True = reached n_episodes, False = paused on the
    budget. main.py must not write final checkpoints or run diagnostics on
    False — the policy is mid-training.
- PPOTrainer: added the sparsity-reward annealing schedule and the automatic
  CV ramp-freeze safeguard.
  * The schedule keys off `update_count` (one update per episodes_per_update
    episodes) — NOT `update()`'s `n_epochs` argument, which is PPO's inner
    passes over the same batch and resets every update, and not
    N_DENOISE_EPOCHS, which belongs to the LoRA phase. `update_count` is the
    only one of the three that measures training progress.
  * Ramp bounds are FRACTIONS of total updates (SPARSITY_RAMP_*_FRAC), not
    absolute indices, because N_EPISODES is an unbenchmarked placeholder and
    absolute indices would silently land in the wrong place once it is
    re-tuned. Resolved and logged at train() entry.
  * lambda_sparsity is PUSHED into env.lambda_sparsity before each rollout.
    One-way: the env never reaches back into the trainer, so
    _compute_reward gets training progress without a new coupling direction.
  * The freeze is LATCHED and holds the last lambda that PASSED a probe, not
    the current one (the current value is what caused the drop). A recovering
    CV does not resume the ramp — an oscillating weight would make the run
    uninterpretable.
  * Probes are confined to the ramp window plus a lead-in for the baseline:
    each probe runs ENV_EVAL_HOLDOUT_N full policy episodes, ~3 updates of
    rollout compute, so probing outside the window is pure waste.
- get_action_and_value(): added `deterministic` flag for policy-guided
  inference (see src/inference.py, MedicalCounterfactualEnv.generate_via_policy).
- update(): FIXED — `shaped_rewards` returned by compute_gae() was being
  unpacked but never used; all loss terms referenced only `advantages`,
  `returns`, or a separately-computed `td_targets`. This meant
  config.DELTA_MIX (the tau-gated reward/advantage blend implemented in
  compute_gae) had ZERO effect on training despite being computed every
  update. Fixed by using shaped_rewards as the value-function target in
  place of the raw GAE `returns`, which is the standard place a shaped
  reward signal is meant to land. `returns` is still computed and
  available (e.g. for logging/debugging) but no longer drives L_value.
  If this doesn't empirically help, the alternative fix is to simply
  discard shaped_rewards and strip the shaping branch out of
  compute_gae() — see the code comment at the L_value line below.
"""

import math
import os
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import config


# Wall clock for PPO_WALLCLOCK_BUDGET_S, taken at import rather than at
# train() entry. It has to cover the WHOLE process, for two reasons:
#   - Steps 1-4 (dataset scan, model loading, LoRA resume) run before any
#     train() call and are charged against the same SLURM --time limit.
#   - `python main.py --direction both` calls train() twice. Timing from
#     train() entry would hand the second direction a fresh full budget, so
#     a 22h budget would permit ~44h and the job would still be SIGKILLed at
#     the walltime — the exact failure this guard exists to prevent.
# ppo.py is imported by main.py before any of that work starts, so import
# time is a good proxy for process start.
_PROCESS_START_S = time.monotonic()


# ─────────────────────────────────────────────────────────────────────────────
# Data structure
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Transition:
    obs:      torch.Tensor
    action:   torch.Tensor
    log_prob: torch.Tensor
    reward:   float
    value:    float
    done:     bool
    tau:      float


# ─────────────────────────────────────────────────────────────────────────────
# Policy network
# ─────────────────────────────────────────────────────────────────────────────

class PPOActorCritic(nn.Module):
    """
    Shared-trunk actor-critic with auxiliary feasibility, progress, and stop heads.

    Parameters
    ----------
    obs_dim : int
    action_dim : int
    init_log_std : float
        Initial value for the learnable log-std parameter, shared across
        all action dimensions at init (each dim can diverge during training
        since log_std has independent per-dim entries).
        exp(-0.5) ~= 0.6, a moderate starting exploration noise — smaller
        than the old fixed std=1.0, but not so small that early exploration
        collapses before the policy has learned anything.
    min_std : float
        Floor on std to prevent it collapsing to (near-)zero and blowing up
        log_prob / making PPO ratios numerically unstable.
    """

    def __init__(self, obs_dim: int, action_dim: int, init_log_std: float = -0.5,
                 min_std: float = 1e-3, trunk: str | None = None):
        super().__init__()
        self.obs_dim = obs_dim
        self.trunk_kind = trunk if trunk is not None else config.POLICY_TRUNK

        if self.trunk_kind == "conv":
            self._build_conv_trunk(obs_dim)
        elif self.trunk_kind == "mlp":
            # Original trunk. Kept so archived policies stay loadable and so the
            # conv change can be A/B'd. See config.POLICY_TRUNK.
            self.shared = nn.Sequential(
                nn.Linear(obs_dim, 512), nn.ReLU(),
                nn.Linear(512, 256),    nn.ReLU(),
            )
        else:
            raise ValueError(
                f"config.POLICY_TRUNK must be 'conv' or 'mlp', got {self.trunk_kind!r}"
            )

        self.actor             = nn.Linear(256, action_dim)
        self.log_std           = nn.Parameter(torch.full((action_dim,), init_log_std))
        self.min_std            = min_std
        self.critic             = nn.Linear(256, 1)
        self.feasibility_head   = nn.Linear(256, 1)
        self.progress_head      = nn.Linear(256, 1)
        self.stop_head          = nn.Linear(256, 1)

    # ──────────────────────────────────────────────────────────────────────────
    # Conv trunk
    # ──────────────────────────────────────────────────────────────────────────

    def _build_conv_trunk(self, obs_dim: int) -> None:
        """
        Small conv encoder over the observation's spatial grids.

        MedicalCounterfactualEnv.get_observation() concatenates, in this order:
            z_orig  LATENT_CHANNELS * LATENT_SIZE^2   (4 * 32 * 32 = 4096)
            z_t     LATENT_CHANNELS * LATENT_SIZE^2   (4096)
            s_t     1     step fraction
            d_t     1     |f(x_t) - c|
            c       1     target class
            m_t     IMG_SIZE^2                        (256 * 256 = 65536)
        for 73,731 total. Three of those six blocks are 2-D grids that the MLP
        trunk saw as an unstructured vector.

        The slicing is derived from config here rather than passed in, so a
        geometry change (LATENT_SIZE, IMG_SIZE) fails loudly at construction
        instead of silently misinterpreting the observation.
        """
        self._z_numel = config.LATENT_CHANNELS * config.LATENT_SIZE * config.LATENT_SIZE
        self._mask_hw = config.IMG_SIZE
        expected = 2 * self._z_numel + 3 + self._mask_hw * self._mask_hw
        if obs_dim != expected:
            raise ValueError(
                f"POLICY_TRUNK='conv' expects the observation layout built by "
                f"MedicalCounterfactualEnv.get_observation(): "
                f"2*{self._z_numel} (z_orig, z_t) + 3 (s_t, d_t, c) + "
                f"{self._mask_hw}^2 (mask) = {expected}, but obs_dim={obs_dim}. "
                f"Check config.LATENT_CHANNELS/LATENT_SIZE/IMG_SIZE against the "
                f"env, or set POLICY_TRUNK='mlp'."
            )

        # z_orig (4) + z_t (4) + pooled mask (1), all at LATENT_SIZE resolution.
        in_ch = 2 * config.LATENT_CHANNELS + 1
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(32,    64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64,   128, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        # + the 3 scalars, which carry the episode's progress state and must not
        # be diluted into the spatial stack.
        self.head = nn.Sequential(
            nn.Linear(128 + 3, 256), nn.ReLU(),
            nn.Linear(256, 256),     nn.ReLU(),
        )

    def _conv_features(self, obs: torch.Tensor) -> torch.Tensor:
        """Flat observation -> 256-d feature, reshaping the grids on the way."""
        B = obs.shape[0]
        Z, L, C = self._z_numel, config.LATENT_SIZE, config.LATENT_CHANNELS

        z_orig  = obs[:, :Z].view(B, C, L, L)
        z_t     = obs[:, Z : 2 * Z].view(B, C, L, L)
        scalars = obs[:, 2 * Z : 2 * Z + 3]
        mask    = obs[:, 2 * Z + 3 :].view(B, 1, self._mask_hw, self._mask_hw)

        # The mask arrives at full image resolution (88.9% of the observation).
        # Downsample to the latent grid so it stacks with the latents; average
        # pooling keeps the soft lung-probability values rather than thresholding.
        mask = F.adaptive_avg_pool2d(mask, (L, L))

        h = self.conv(torch.cat([z_orig, z_t, mask], dim=1))
        return self.head(torch.cat([h, scalars], dim=-1))

    def _std(self, like: torch.Tensor) -> torch.Tensor:
        """Broadcast the learnable std to the batch shape of `like`."""
        return self.log_std.exp().clamp(min=self.min_std).expand_as(like)

    def forward(
        self, obs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        feat = (
            self._conv_features(obs) if self.trunk_kind == "conv"
            else self.shared(obs)
        )
        return (
            self.actor(feat),
            self.critic(feat).squeeze(-1),
            torch.sigmoid(self.feasibility_head(feat)).squeeze(-1),
            self.progress_head(feat).squeeze(-1),
            torch.sigmoid(self.stop_head(feat)).squeeze(-1),
        )

    def get_action_and_value(
        self, obs: torch.Tensor, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        obs : torch.Tensor
        deterministic : bool
            If True, returns the distribution mean (the raw actor logits)
            directly instead of sampling from Normal(logits, std). Used at
            evaluation/inference time (see src/inference.py and
            MedicalCounterfactualEnv.generate_via_policy) so PPO-guided
            counterfactual generation is reproducible and not subject to
            the exploration noise that's appropriate during training but
            not desirable when reporting metrics or generating figures.
            Defaults to False so existing training call sites (collect_rollout,
            update) are unaffected.
        """
        logits, value, feas, prog, stop = self(obs)
        std      = self._std(logits)
        dist     = torch.distributions.Normal(logits, std)
        action   = logits if deterministic else dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        return action, log_prob, value, feas, prog, stop


# ─────────────────────────────────────────────────────────────────────────────
# Generalised Advantage Estimation
# ─────────────────────────────────────────────────────────────────────────────

def compute_gae(
    transitions: list[Transition],
    last_value: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute shaped rewards, advantages, and returns via GAE.

    Parameters
    ----------
    transitions : list[Transition]
    last_value : float — bootstrap value for the state after the last transition

    Returns
    -------
    (shaped_rewards, advantages, returns) : 3-tuple of 1-D float tensors on DEVICE

    Note
    ----
    shaped_rewards blends the raw step reward with a tau-gated advantage
    term (config.DELTA_MIX), intended to inject additional shaping signal
    on steps where the policy hasn't yet decided to "stop" (tau <= 0.5).
    As of this revision, PPOTrainer.update() uses shaped_rewards as the
    value-function regression target (see L_value below) so this signal
    actually participates in training instead of being discarded.
    """
    T_len      = len(transitions)
    adv        = np.zeros(T_len, dtype=np.float32)
    ret        = np.zeros(T_len, dtype=np.float32)
    shaped_r   = np.zeros(T_len, dtype=np.float32)
    gae, next_val = 0.0, last_value

    for t in reversed(range(T_len)):
        tr = transitions[t]
        if tr.done:
            delta_t = tr.reward - tr.value
            gae     = delta_t
            next_val = 0.0
        else:
            delta_t = tr.reward + config.GAMMA * next_val - tr.value
            gae     = delta_t + config.GAMMA * config.GAE_LAMBDA * gae
        adv[t] = gae
        ret[t] = gae + tr.value
        next_val = tr.value
        shaped_r[t] = (
            tr.reward if tr.tau > 0.5 else tr.reward + config.DELTA_MIX * adv[t]
        )

    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    return (
        torch.tensor(shaped_r, device=config.DEVICE),
        torch.tensor(adv,      device=config.DEVICE),
        torch.tensor(ret,      device=config.DEVICE),
    )


# ─────────────────────────────────────────────────────────────────────────────
# PPO trainer
# ─────────────────────────────────────────────────────────────────────────────

class PPOTrainer:
    """
    Handles rollout collection and PPO policy updates.

    Parameters
    ----------
    env : MedicalCounterfactualEnv
    policy : PPOActorCritic
    lr : float
    """

    def __init__(self, env, policy: PPOActorCritic, lr: float = config.LR):
        self.env    = env
        self.policy = policy
        self.opt    = torch.optim.AdamW(policy.parameters(), lr=lr)

        # ── sparsity annealing + CV safeguard state ───────────────────────────
        # cv_pre_ramp    : CV baseline measured just before the ramp starts
        # lambda_frozen  : once set, lambda_sparsity is pinned here for the
        #                  REST of training. Latched deliberately — see
        #                  _lambda_for_update.
        # lambda_last_ok : lambda at the most recent probe that passed; this is
        #                  what the freeze holds, not the current value, since
        #                  the current value is the one that caused the drop.
        self.cv_pre_ramp     = None
        self.lambda_frozen   = None
        self.lambda_last_ok  = 0.0
        self.ramp_history    = []

        # Per-step reward diagnostics from the most recent collect_rollout.
        # Populated there rather than returned, so collect_rollout's signature
        # stays a plain Transition list for main.py's plot_reward_trace call.
        self.last_rollout_stats: dict = {}

    # ──────────────────────────────────────────────────────────────────────────
    # Resumable training state
    #
    # Mirrors factory/finetuning.py's per-epoch checkpoint/resume: a single
    # overwritten '<label>/ppo_latest.pt', written via a .tmp + os.replace so a
    # crash mid-save cannot corrupt the file you'd resume from.
    #
    # DISTINCT FROM evaluate.save_checkpoints(), which writes the FINAL
    # ppo_policy.pt that eval_main.py/pipeline.load_directional_policy reload.
    # That one only appears once a direction has fully finished. This one
    # exists to survive a walltime kill mid-direction, and carries the
    # optimizer and schedule state that ppo_policy.pt deliberately does not.
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _state_path(ckpt_dir: str, label: str) -> str:
        return os.path.join(ckpt_dir, label, "ppo_latest.pt")

    def _save_state(
        self,
        ckpt_dir: str,
        label: str,
        update_count: int,
        total_episodes_run: int,
    ) -> None:
        """
        Persist everything train() needs to pick up exactly where it left off.

        The sparsity/CV fields are NOT optional bookkeeping. Dropping them
        would restart a resumed run with cv_pre_ramp=None and
        lambda_frozen=None, which re-baselines the CV probe against a
        mid-ramp policy and silently releases a latched freeze — precisely
        the ramp oscillation _lambda_for_update's docstring rules out.
        """
        path = self._state_path(ckpt_dir, label)
        os.makedirs(os.path.dirname(path), exist_ok=True)

        checkpoint = {
            "update_count":        update_count,
            "total_episodes_run":  total_episodes_run,
            "policy_state_dict":   self.policy.state_dict(),
            "optimizer_state_dict": self.opt.state_dict(),
            # sparsity annealing + CV safeguard state
            "cv_pre_ramp":         self.cv_pre_ramp,
            "lambda_frozen":       self.lambda_frozen,
            "lambda_last_ok":      self.lambda_last_ok,
            "ramp_history":        self.ramp_history,
        }
        tmp_path = path + ".tmp"
        torch.save(checkpoint, tmp_path)
        os.replace(tmp_path, path)
        print(f"[ppo] state saved: {path} (update {update_count}, "
              f"{total_episodes_run} episodes)")

    def _load_state(self, ckpt_dir: str, label: str) -> tuple[int, int]:
        """
        Restore from '<label>/ppo_latest.pt' if present.

        Returns
        -------
        (update_count, total_episodes_run) already COMPLETED — train() resumes
        by continuing its loop from these, so (0, 0) means a fresh start.
        """
        path = self._state_path(ckpt_dir, label)
        if not os.path.exists(path):
            print(f"[ppo] no resumable state at {path} — starting fresh")
            return 0, 0

        print(f"[ppo] found resumable state at {path} — resuming")
        checkpoint = torch.load(path, map_location=config.DEVICE)

        try:
            self.policy.load_state_dict(checkpoint["policy_state_dict"])
        except (RuntimeError, ValueError, KeyError) as err:
            # obs_dim comes off a live env.reset() and the trunk is selected by
            # config.POLICY_TRUNK, so a trunk switch ("conv" <-> "mlp", a 200k
            # vs 37.9M parameter difference) or an ACTION_DIM change makes a
            # saved policy unloadable. Same failure mode finetuning._load_checkpoint
            # guards for on LORA_R, and the raw error does not say why either.
            raise RuntimeError(
                f"[ppo] cannot resume '{path}' into the policy this config builds "
                f"(POLICY_TRUNK={config.POLICY_TRUNK!r}, ACTION_DIM={config.ACTION_DIM}).\n"
                f"The state was written under a different trunk or action dim; both "
                f"are baked into the parameter shapes.\n"
                f"To retrain this direction from scratch, delete the state file:\n"
                f"    rm {path}\n"
                f"Original error:\n{err}"
            ) from err
        self.opt.load_state_dict(checkpoint["optimizer_state_dict"])

        self.cv_pre_ramp    = checkpoint["cv_pre_ramp"]
        self.lambda_frozen  = checkpoint["lambda_frozen"]
        self.lambda_last_ok = checkpoint["lambda_last_ok"]
        self.ramp_history   = checkpoint["ramp_history"]

        update_count       = checkpoint["update_count"]
        total_episodes_run = checkpoint["total_episodes_run"]
        frozen = "" if self.lambda_frozen is None else f", ramp FROZEN at {self.lambda_frozen:.4f}"
        print(f"[ppo] resumed from update {update_count} "
              f"({total_episodes_run} episodes done{frozen})")
        return update_count, total_episodes_run

    # ──────────────────────────────────────────────────────────────────────────
    # Sparsity annealing schedule
    # ──────────────────────────────────────────────────────────────────────────

    def _ramp_bounds(self, total_updates: int) -> tuple[int, int]:
        """Resolve the fractional ramp window to absolute update indices."""
        start = int(round(config.SPARSITY_RAMP_START_FRAC * total_updates))
        end   = int(round(config.SPARSITY_RAMP_END_FRAC   * total_updates))
        return start, max(end, start + 1)      # guarantee a non-empty ramp

    def _lambda_for_update(self, update_idx: int, total_updates: int) -> float:
        """
        lambda_sparsity for this update.

        [0, start)     -> 0.0                        (validity only)
        [start, end)   -> ramp 0 -> LAM_SPARSITY_TARGET
        [end, ...]     -> LAM_SPARSITY_TARGET

        A latched freeze (self.lambda_frozen) overrides everything. The freeze
        is intentionally NOT re-evaluated per probe: letting a recovering CV
        un-freeze the ramp would make lambda oscillate and the run
        uninterpretable.
        """
        if not config.USE_SPARSITY_REWARD:
            return 0.0
        if self.lambda_frozen is not None:
            return self.lambda_frozen
        target = config.LAM_SPARSITY_TARGET
        if not config.USE_SPARSITY_ANNEALING:
            return target                       # full weight from update 0

        start, end = self._ramp_bounds(total_updates)
        if update_idx < start:
            return 0.0
        if update_idx >= end:
            return target
        frac = (update_idx - start) / max(1, end - start)
        if config.SPARSITY_RAMP_SHAPE == "cosine":
            frac = 0.5 * (1.0 - np.cos(np.pi * frac))
        elif config.SPARSITY_RAMP_SHAPE != "linear":
            raise ValueError(
                f"SPARSITY_RAMP_SHAPE must be 'linear' or 'cosine', "
                f"got {config.SPARSITY_RAMP_SHAPE!r}"
            )
        return float(target * frac)

    # ──────────────────────────────────────────────────────────────────────────
    # CV safeguard probe
    # ──────────────────────────────────────────────────────────────────────────

    def _measure_holdout_cv(self) -> float | None:
        """
        Mean CV on the env's held-out far-pool images, using the SAME
        criterion as metrics.run_full_evaluation (tau=0.8, boundary=0.5) so the
        number is comparable to reported CV rather than a second definition of
        validity.

        Runs the trained policy deterministically over each held-out image.
        COST: one full episode per image — see config.ENV_EVAL_HOLDOUT_N.

        Temporarily shortens env.T rather than passing max_steps: per the note
        in MedicalCounterfactualEnv.generate_via_policy, get_observation's
        step-fraction feature is computed against self.T, so a max_steps that
        differs from T feeds the policy step fractions it never saw in
        training. Restored in a finally.
        """
        from src.metrics import counterfactual_validity

        images = self.env.eval_holdout_images()
        if not images:
            return None

        to_rgb = lambda t: t.repeat(1, 3, 1, 1) if t.shape[1] == 1 else t
        target = "positive" if float(self.env.c) == 1.0 else "negative"

        # The sparsity term must not perturb the probe; reward is unused during
        # generate_via_policy anyway, but keep it explicit and restore after.
        saved_T, saved_lam = self.env.T, self.env.lambda_sparsity
        self.env.T = min(config.CV_CHECK_MAX_STEPS, saved_T)
        self.env.lambda_sparsity = 0.0
        f_x, f_xc = [], []
        try:
            with torch.no_grad():
                for x in images:
                    x_cf, _traj, _info = self.env.generate_via_policy(
                        x_orig=x, policy=self.policy, deterministic=True
                    )
                    f_x.append(self.env.f(to_rgb(x)).reshape(-1))
                    f_xc.append(self.env.f(to_rgb(x_cf.to(x.device))).reshape(-1))
        finally:
            self.env.T = saved_T
            self.env.lambda_sparsity = saved_lam

        return counterfactual_validity(
            torch.cat(f_x), torch.cat(f_xc), target=target
        )

    def _cv_probe_due(self, update_idx: int, total_updates: int) -> bool:
        """
        Probes are restricted to the ramp window plus a short lead-in for the
        baseline: outside it the safeguard has nothing to act on, and each
        probe costs ~3 updates of rollout compute.
        """
        if not (config.USE_SPARSITY_REWARD and config.USE_SPARSITY_ANNEALING):
            return False
        if self.lambda_frozen is not None:
            return False                        # already latched, nothing to do
        start, end = self._ramp_bounds(total_updates)
        lead_in = max(1, config.CV_CHECK_EVERY_N_UPDATES)
        if not (start - lead_in <= update_idx <= end):
            return False
        return update_idx % config.CV_CHECK_EVERY_N_UPDATES == 0

    def _apply_cv_safeguard(
        self, update_idx: int, total_updates: int, lam: float
    ) -> float | None:
        """
        Measure held-out CV and latch the ramp freeze if CV has dropped more
        than CV_DROP_TOLERANCE below its pre-ramp baseline. Returns the
        measured CV (None if no probe ran).
        """
        if not self._cv_probe_due(update_idx, total_updates):
            return None

        cv = self._measure_holdout_cv()
        if cv is None or not np.isfinite(cv):
            print(f"[ppo] CV probe at update {update_idx} returned {cv} — skipping check")
            return cv

        start, _end = self._ramp_bounds(total_updates)
        n_hold = len(self.env._far_indices_eval)
        if update_idx < start or self.cv_pre_ramp is None:
            # Pre-ramp: this is the baseline the freeze condition compares to.
            self.cv_pre_ramp = cv
            print(f"[ppo] CV baseline (pre-ramp) = {cv:.4f} on {n_hold} held-out "
                  f"images (resolution {1.0 / max(n_hold, 1):.3f}/image)")
        else:
            drop = self.cv_pre_ramp - cv
            if drop > config.CV_DROP_TOLERANCE:
                self.lambda_frozen = self.lambda_last_ok
                print(
                    f"[ppo] ⚠ RAMP FROZEN at update {update_idx}: CV {cv:.4f} is "
                    f"{drop:.4f} below the pre-ramp baseline {self.cv_pre_ramp:.4f} "
                    f"(tolerance {config.CV_DROP_TOLERANCE}). Holding "
                    f"lambda_sparsity={self.lambda_frozen:.4f} (last value that "
                    f"passed) for the rest of training. Freeze is latched — it "
                    f"will not resume if CV recovers."
                )
            else:
                self.lambda_last_ok = lam
                print(f"[ppo] CV probe at update {update_idx}: {cv:.4f} "
                      f"(baseline {self.cv_pre_ramp:.4f}, drop {drop:+.4f}, "
                      f"lambda={lam:.4f}) — ramp continues")
        return cv

    # ──────────────────────────────────────────────────────────────────────────

    def collect_rollout(self, n_episodes: int = 1) -> list[Transition]:
        """
        Collect episodes. Return type is unchanged (a flat Transition list, as
        main.py's plot_reward_trace call expects); per-step reward diagnostics
        are stashed on `self.last_rollout_stats` instead of widening the
        signature.
        """
        all_transitions = []
        term_sums: dict[str, float] = {}
        edit_l1s, n_valid, n_score_ok, n_budget_ok = [], 0, 0, 0

        for _ in range(n_episodes):
            obs, done = self.env.reset(), False
            while not done:
                with torch.no_grad():
                    action, log_prob, value, _, _, _ = self.policy.get_action_and_value(
                        obs.unsqueeze(0)
                    )
                action = action.squeeze(0)
                obs_next, reward, done, info = self.env.step(action)
                all_transitions.append(
                    Transition(obs, action, log_prob.squeeze(0),
                               reward, value.item(), done, info["tau"])
                )
                for k, v in info.get("reward_terms", {}).items():
                    term_sums[k] = term_sums.get(k, 0.0) + v
                if "edit_l1" in info:
                    edit_l1s.append(info["edit_l1"])
                n_valid     += int(bool(info.get("valid", False)))
                n_score_ok  += int(bool(info.get("score_ok", False)))
                n_budget_ok += int(bool(info.get("budget_ok", False)))
                obs = obs_next

        n_steps = max(len(all_transitions), 1)
        self.last_rollout_stats = dict(
            # Mean weighted contribution of each reward term per STEP. Summing
            # these gives the mean per-step reward, so `share` below is a true
            # decomposition rather than a normalised guess.
            terms={k: v / n_steps for k, v in term_sums.items()},
            mean_edit_l1=(float(np.mean(edit_l1s)) if edit_l1s else float("nan")),
            frac_valid=n_valid / n_steps,
            frac_score_ok=n_score_ok / n_steps,
            frac_budget_ok=n_budget_ok / n_steps,
            n_steps=n_steps,
        )
        return all_transitions

    @staticmethod
    def _format_reward_split(stats: dict) -> str:
        """
        One-line reward decomposition: each term's mean weighted contribution
        and its share of total ABSOLUTE reward mass.

        Share is over |contribution| deliberately — the terms have mixed signs,
        so a signed denominator lets a large positive and a large negative term
        cancel and report a misleadingly small total. What matters here is how
        much each term is PULLING, regardless of direction.

        This is the check that the v1 rebalance landed: the flip terms
        (hit+stop) were 93% of the return before, and the target is ~40%.
        """
        terms = stats.get("terms", {})
        if not terms:
            return "reward split: n/a (env did not report reward_terms)"

        mass = sum(abs(v) for v in terms.values()) or 1.0
        flip_share = (abs(terms.get("hit", 0.0)) + abs(terms.get("stop", 0.0))) / mass
        pres_share = (
            abs(terms.get("pres", 0.0))
            + abs(terms.get("fid", 0.0))
            + abs(terms.get("sparsity", 0.0))
        ) / mass
        parts = " ".join(
            f"{k}={v:+.3f}({100 * abs(v) / mass:.0f}%)"
            for k, v in sorted(terms.items(), key=lambda kv: -abs(kv[1]))
        )
        return (
            f"reward split [flip={100 * flip_share:.0f}% pres={100 * pres_share:.0f}%] "
            f"{parts}"
        )

    # ──────────────────────────────────────────────────────────────────────────

    def update(
        self, transitions: list[Transition], n_epochs: int = 4
    ) -> dict[str, float]:
        shaped_rewards, advantages, returns = compute_gae(transitions)

        obs_b       = torch.stack([t.obs      for t in transitions])
        act_b       = torch.stack([t.action   for t in transitions])
        logp_old_b  = torch.stack([t.log_prob for t in transitions])
        stop_labels = torch.tensor(
            [float(t.tau > 0.5) for t in transitions], device=config.DEVICE
        )
        next_vals   = [
            transitions[min(i + 1, len(transitions) - 1)].value
            for i in range(len(transitions))
        ]
        td_targets  = torch.tensor(
            [
                t.reward + config.GAMMA * (0.0 if t.done else next_vals[i])
                for i, t in enumerate(transitions)
            ],
            dtype=torch.float32,
            device=config.DEVICE,
        )

        logs = {}
        for _ in range(n_epochs):
            logits, values, feas_pred, prog_pred, stop_pred = self.policy(obs_b)
            std      = self.policy._std(logits)
            dist     = torch.distributions.Normal(logits, std)
            logp_new = dist.log_prob(act_b).sum(-1)
            entropy  = dist.entropy().sum(-1).mean()
            ratio    = torch.exp(logp_new - logp_old_b.detach()).to(config.DEVICE)
            surr1    = ratio * advantages
            surr2    = torch.clamp(ratio, 1 - config.PPO_CLIP, 1 + config.PPO_CLIP) * advantages
            L_ppo    = -torch.min(surr1, surr2).mean()

            # FIX: previously F.mse_loss(values, returns), which meant
            # shaped_rewards (and therefore config.DELTA_MIX) had no effect
            # anywhere in training. shaped_rewards is the tau-gated blend
            # of raw reward and advantage computed in compute_gae() —
            # using it here as the value regression target is the standard
            # place for a shaped-reward signal to land, and actually
            # activates DELTA_MIX. `returns` (raw GAE bootstrap target) is
            # still computed above and available if you want to compare
            # the two empirically or revert to the old behaviour.
            L_value  = F.mse_loss(values, shaped_rewards)

            feas_labels = (returns > 0).float()
            L_feas      = F.binary_cross_entropy(feas_pred, feas_labels)
            L_prog_aux  = F.mse_loss(prog_pred, td_targets)
            L_stop_aux  = F.binary_cross_entropy(stop_pred, stop_labels)

            # Saturation hinge. Every action dim is squashed through a sigmoid
            # in environment.step(), and in v1 the actor's mean drifted to
            # ~±5.5 where that sigmoid is flat — guidance diagnostics measured
            # guidance_bias at the two ENDS of its range with a std of 0.005
            # despite sampling at std≈0.61, i.e. the squash absorbed all
            # exploration and the policy was effectively deterministic and
            # state-independent. Penalising only |logit| beyond
            # ACT_SAT_THRESHOLD leaves the responsive region untouched and
            # pulls the dead tail back in. See config.LAM_ACT_SAT.
            L_act_sat = (
                F.relu(logits.abs() - config.ACT_SAT_THRESHOLD).pow(2).mean()
            )

            L_total = (
                L_ppo
                + config.C_VALUE    * L_value
                - config.C_ENTROPY  * entropy
                + config.LAM_FEAS_AUX * L_feas
                + config.LAM_PROG_AUX * L_prog_aux
                + config.LAM_STOP_AUX * L_stop_aux
                + config.LAM_ACT_SAT  * L_act_sat
            )
            self.opt.zero_grad()
            L_total.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.opt.step()
            logs = dict(
                L_total=L_total.item(), L_ppo=L_ppo.item(),
                L_value=L_value.item(), entropy=entropy.item(),
                std_mean=std.mean().item(),
                L_act_sat=L_act_sat.item(),
                logit_absmax=logits.abs().max().item(),
            )

        return logs

    # ──────────────────────────────────────────────────────────────────────────

    def train(
        self,
        n_episodes: int = config.N_EPISODES,
        log_every: int = 5,
        episodes_per_update: int = 8,
        label: str | None = None,
        ckpt_dir: str = config.CKPT_DIR,
        resume: bool = True,
    ) -> bool:
        """
        Run PPO for n_episodes, checkpointing so a walltime kill is resumable.

        Parameters
        ----------
        label    : direction name ("n_to_p" / "p_to_n"). Selects the state file
                   at ckpt_dir/<label>/ppo_latest.pt. None disables
                   checkpointing entirely (used by smoke tests that have no
                   direction of their own) — the wall-clock guard is disabled
                   with it, since stopping early without saving would just
                   throw the work away.
        ckpt_dir : root checkpoint directory; the per-direction subdirectory is
                   created under it, matching where save_checkpoints() writes.
        resume   : continue from an existing state file if one is present. Set
                   False to force a fresh run WITHOUT deleting the file.

        Returns
        -------
        bool : True if training reached n_episodes, False if it stopped early
               on the wall-clock budget. main.py uses this to decide whether
               it may go on to write the final checkpoint and diagnostics.
        """
        checkpointing = label is not None

        total_episodes_run = 0
        update_count       = 0
        total_updates      = max(1, math.ceil(n_episodes / episodes_per_update))

        if checkpointing and resume:
            update_count, total_episodes_run = self._load_state(ckpt_dir, label)
            if total_episodes_run >= n_episodes:
                print(f"[ppo] state already at {total_episodes_run} episodes "
                      f">= n_episodes={n_episodes} — nothing to do")
                return True

        # Soft deadline, set by run_ppo.slurm below the SLURM --time limit. The
        # point is to convert a SIGKILL at the walltime (which loses everything
        # since the last checkpoint AND leaves no final artefacts) into a clean
        # save-and-exit that the next submission continues from.
        budget_s = os.environ.get("PPO_WALLCLOCK_BUDGET_S")
        deadline_s = float(budget_s) if budget_s else None
        if deadline_s is not None:
            if checkpointing:
                spent = (time.monotonic() - _PROCESS_START_S) / 3600
                print(f"[ppo] wall-clock budget: {deadline_s / 3600:.2f}h for the "
                      f"whole process ({spent:.2f}h already spent on startup) "
                      f"— will checkpoint and exit before it is reached")
            else:
                # Without a label there is nowhere to save, so honouring the
                # budget would discard the run rather than pause it. Say so
                # rather than silently ignoring an env var the user set.
                print("[ppo] NOTE: PPO_WALLCLOCK_BUDGET_S is set but label=None, "
                      "so there is no state file to resume from — the budget is "
                      "IGNORED and this run will not stop early.")
                deadline_s = None

        if config.USE_SPARSITY_ANNEALING and not config.USE_SPARSITY_REWARD:
            print("[ppo] NOTE: USE_SPARSITY_ANNEALING=True but "
                  "USE_SPARSITY_REWARD=False — the annealing schedule is inert.")
        if config.USE_SPARSITY_REWARD:
            start, end = self._ramp_bounds(total_updates)
            mode = (
                f"ramp {config.SPARSITY_RAMP_SHAPE} over updates [{start}, {end})"
                if config.USE_SPARSITY_ANNEALING
                else "FULL WEIGHT from update 0 (annealing disabled)"
            )
            print(
                f"[ppo] sparsity reward ON: target={config.LAM_SPARSITY_TARGET}, "
                f"{mode}; {total_updates} total updates "
                f"({n_episodes} episodes / {episodes_per_update} per update)"
            )

        # Worst per-update duration seen so far, used to decide whether ANOTHER
        # update fits in the remaining budget. Max rather than mean: an update
        # that happens to carry a CV probe costs ~3 extra updates of rollout, so
        # a mean would under-predict exactly the updates most likely to overrun.
        slowest_update_s = 0.0

        while total_episodes_run < n_episodes:
            t_update = time.monotonic()
            batch_size  = min(episodes_per_update, n_episodes - total_episodes_run)

            # Push the annealed weight into the env BEFORE collecting, so every
            # transition in this batch is scored under one consistent lambda.
            lam = self._lambda_for_update(update_count, total_updates)
            self.env.lambda_sparsity = lam

            transitions = self.collect_rollout(n_episodes=batch_size)
            logs        = self.update(transitions)
            total_episodes_run += batch_size
            update_count       += 1
            avg_ep_len          = len(transitions) / batch_size

            # Automatic safeguard: measures held-out CV and latches the ramp
            # freeze itself — not just a log line to watch.
            cv_now = self._apply_cv_safeguard(update_count, total_updates, lam)
            if cv_now is not None:
                self.ramp_history.append(dict(
                    update=update_count, lambda_sparsity=lam, cv=float(cv_now),
                    cv_pre_ramp=self.cv_pre_ramp, frozen=self.lambda_frozen,
                ))

            # Print the reward decomposition on the FIRST update as well as on
            # the regular cadence: the whole point of the v2 rebalance is that
            # you can see whether it landed without waiting for a full run.
            if update_count == 1 or update_count % log_every == 0:
                ep_return = sum(t.reward for t in transitions) / batch_size
                frozen = "" if self.lambda_frozen is None else " [FROZEN]"
                st = self.last_rollout_stats
                print(
                    f"Update {update_count:4d} | episodes={total_episodes_run:4d} | "
                    f"avg_ep_len={avg_ep_len:5.2f} | avg_return={ep_return:8.3f} | "
                    f"L_ppo={logs['L_ppo']:.4f} | L_value={logs['L_value']:.4f} | "
                    f"std_mean={logs['std_mean']:.4f} | "
                    f"lam_sparse={lam:.4f}{frozen}"
                )
                print(f"           {self._format_reward_split(st)}")
                print(
                    f"           edit_L1={st.get('mean_edit_l1', float('nan')):.4f} "
                    f"(budget {config.EDIT_BUDGET_L1}) | "
                    f"steps score_ok={100 * st.get('frac_score_ok', 0):.0f}% "
                    f"budget_ok={100 * st.get('frac_budget_ok', 0):.0f}% "
                    f"valid={100 * st.get('frac_valid', 0):.0f}% | "
                    f"|logit|max={logs['logit_absmax']:.2f} "
                    f"(sat>{config.ACT_SAT_THRESHOLD})"
                )

            slowest_update_s = max(slowest_update_s, time.monotonic() - t_update)

            if checkpointing and update_count % config.PPO_CKPT_EVERY_N_UPDATES == 0:
                self._save_state(ckpt_dir, label, update_count, total_episodes_run)

            # Stop BEFORE starting an update that would run past the budget,
            # not after overrunning it — an update in flight at the SLURM
            # walltime is killed mid-rollout and its work is lost either way.
            if deadline_s is not None and total_episodes_run < n_episodes:
                elapsed = time.monotonic() - _PROCESS_START_S
                if elapsed + slowest_update_s >= deadline_s:
                    self._save_state(ckpt_dir, label, update_count, total_episodes_run)
                    print(
                        f"[ppo] wall-clock budget reached at update "
                        f"{update_count}/{total_updates} "
                        f"({elapsed / 3600:.2f}h of {deadline_s / 3600:.2f}h, "
                        f"slowest update {slowest_update_s / 60:.1f} min) — "
                        f"checkpoint saved, resubmit the SAME job to continue."
                    )
                    return False

        if checkpointing:
            self._save_state(ckpt_dir, label, update_count, total_episodes_run)

        if config.USE_GRADCAM_MASK and self.env._cam_episodes_total:
            n_bad = self.env._cam_episodes_degenerate
            n_tot = self.env._cam_episodes_total
            print(f"[ppo] Grad-CAM mask: {n_tot - n_bad}/{n_tot} episodes constrained, "
                  f"{n_bad} degenerate ({100.0 * n_bad / n_tot:.1f}%)")
            if n_bad:
                print("[ppo] ⚠ degenerate episodes ran with the lung-only mask and "
                      "did NOT carry the locality constraint — do not report this "
                      "run as a clean '+gradcam_mask' arm without accounting for them.")

            # PARTIAL degeneracy: the ensemble CAM was averaged over fewer
            # members than intended. Distinct from the total failures above,
            # which fall back to the lung-only mask entirely.
            #
            # Worth watching: on synthetic probes the rsna member returned a
            # degenerate CAM far more often than chex, and at probabilities
            # around 0.5 — i.e. not from output saturation but from Grad-CAM's
            # all-negative-contribution edge case (relu zeroes the whole map).
            # If this rate is high, the "ensemble" CAM is effectively a
            # chex-only CAM and should be described that way.
            n_part = getattr(self.env.generator, "_cam_partial_degenerate", 0)
            if n_part:
                print(f"[ppo] Grad-CAM partial degeneracy: {n_part}/{n_tot} episodes "
                      f"({100.0 * n_part / n_tot:.1f}%) averaged over a SUBSET of "
                      f"ensemble members. At a high rate the mask is effectively "
                      f"single-member — report it as such.")

        return True