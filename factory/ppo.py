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

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import config


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

    def __init__(self, obs_dim: int, action_dim: int, init_log_std: float = -0.5, min_std: float = 1e-3):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, 512), nn.ReLU(),
            nn.Linear(512, 256),    nn.ReLU(),
        )
        self.actor             = nn.Linear(256, action_dim)
        self.log_std           = nn.Parameter(torch.full((action_dim,), init_log_std))
        self.min_std            = min_std
        self.critic             = nn.Linear(256, 1)
        self.feasibility_head   = nn.Linear(256, 1)
        self.progress_head      = nn.Linear(256, 1)
        self.stop_head          = nn.Linear(256, 1)

    def _std(self, like: torch.Tensor) -> torch.Tensor:
        """Broadcast the learnable std to the batch shape of `like`."""
        return self.log_std.exp().clamp(min=self.min_std).expand_as(like)

    def forward(
        self, obs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        feat = self.shared(obs)
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

    # ──────────────────────────────────────────────────────────────────────────

    def collect_rollout(self, n_episodes: int = 1) -> list[Transition]:
        all_transitions = []
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
                obs = obs_next
        return all_transitions

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
            L_total = (
                L_ppo
                + config.C_VALUE    * L_value
                - config.C_ENTROPY  * entropy
                + config.LAM_FEAS_AUX * L_feas
                + config.LAM_PROG_AUX * L_prog_aux
                + config.LAM_STOP_AUX * L_stop_aux
            )
            self.opt.zero_grad()
            L_total.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.opt.step()
            logs = dict(
                L_total=L_total.item(), L_ppo=L_ppo.item(),
                L_value=L_value.item(), entropy=entropy.item(),
                std_mean=std.mean().item(),
            )

        return logs

    # ──────────────────────────────────────────────────────────────────────────

    def train(
        self,
        n_episodes: int = config.N_EPISODES,
        log_every: int = 5,
        episodes_per_update: int = 8,
    ) -> None:
        total_episodes_run = 0
        update_count       = 0

        while total_episodes_run < n_episodes:
            batch_size  = min(episodes_per_update, n_episodes - total_episodes_run)
            transitions = self.collect_rollout(n_episodes=batch_size)
            logs        = self.update(transitions)
            total_episodes_run += batch_size
            update_count       += 1
            avg_ep_len          = len(transitions) / batch_size

            if update_count % log_every == 0:
                ep_return = sum(t.reward for t in transitions) / batch_size
                print(
                    f"Update {update_count:4d} | episodes={total_episodes_run:4d} | "
                    f"avg_ep_len={avg_ep_len:5.2f} | avg_return={ep_return:8.3f} | "
                    f"L_ppo={logs['L_ppo']:.4f} | L_value={logs['L_value']:.4f} | "
                    f"std_mean={logs['std_mean']:.4f}"
                )