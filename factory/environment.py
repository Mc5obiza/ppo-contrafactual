"""
src/environment.py
==================
Gym-style PPO environment for medical counterfactual generation.

The environment:
  - Samples a "far" image (one whose classifier output differs significantly
    from the target class c) at the start of each episode.
  - At each step, calls the generator to produce a counterfactual and computes
    a shaped reward.
  - Terminates when the counterfactual is valid (|f(x_cf) - c| < ε) or the
    budget is exhausted.

CHANGE LOG
----------
- _compute_reward: `direction_scale` (2.25x boost for the harder target_c=1.0
  / n->p direction) previously scaled ONLY the "push toward target" reward
  terms (R_prog, R_val, R_close2, R_hit, R_stop) and left R_fid / R_pres
  unscaled. This made preservation ~2.25x *less* important relative to
  progress specifically on the direction that already struggled most,
  which is the direct mechanism behind diffuse, whole-image n->p edits
  (81% of pixels changed, foreign-object corruption) observed in
  qualitative diff plots. Fixed by applying direction_scale uniformly to
  ALL reward terms, preserving the push/preserve ratio across both
  directions.
  NOTE: under PPO+GAE's advantage normalization (adv is renormalized to
  zero mean / unit std per update in compute_gae), a uniform scalar
  multiplying the whole reward washes out almost entirely — so this fix
  is close to equivalent to training as if direction_scale were always
  1.0. That's fine (it removes the *relative* re-weighting that caused
  the bug), but don't expect direction_scale to still give the n->p
  direction extra learning signal the way it originally intended to.
  If the harder direction still lags after retraining, address it via
  training hyperparameters (more episodes, lower entropy coef, etc.) for
  that policy specifically, not via reward scaling.
- step(): `guidance_bias` (action[7]) previously multiplied cls_scale and
  divided pres_scale by the same factor with no floor. Added a floor on
  the suppression factor (1/guidance_bias) and per-pass decay. NOTE: given
  guidance_bias's actual range (0.70, 1.30) from its sigmoid transform,
  1/guidance_bias only ranges (0.769, 1.428) and the per-pass decay only
  reaches 0.55 at pass_idx=3 — so the 0.6/0.4 floors below never actually
  bind given the current action-space bounds. They're left in as a safety
  margin against future action-range changes, but the real constraint
  today is the action transform itself. If you want a floor that actually
  matters, tighten it above the natural minimum (e.g. min=1.0 to prevent
  guidance_bias from reducing pres_scale at all) or widen guidance_bias's
  sigmoid range and re-add a binding floor.
  Both are logged in the info dict (`pres_suppression_factor`) for
  diagnostics — see main.py's run_guidance_diagnostics.
- Added generate_via_policy(): drives this env's reward/guidance logic
  using a TRAINED policy's actions on an arbitrary caller-supplied query
  image, instead of only supporting PPO's own rollout sampling from
  self._far_indices. This is what lets evaluation (metrics.py,
  diff_viz.py) actually reflect the trained PPO policy instead of the
  generator's hardcoded default schedule. Wrapped in policy.eval()/
  policy.train() (harmless today since PPOActorCritic has no dropout/
  batchnorm, but correct practice and future-proof) inside a try/finally
  so eval mode isn't left stuck on if generation raises partway through.
- _compute_reward: caches the fixed-for-the-episode segmentation mask
  S_x = generator._get_seg_mask(x_orig) instead of recomputing it on
  every single step of every episode (x_orig never changes mid-episode).
  Computed once in reset() / generate_via_policy() and reused in
  _compute_reward().
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import config


class MedicalCounterfactualEnv:
    """
    PPO environment for iterative counterfactual refinement.

    Parameters
    ----------
    dataset : torch.utils.data.Dataset
    classifier : nn.Module
    target_c : float
    generator : PretrainedLatentDiffusionGenerator
    seg_net : nn.Module
    obj_det : nn.Module
    budget_T : int
    device : str
    far_threshold : float
    scan_batch_size : int
    scan_num_workers : int
    """

    def __init__(
        self,
        dataset,
        classifier,
        target_c: float,
        generator,
        seg_net,
        obj_det,
        budget_T: int = config.BUDGET_T,
        device: str = config.DEVICE,
        far_threshold: float = config.FAR_THRESHOLD,
        scan_batch_size: int = config.ENV_SCAN_BATCH_SIZE,
        scan_num_workers: int = 2,
    ):
        self.dataset   = dataset
        self.f         = classifier
        self.c         = target_c
        self.c_tensor  = torch.tensor([[target_c]], dtype=torch.float32, device=device)
        self.generator = generator
        self.S         = seg_net
        self.O         = obj_det
        self.T         = budget_T
        self.device    = device
        self.far_threshold = far_threshold

        self._far_indices = self._build_far_index_pool(scan_batch_size, scan_num_workers)
        print(
            f"[env] {len(self._far_indices)}/{len(dataset)} images are "
            f"> {far_threshold} away from target_c={target_c}"
        )

        # Episode state (set by reset() / generate_via_policy())
        self.current_step = 0
        self.x_orig = self.x_t = self.z_orig = self.d_t = None
        # Cached per-episode segmentation mask of x_orig, used by
        # _compute_reward. x_orig is fixed for the whole episode, so
        # there's no need to recompute this on every step.
        self._S_x_orig = None

    # ──────────────────────────────────────────────────────────────────────────
    # Initialisation helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _build_far_index_pool(self, batch_size: int, num_workers: int) -> list[int]:
        """
        Pre-scan the dataset and collect indices of images whose classifier
        output is far from the target class.
        """
        far_indices = []
        loader = DataLoader(
            self.dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
        base_idx = 0
        with torch.no_grad():
            for x_batch, _labels in loader:
                x_batch = x_batch.to(self.device)
                x3      = x_batch.repeat(1, 3, 1, 1) if x_batch.shape[1] == 1 else x_batch
                p_batch = self.f(x3).squeeze(-1)
                diffs   = (p_batch - self.c).abs()
                mask    = diffs > self.far_threshold
                hits    = torch.nonzero(mask, as_tuple=True)[0]
                far_indices.extend((base_idx + hits).tolist())
                base_idx += x_batch.shape[0]

        if not far_indices:
            raise RuntimeError(
                f"No images with |f(x) - {self.c}| > {self.far_threshold}. "
                f"Lower far_threshold and try again."
            )
        return far_indices

    def _sample_query_image(self) -> torch.Tensor:
        idx = self._far_indices[
            torch.randint(0, len(self._far_indices), (1,)).item()
        ]
        x, _label = self.dataset[idx]
        return x.unsqueeze(0).to(self.device)

    def _init_episode_state(self, x_orig: torch.Tensor) -> None:
        """
        Shared initialization logic for a new episode, given a query image.
        Used by both reset() (dataset-sampled x_orig) and
        generate_via_policy() (externally-supplied x_orig), so the two
        code paths can't drift out of sync.

        Sets: x_orig, x_t, z_orig, d_t, current_step, and caches the
        segmentation mask of x_orig for reuse in _compute_reward.
        """
        self.current_step = 0
        self.x_orig = x_orig.to(self.device)
        self.x_t    = self.x_orig.clone()
        with torch.no_grad():
            self.z_orig = self.generator.encoder(self.x_orig)
            x3_orig     = (
                self.x_orig.repeat(1, 3, 1, 1) if self.x_orig.shape[1] == 1 else self.x_orig
            )
            p_orig   = self.f(x3_orig)
            self.d_t = torch.abs(p_orig - self.c_tensor)
            # Cache the segmentation mask once per episode — x_orig is
            # fixed for the whole episode, so _compute_reward doesn't
            # need to recompute this on every single step.
            self._S_x_orig = self.generator._get_seg_mask(self.x_orig)

    # ──────────────────────────────────────────────────────────────────────────
    # Observation
    # ──────────────────────────────────────────────────────────────────────────

    def get_observation(self) -> torch.Tensor:
        """
        Concatenate [z_orig, z_t, step/T, d_t, c, mask] into a flat vector.
        """
        with torch.no_grad():
            z_t = self.generator.encoder(self.x_t)
            m_t = self.generator._get_seg_mask(self.x_t).flatten()
        s_t = torch.tensor([self.current_step / self.T], device=self.device)
        obs = torch.cat([
            self.z_orig.flatten(), z_t.flatten(),
            s_t, self.d_t.flatten(), self.c_tensor.flatten(), m_t,
        ])
        return obs

    # ──────────────────────────────────────────────────────────────────────────
    # Reward
    # ──────────────────────────────────────────────────────────────────────────

    def _compute_reward(
        self,
        x_next: torch.Tensor,
        d_next: torch.Tensor,
        tau_t: float,
    ) -> tuple[float, bool, dict]:
        with torch.no_grad():
            direction_scale = 2.25 if float(self.c) == 1.0 else 1.0

            R_prog   = (self.d_t - d_next).item()
            R_val    = -d_next.item()
            R_close2 = -(d_next.item() ** 2)
            R_fid    = -F.mse_loss(x_next, self.x_orig).item()

            # Use the cached per-episode mask instead of recomputing it
            # every step (x_orig — and therefore its mask — never changes
            # within an episode).
            S_x    = self._S_x_orig
            l1_map = torch.abs(x_next - self.x_orig)
            seg_term = (S_x * l1_map).sum() / (S_x.sum() + 1e-8)

            x3_orig = self.x_orig.repeat(1, 3, 1, 1) if self.x_orig.shape[1] == 1 else self.x_orig
            x3_next = x_next.repeat(1, 3, 1, 1)  if x_next.shape[1] == 1  else x_next
            O_x     = self.O(x3_orig)
            O_next  = self.O(x3_next)
            p_O  = F.softmax(O_x,    dim=-1).clamp(1e-8, 1.0)
            q_O  = F.softmax(O_next, dim=-1).clamp(1e-8, 1.0)
            kl_obj   = (p_O * (p_O / q_O).log()).sum()
            R_pres   = -(seg_term + config.ETA_OBJ * kl_obj).item()

            R_bud    = -1.0
            is_valid = d_next.item() < config.EPSILON_STOP
            R_hit    = float(is_valid)
            R_stop   = float(is_valid) if tau_t > 0.5 else 0.0

        # direction_scale is applied uniformly to ALL reward terms (see
        # module-level CHANGE LOG note above for why this differs from
        # the original, buggy, partially-scaled version — and why it's
        # close to a no-op under GAE's advantage normalization).
        total = direction_scale * (
            config.LAM_PROG     * R_prog
            + config.LAM_VAL    * R_val
            + config.LAM_CLOSE2 * R_close2
            + config.LAM_FID    * R_fid
            + config.LAM_PRES_R * R_pres
            + config.LAM_BUD    * R_bud
            + config.LAM_HIT    * R_hit
            + config.LAM_STOP   * R_stop
        )
        return total, is_valid, dict(
            R_prog=R_prog, R_val=R_val, R_close2=R_close2, R_fid=R_fid,
            R_pres=R_pres, R_bud=R_bud, R_hit=R_hit, R_stop=R_stop,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Gym interface
    # ──────────────────────────────────────────────────────────────────────────

    def reset(self) -> torch.Tensor:
        self._init_episode_state(self._sample_query_image())
        return self.get_observation()

    def step(self, action: torch.Tensor) -> tuple[torch.Tensor, float, bool, dict]:
        self.current_step += 1
        strength_t = 0.2 + 0.5 * torch.sigmoid(action[0])
        cls_scale_t = 0.5 + 2.5 * torch.sigmoid(action[1])
        pres_scale_t = 0.3 + 1.2 * torch.sigmoid(action[2])
        tau_t = torch.sigmoid(action[3])
        extra_pass_1 = int(torch.sigmoid(action[4]).item() > 0.5)
        extra_pass_2 = int(torch.sigmoid(action[5]).item() > 0.5)
        strength_boost = 0.85 + 0.30 * torch.sigmoid(action[6])
        guidance_bias = 0.70 + 0.60 * torch.sigmoid(action[7])
        # 9th action dim: where guidance starts ramping up, as a fraction
        # of the trajectory. Mapped to [0.1, 0.5] — never lets guidance
        # start in the first 10% (too early risks the same kind of noisy
        # collapse the ramp was meant to fix) or later than the midpoint
        # (leaves too few steps for guidance to have any effect).
        ramp_start_frac_t = 0.1 + 0.4 * torch.sigmoid(action[8])
        n_passes = config.ENV_REFINEMENT_PASSES + extra_pass_1 + extra_pass_2

        # Floor on how much guidance_bias is allowed to suppress
        # preservation. NOTE (see module CHANGE LOG): given
        # guidance_bias's actual range (0.70, 1.30), this floor does not
        # currently bind — 1/guidance_bias never goes below ~0.769. Left
        # in as a safety margin; see the note above for what to change if
        # you want it to actually constrain behavior.
        pres_suppression_floor = 0.6
        inv_guidance_bias = torch.clamp(1.0 / guidance_bias, min=pres_suppression_floor)

        with torch.no_grad():
            x_next = self.x_t
            for pass_idx in range(n_passes):
                pass_strength = min(
                    0.9,
                    float(strength_t.item()) * float(strength_boost.item()) * (1.0 + 0.15 * pass_idx),
                )
                pass_cls_scale = (
                    float(cls_scale_t.item())
                    * float(guidance_bias.item())
                    * (1.0 + 0.35 * pass_idx)
                )
                pass_pres_scale = (
                    float(pres_scale_t.item())
                    * float(inv_guidance_bias.item())
                    * max(1.0 - 0.15 * pass_idx, 0.4)  # per-pass decay floor
                )
                x_next, _ = self.generator.generate_counterfactual(
                    x_next,
                    self.c,
                    M=1,
                    strength=pass_strength,
                    cls_guidance_scale=pass_cls_scale,
                    pres_guidance_scale=pass_pres_scale,
                    guidance_ramp_start_frac=float(ramp_start_frac_t.item()),
                )

            x3_next   = x_next.repeat(1, 3, 1, 1) if x_next.shape[1] == 1 else x_next
            p_next    = self.f(x3_next)
            d_next    = torch.abs(p_next - self.c_tensor)

        reward, is_valid, info = self._compute_reward(x_next, d_next, tau_t.item())
        self.x_t = x_next.detach()
        self.d_t = d_next.detach()
        done = bool(is_valid or self.current_step >= self.T)
        info.update({
            "x_t": self.x_t,
            "tau": tau_t.item(),
            "strength": float(strength_t.item()),
            "cls_scale": float(cls_scale_t.item()),
            "pres_scale": float(pres_scale_t.item()),
            "strength_boost": float(strength_boost.item()),
            "guidance_bias": float(guidance_bias.item()),
            "ramp_start_frac": float(ramp_start_frac_t.item()),
            "n_passes": n_passes,
            "valid": is_valid,
            "pres_suppression_factor": float(inv_guidance_bias.item()),
        })
        return self.get_observation(), reward, done, info

    # ──────────────────────────────────────────────────────────────────────────
    # Policy-guided inference
    # ──────────────────────────────────────────────────────────────────────────

    def generate_via_policy(
        self,
        x_orig: torch.Tensor,
        policy,
        deterministic: bool = True,
        max_steps: int | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor], dict]:
        """
        Generate a counterfactual for an arbitrary query image using a
        TRAINED policy's actions, instead of generator.generate_counterfactual's
        hardcoded default strength/guidance schedule.

        This is the missing link between PPO training and evaluation: prior
        to this, run_full_evaluation() and diff_viz called
        generator.generate_counterfactual directly with default args, so the
        reported CV/FID/FOP numbers and diff plots never actually reflected
        what the trained PPO policy does. This method steps the policy the
        same way collect_rollout() does, but on a caller-supplied image
        rather than one sampled from self._far_indices.

        Parameters
        ----------
        x_orig        : (1, 1, H, W) query image, NOT necessarily from this
                         env's dataset (e.g. a held-out eval image)
        policy        : trained PPOActorCritic. Must match this env's target_c
                         — i.e. use the n→p env with a policy trained on
                         target_c=1.0, and the p→n env with target_c=0.0
        deterministic : use the policy's mean action (recommended for eval)
                         rather than sampling, for reproducible results
        max_steps     : override self.T if provided (e.g. to allow a longer
                         budget at eval time than was used during training).
                         NOTE: get_observation()'s step-fraction feature
                         (current_step / self.T) is computed against
                         self.T, not max_steps. Passing a max_steps that
                         differs from self.T means the policy will see
                         step-fraction values it never saw during training
                         (e.g. never reaching ~1.0 near termination, or
                         exceeding 1.0). If you need a different budget at
                         eval time, prefer temporarily setting self.T to
                         that value (and restoring it afterward) over
                         passing max_steps, unless you're deliberately
                         testing generalization to an out-of-distribution
                         budget.

        Returns
        -------
        (x_cf, trajectory, info) :
            x_cf       : final counterfactual image, (1, 1, H, W)
            trajectory : list of x_t at each step (for plot_trajectory)
            info       : last step's info dict, plus 'n_steps_used' and
                         'converged' (whether it stopped via validity rather
                         than budget exhaustion)
        """
        budget = max_steps if max_steps is not None else self.T

        self._init_episode_state(x_orig)

        trajectory = [self.x_t.clone()]
        info = {}
        done = False
        step_count = 0

        policy.eval()
        try:
            with torch.no_grad():
                while not done and step_count < budget:
                    obs = self.get_observation()
                    action, _log_prob, _value, _feas, _prog, _stop = (
                        policy.get_action_and_value(obs.unsqueeze(0), deterministic=deterministic)
                    )
                    action = action.squeeze(0)
                    _obs_next, _reward, done, info = self.step(action)
                    trajectory.append(self.x_t.clone())
                    step_count += 1
        finally:
            policy.train()

        info["n_steps_used"] = step_count
        info["converged"]    = bool(info.get("valid", False))
        return self.x_t, trajectory, info