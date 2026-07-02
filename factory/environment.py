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

        # Episode state (set by reset())
        self.current_step = 0
        self.x_orig = self.x_t = self.z_orig = self.d_t = None

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
            direction_scale = 1.75 if float(self.c) == 1.0 else 1.0
            R_prog = (self.d_t - d_next).item()
            R_val  = -d_next.item()
            R_close2 = -(d_next.item() ** 2)
            R_fid  = -F.mse_loss(x_next, self.x_orig).item()

            S_x    = self.generator._get_seg_mask(self.x_orig)
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

        total = (
            config.LAM_PROG   * (direction_scale * R_prog)
            + config.LAM_VAL  * (direction_scale * R_val)
            + config.LAM_CLOSE2 * (direction_scale * R_close2)
            + config.LAM_FID  * R_fid
            + config.LAM_PRES_R * R_pres
            + config.LAM_BUD  * R_bud
            + config.LAM_HIT  * (direction_scale * R_hit)
            + config.LAM_STOP * (direction_scale * R_stop)
        )
        return total, is_valid, dict(
            R_prog=R_prog, R_val=R_val, R_close2=R_close2, R_fid=R_fid,
            R_pres=R_pres, R_bud=R_bud, R_hit=R_hit, R_stop=R_stop,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Gym interface
    # ──────────────────────────────────────────────────────────────────────────

    def reset(self) -> torch.Tensor:
        self.current_step = 0
        self.x_orig = self._sample_query_image()
        self.x_t    = self.x_orig.clone()
        with torch.no_grad():
            self.z_orig = self.generator.encoder(self.x_orig)
            x3_orig     = (
                self.x_orig.repeat(1, 3, 1, 1) if self.x_orig.shape[1] == 1 else self.x_orig
            )
            p_orig   = self.f(x3_orig)
            self.d_t = torch.abs(p_orig - self.c_tensor)
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
        n_passes = config.ENV_REFINEMENT_PASSES + extra_pass_1 + extra_pass_2

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
                    / float(guidance_bias.item())
                    * (1.0 - 0.15 * pass_idx)
                )
                x_next, _ = self.generator.generate_counterfactual(
                    x_next,
                    self.c,
                    M=1,
                    strength=pass_strength,
                    cls_guidance_scale=pass_cls_scale,
                    pres_guidance_scale=pass_pres_scale,
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
            "n_passes": n_passes,
            "valid": is_valid,
        })
        return self.get_observation(), reward, done, info
