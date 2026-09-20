"""Seven-action local 3D logit refinement; no ground truth in control flow."""
from __future__ import annotations

import gymnasium as gym
import numpy as np
from scipy import ndimage as ndi
from scipy.special import softmax

from .metrics import region_metrics


def class_boundary(mask):
    edge = np.zeros(mask.shape, dtype=bool)
    for axis in range(3):
        left, right = [slice(None)] * 3, [slice(None)] * 3
        left[axis], right[axis] = slice(None, -1), slice(1, None)
        different = mask[tuple(left)] != mask[tuple(right)]
        edge[tuple(left)] |= different
        edge[tuple(right)] |= different
    return edge


def patch_slices(center, shape, patch_size):
    starts = [max(0, min(int(c) - patch_size // 2, s - patch_size))
              for c, s in zip(center, shape)]
    return tuple(slice(a, min(a + patch_size, s)) for a, s in zip(starts, shape))


class BoundaryEnv(gym.Env):
    """Action 0..5: +/- class 1,2,3 logits; action 6: keep.

    `target` is optional. With reward_enabled=False it is never consulted.
    A finite fixed horizon is an MDP termination (not a time-limit truncation).
    Final masks contain policy edits only; postprocessing is an external stage.
    """
    metadata = {"render_modes": []}

    def __init__(self, cases, patch_size=24, max_steps=16, band_mm=3.0,
                 logit_delta=1.0, hd95_weight=.2, hd95_scale_mm=20.,
                 reward_scale=100., tolerance_mm=2., reward_enabled=True,
                 dsc_drop_penalty=3.0, dsc_boost_multiplier=500.0,
                 reward_clip=None, dsc_floor_ratio=None):
        super().__init__()
        if not cases or patch_size < 8 or max_steps < 1 or band_mm <= 0 or logit_delta <= 0:
            raise ValueError("Invalid environment configuration")
        if hd95_scale_mm <= 0 or reward_scale <= 0 or hd95_weight < 0:
            raise ValueError("Invalid reward configuration")
        if dsc_drop_penalty < 1.0 or dsc_boost_multiplier < 1.0:
            raise ValueError("dsc_drop_penalty and dsc_boost_multiplier must be >= 1.0")
        self.cases, self.patch_size, self.max_steps = cases, patch_size, max_steps
        self.band_mm, self.logit_delta = band_mm, logit_delta
        self.hd95_weight, self.hd95_scale_mm = hd95_weight, hd95_scale_mm
        self.reward_scale, self.tolerance_mm = reward_scale, tolerance_mm
        self.reward_enabled = reward_enabled
        self.dsc_drop_penalty = dsc_drop_penalty
        self.dsc_boost_multiplier = dsc_boost_multiplier
        # V5: reward clipping prevents value function explosion (±reward_clip)
        self.reward_clip = float(reward_clip) if reward_clip is not None else None
        # V5: DSC floor early termination — episode ends with large penalty
        #     if DSC drops below (initial_dsc * dsc_floor_ratio)
        self.dsc_floor_ratio = float(dsc_floor_ratio) if dsc_floor_ratio is not None else None
        self.action_space = gym.spaces.Discrete(7)
        # MRI 4 + initial 4 + current 4 + current boundary distance + edited + budget.
        self.observation_space = gym.spaces.Box(-5., 5., (15, patch_size, patch_size, patch_size), np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        index = (options or {}).get("case_index")
        if index is None:
            index = int(self.np_random.integers(len(self.cases)))
        self.case = self.cases[index]
        self.image = self.case["image"]
        self.initial = softmax(self.case["logits"], axis=0).astype(np.float32)
        self.logits = self.case["logits"].astype(np.float32).copy()
        if self.image.shape[0] != 4 or self.logits.shape != self.image.shape:
            raise ValueError("Expected image and logits with shape (4, X, Y, Z)")
        self.mask = self.logits.argmax(0).astype(np.uint8)
        self.edited = np.zeros(self.mask.shape, np.float32)
        boundary = class_boundary(self.mask)
        self.band = (ndi.distance_transform_edt(~boundary, sampling=self.case["spacing"]) <= self.band_mm
                     if boundary.any() else np.zeros_like(boundary))
        points = np.argwhere(boundary)
        if not len(points):
            points = np.asarray([np.asarray(self.mask.shape) // 2])
        # Fixed deterministic spatial coverage from prediction only.
        if len(points) > 2048:
            points = points[np.linspace(0, len(points) - 1, 2048).astype(int)]
        physical = points * np.asarray(self.case["spacing"])
        uncertainty = 1. - self.initial.max(0)[tuple(points.T)]
        selected, distances = [], np.full(len(points), np.inf)
        current = int(uncertainty.argmax())
        for _ in range(min(self.max_steps, len(points))):
            selected.append(points[current])
            distances = np.minimum(distances, ((physical - physical[current]) ** 2).sum(1))
            distances[current] = -1
            current = int(distances.argmax())
        self.candidates = selected
        self.t, self.done = 0, False
        self.current_metrics = self.measure() if self.reward_enabled else None
        # V5: record initial DSC for DSC floor constraint
        self.initial_dsc = float(self.current_metrics["mean"]["dice"]) if self.current_metrics is not None else None
        return self._observation(), {"patient": self.case["patient"], "candidates": len(selected)}

    def measure(self, mask=None):
        if "target" not in self.case:
            raise ValueError("Labels required only for training rewards / diagnostic metrics")
        return region_metrics(self.mask if mask is None else mask, self.case["target"],
                              self.case["spacing"], self.tolerance_mm,
                              self.case.get("empty_distance_mm"))

    def _observation(self):
        if self.done:
            return np.zeros(self.observation_space.shape, np.float32)
        slices = patch_slices(self.candidates[self.t], self.mask.shape, self.patch_size)
        margin = 24
        exp_slices = tuple(slice(max(0, s.start - margin), min(dim, s.stop + margin))
                           for s, dim in zip(slices, self.mask.shape))
        mask_sub = self.mask[exp_slices]
        edge_sub = class_boundary(mask_sub)
        dist_sub = (np.minimum(ndi.distance_transform_edt(~edge_sub, sampling=self.case["spacing"]), 20.) / 20.
                    if edge_sub.any() else np.ones(mask_sub.shape))
        rel_slices = tuple(slice(s.start - exp.start, s.stop - exp.start) for s, exp in zip(slices, exp_slices))
        obs = np.concatenate((self.image[(slice(None),) + slices],
                              self.initial[(slice(None),) + slices],
                              softmax(self.logits[(slice(None),) + slices], axis=0),
                              dist_sub[rel_slices][None], self.edited[slices][None],
                              np.full((1,) + self.mask[slices].shape,
                                      (len(self.candidates) - self.t) / len(self.candidates))))
        padding = [(0, 0)] + [(0, self.patch_size - n) for n in obs.shape[1:]]
        return np.pad(obs, padding).astype(np.float32)

    def _apply(self, action):
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")
        slices = patch_slices(self.candidates[self.t], self.mask.shape, self.patch_size)
        old = self.mask[slices].copy()
        if action != 6:
            channel, sign = 1 + int(action) // 2, 1 if int(action) % 2 == 0 else -1
            self.logits[(channel,) + slices] += sign * self.logit_delta * self.band[slices]
            self.edited[slices] = np.maximum(self.edited[slices], self.band[slices])
            self.mask[slices] = self.logits[(slice(None),) + slices].argmax(0).astype(np.uint8)
        return int(np.count_nonzero(old != self.mask[slices]))

    def reward_breakdown(self, before, after):
        """Return the shared training/oracle reward and its observable diagnostics.

        A DSC decrease receives an additional penalty. This mitigates DSC/HD95
        tradeoffs but does not guarantee preservation of DSC or either region.
        With penalty > 1 the episode return includes all intermediate DSC drops;
        it is no longer just a scaled difference of final and initial quality.
        Metrics belong only to training/diagnostic info, never policy inputs.
        """
        delta_dsc = float(after["mean"]["dice"] - before["mean"]["dice"])
        hd95_reduction = float(before["mean"]["hd95_mm"] - after["mean"]["hd95_mm"])
        clipped_hd95_reduction = max(-15.0, min(15.0, hd95_reduction))
        extra_penalty = self.reward_scale * (self.dsc_drop_penalty - 1.) * max(-delta_dsc, 0.)
        extra_bonus = self.reward_scale * (self.dsc_boost_multiplier - 1.) * max(delta_dsc, 0.)
        dsc_reward = self.reward_scale * delta_dsc - extra_penalty + extra_bonus
        hd95_reward = self.reward_scale * clipped_hd95_reduction * self.hd95_weight / self.hd95_scale_mm
        return dict(delta_dsc=delta_dsc, hd95_reduction_mm=hd95_reduction,
                    dsc_reward=float(dsc_reward), hd95_reward=float(hd95_reward),
                    dsc_drop_extra_penalty=float(extra_penalty),
                    reward=float(dsc_reward + hd95_reward))

    def _asymmetric_reward(self, before, after):
        """Compatibility wrapper for the shared reward calculation."""
        return self.reward_breakdown(before, after)["reward"]

    def step(self, action):
        if self.done:
            raise RuntimeError("Call reset after episode termination")
        changed = self._apply(action)
        reward = 0.
        info = dict(patient=self.case["patient"], action=int(action),
                    changed_voxels=changed, step=self.t + 1)
        if self.reward_enabled:
            after = self.measure() if changed else self.current_metrics
            breakdown = self.reward_breakdown(self.current_metrics, after)
            reward = breakdown.pop("reward")
            info.update(breakdown)
            self.current_metrics = after
            # V5: DSC floor early termination — penalise & stop if DSC collapses
            if (self.dsc_floor_ratio is not None and self.initial_dsc is not None
                    and self.initial_dsc > 0):
                current_dsc = float(after["mean"]["dice"])
                floor = self.initial_dsc * self.dsc_floor_ratio
                if current_dsc < floor:
                    reward += -abs(self.reward_scale) * 5.0  # large floor-breach penalty
                    info["dsc_floor_breach"] = True
                    self.done = True
            # V5: hard reward clipping — prevents value function explosion
            if self.reward_clip is not None:
                reward = max(-self.reward_clip, min(self.reward_clip, reward))
        self.t += 1
        if not self.done:
            self.done = self.t >= len(self.candidates)
        return self._observation(), float(reward), self.done, False, info

    def action_gains(self):
        """Diagnostic/training oracle only; does not mutate the episode state."""
        if self.done:
            raise RuntimeError("Call reset after episode termination")
        baseline = self.measure()
        gains, changed = [], []
        slices = patch_slices(self.candidates[self.t], self.mask.shape, self.patch_size)
        old_logits = self.logits[(slice(None),) + slices].copy()
        old_mask, old_edited = self.mask[slices].copy(), self.edited[slices].copy()
        try:
            for action in range(7):
                count = self._apply(action)
                score = self.measure() if count else baseline
                gains.append(self._asymmetric_reward(baseline, score))
                changed.append(count)
                self.logits[(slice(None),) + slices] = old_logits
                self.mask[slices], self.edited[slices] = old_mask, old_edited
        finally:
            self.logits[(slice(None),) + slices] = old_logits
            self.mask[slices], self.edited[slices] = old_mask, old_edited
        return np.asarray(gains, np.float32), changed
