"""Single PPO mask-refiner observation and discrete action contract."""
import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn
from src.envs.mask_refinement_env import MaskRefinementEnv
from src.envs import torch_ops as tops

MODES = ('small', 'medium', 'large')

class PPOMaskRefinementEnv(MaskRefinementEnv):
    def __init__(self, *args, **kwargs):
        kwargs.update(enable_stop=False, use_zoom_obs=True, zoom_size=64)
        kwargs.setdefault('seg_bias', 'under')
        kwargs.setdefault('fp_penalty_ratio', 4.0)
        kwargs.setdefault('fn_penalty_ratio', 2.0)
        kwargs.setdefault('expand_prob_thr', 0.60)
        kwargs.setdefault('shrink_block_prob_thr', 0.70)
        super().__init__(*args, **kwargs)
        self.action_space = spaces.MultiDiscrete([5] * 8)
        self._obs_ch = self.img_ch + 6
        self.observation_space = spaces.Dict({
            'image': spaces.Box(0, 1, (self._obs_ch, 64, 64), np.float32),
            'mode': spaces.Discrete(3),
        })

    def _obs(self):
        mask = self._mask.clamp(0, 1)
        prob = self._prob.clamp(0, 1)
        uncert = 1 - (2 * prob - 1).abs()
        band = self._predicted_action_band(mask > .5).float()
        # Candidates, never ground-truth errors: identical at train and deployment.
        fp = mask * (1 - prob) * band
        fn = (1 - mask) * prob * band
        sdf = (tops.signed_distance(mask > .5).clamp(-16, 16) + 16) / 32
        obs = torch.cat((self._image.clamp(0, 1), torch.stack((prob, mask, uncert, fp, fn, sdf))))
        crop = self._crop_zoom(obs, int(self._zoom_cy), int(self._zoom_cx))
        return {'image': crop.detach().cpu().numpy().astype(np.float32),
                'mode': MODES.index(self.refinement_mode)}

    def step(self, action):
        action = np.asarray(action, dtype=np.int64)
        if action.shape != (8,) or np.any((action < 0) | (action > 4)):
            raise ValueError('PPO mask refiner requires eight actions in [0, 4]')
        action = self._candidate_gate_actions(action)
        # Small's existing SDF environment consumes signed continuous shifts.
        return super().step(action.astype(np.float32) - 2 if self.refinement_mode == 'small' else action)

    def _candidate_gate_actions(self, action):
        mask = self._mask > .5
        if not bool(mask.any()):
            return action
        cy, cx = tops.largest_component_centroid(self._mask)
        angles = torch.atan2(self._YY - cy, self._XX - cx)
        sectors = ((angles + np.pi) / (2.0 * np.pi) * 8.0).long().clamp(0, 7)
        band = self._predicted_action_band(mask)
        if self.use_zoom_obs:
            band = band & self._zoom_window
        prob = self._prob.clamp(0, 1)
        fp_cand = mask.float() * (1 - prob) * band.float()
        fn_cand = (~mask).float() * prob * band.float()
        gated = action.copy()
        for i in range(8):
            sel = (sectors == i) & band
            if not bool(sel.any()):
                gated[i] = 2
                continue
            fp_score = float(fp_cand[sel].mean().item())
            fn_score = float(fn_cand[sel].mean().item())
            if gated[i] >= 3 and fn_score < max(0.08, 1.35 * fp_score):
                gated[i] = 2
            elif gated[i] <= 1 and fp_score < max(0.04, 0.80 * fn_score):
                gated[i] = 2
        return gated

class ClassConditionedFeatures(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=128):
        super().__init__(observation_space, features_dim)
        channels = observation_space['image'].shape[0]
        self.encoder = nn.Sequential(nn.Conv2d(channels, 32, 5, 2), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.embedding = nn.Embedding(3, 16)
        self.film = nn.Linear(16, 128)
        self.output = nn.Sequential(nn.Linear(64, features_dim), nn.ReLU())

    def forward(self, observations):
        mode = observations['mode'].reshape(observations['image'].shape[0], -1)
        ids = mode.argmax(-1) if mode.ndim > 1 and mode.shape[-1] == 3 else mode.long().reshape(-1)
        gamma, beta = self.film(self.embedding(ids)).chunk(2, dim=-1)
        return self.output(self.encoder(observations['image']) * (1 + gamma) + beta)
