#!/usr/bin/env python3
"""
gail_wrapper.py — Environment wrapper and network classes for GAIL training.
===========================================================================

Provides a flat Box observation wrapper for Gymnasium to avoid ReplayBuffer crashes
in the imitation library, while keeping CNN feature extraction internally in PyTorch.
"""

import numpy as np
import torch as th
import torch.nn as nn
import gymnasium as gym
from gymnasium import spaces

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from imitation.rewards.reward_nets import RewardNet

# Observation dimensions
IMG_SIZE = 128
IMG_DIM = IMG_SIZE * IMG_SIZE  # 16384
VEC_DIM = 1 + 7                # force (1) + pose (7) = 8
TOTAL_DIM = IMG_DIM + VEC_DIM  # 16392


class FlattenMultiInputWrapper(gym.ObservationWrapper):
    """Wraps RoboticUltrasoundGymEnv to output a flat Box observation.

    Avoids ReplayBuffer failures in imitation library GAIL/AIRL algorithms.
    """
    def __init__(self, env):
        super().__init__(env)
        # Redefine observation space as a flat float32 Box
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(TOTAL_DIM,),
            dtype=np.float32
        )

    def observation(self, obs):
        # Flatten image and normalize to [0, 1]
        image_flat = obs["image"].flatten().astype(np.float32) / 255.0
        # Format force and pose
        force = obs["force"].astype(np.float32)
        pose = obs["pose"].astype(np.float32)
        # Concatenate into a single flat vector
        return np.concatenate([image_flat, force, pose])


class CustomFlatFeatureExtractor(BaseFeaturesExtractor):
    """Feature Extractor for the PPO Generator policy.

    Reconstructs the CNN+MLP architecture from the flat 1D observation vector.
    """
    def __init__(self, observation_space, features_dim=256):
        super().__init__(observation_space, features_dim)
        
        # 1. CNN for image component (input: 1x128x128)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=8, stride=4),   # 16x31x31
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),  # 32x14x14
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1),  # 32x12x12
            nn.ReLU(),
            nn.Flatten(),
        )
        cnn_out_dim = 4608

        # 2. MLP for vector components (force + pose = 8-dim)
        self.vector_mlp = nn.Sequential(
            nn.Linear(VEC_DIM, 32),
            nn.ReLU(),
        )
        
        # 3. Final projection
        self.linear = nn.Sequential(
            nn.Linear(cnn_out_dim + 32, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        # Split flat input: (batch_size, 16392)
        img = observations[:, :IMG_DIM].view(-1, 1, IMG_SIZE, IMG_SIZE)
        vec = observations[:, IMG_DIM:]
        
        img_feats = self.cnn(img)
        vec_feats = self.vector_mlp(vec)
        
        combined = th.cat([img_feats, vec_feats], dim=1)
        return self.linear(combined)


class FlatRewardNet(RewardNet):
    """Custom Reward Network (Discriminator) for GAIL.

    Reconstructs the image and vector component flows to predict adversarial rewards.
    """
    def __init__(self, observation_space, action_space):
        super().__init__(observation_space, action_space)
        
        # 1. CNN for image component
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        cnn_out_dim = 4608

        # 2. MLP for vector components
        self.vector_mlp = nn.Sequential(
            nn.Linear(VEC_DIM, 32),
            nn.ReLU(),
        )

        # 3. MLP for continuous actions (6-dim)
        self.action_mlp = nn.Sequential(
            nn.Linear(6, 32),
            nn.ReLU(),
        )

        # 4. Joint projection head to output scalar reward
        total_features = cnn_out_dim + 32 + 32
        self.reward_head = nn.Sequential(
            nn.Linear(total_features, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, state, action, next_state, done) -> th.Tensor:
        # Split state
        img = state[:, :IMG_DIM].view(-1, 1, IMG_SIZE, IMG_SIZE)
        vec = state[:, IMG_DIM:]
        
        img_feats = self.cnn(img)
        vec_feats = self.vector_mlp(vec)
        act_feats = self.action_mlp(action.float())
        
        combined = th.cat([img_feats, vec_feats, act_feats], dim=1)
        reward = self.reward_head(combined)
        return reward.squeeze(-1)
