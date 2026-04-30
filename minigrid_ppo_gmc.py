"""
MiniGrid PPO with Curiosity-Driven Exploration
===============================================

Supplementary code for "Measuring Learning Progress via Gradient-Momentum
Correlation". This script implements the MiniGrid reinforcement learning
experiments described in Section 3.2 of the paper.

Implements 4 algorithms:
  1. ppo       -- Vanilla PPO (no intrinsic reward, no dynamics model)
  2. icm       -- PPO + Intrinsic Curiosity Module (ICM)
                  (feature encoder + inverse model + forward model in learned
                  feature space; intrinsic reward = forward prediction error)
  3. ppo_gmc   -- PPO + raw forward dynamics model + GMC intrinsic reward
                  (forward dynamics predicts next raw observation;
                  intrinsic reward = gradient-momentum correlation, Eq. 2)
  4. icm_gmc   -- PPO + ICM architecture + GMC intrinsic reward
                  (same ICM networks trained with standard losses, but
                  intrinsic reward comes from GMC on the ICM forward model
                  instead of raw prediction error)

GMC (Gradient-Momentum Correlation) measures how aligned each sample's
gradient is with the exponential moving average of past gradients on the
dynamics model.  This replaces the raw prediction-error intrinsic reward.

Key design choice:
  backpack(BatchGrad()) is ONLY used when backwarding the dynamics model
  (for GMC computation).  The policy network is never extended / never uses
  backpack.

Noisy-TV observation: a state-dependent noise channel is concatenated to the
observation when the agent is near the door, to test robustness to irrelevant
observation noise (Section 3.2, "Noise Condition").

Usage:
    python minigrid_ppo_gmc.py                  # Run training experiments
    python minigrid_ppo_gmc.py --graph           # Generate graphs from logs
    python minigrid_ppo_gmc.py --simulate        # Train with live visualization

Requires: torch, gymnasium, minigrid, backpack-for-pytorch, numpy, matplotlib.
"""

import sys
import os
import glob
import time
import multiprocessing
import numpy as np
from scipy import stats as scipy_stats
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import gymnasium as gym
import minigrid
from minigrid.wrappers import FullyObsWrapper
import matplotlib
matplotlib.use("Agg")  # non-interactive backend so saving works headless
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from backpack import backpack, extend
from backpack.extensions import BatchGrad

# ============================================================================
# CONFIGURATION
# ============================================================================

# Environment (paper uses MiniGrid-DoorKey-8x8-v0)
ENV_NAME = "MiniGrid-DoorKey-8x8-v0"

# Noisy-TV mode:
# - "none":       No noise channel added (clean observations)
# - "door":       Noise when agent is within Manhattan distance <= 1 of door
NOISY_TV_MODE = "none"
# NOISY_TV_MODE = "door"

FULLY_OBSERVABLE = True   # if True, use fully observable wrapper. If False, use partial obs (agent's view)

# Which algorithms / seeds to run (paper uses all 4 algorithms, seeds 0-9)
EXPERIMENT_CONFIG = {
    "algorithms": ["ppo", "icm", "ppo_gmc", "icm_gmc"],
    # "seeds": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    # "seeds": [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    "seeds": [n for n in range(20)],
}

# PPO hyper-parameters (see Table 3 in the paper)
PPO_CONFIG = {
    "lr_policy":       3e-4,      # Standard learning rate for policy networks
    "lr_value":        1e-3,      # Value network can learn faster
    "gamma":           0.99,      # Standard discount factor for episodic tasks
    "gae_lambda":      0.95,      # GAE parameter for advantage estimation
    "clip_epsilon":    0.2,       # Standard PPO clipping parameter
    "entropy_coef":    0.01,      # Entropy bonus to encourage exploration
    "max_grad_norm":   0.5,       # Gradient clipping for stability
    "ppo_epochs":      4,         # Number of PPO update epochs per rollout
    "mini_batch_size": 256,       # Larger batch for more stable updates
    "rollout_length":  2048,      # Steps to collect before PPO update
    "reward_scaling":  10.0,      # Reward scaling to keep values in a reasonable range for learning
}

# ICM hyper-parameters (see Table 3 in the paper)
ICM_CONFIG = {
    "feature_dim":       64,      # Feature space dimension for ICM encoder
    "lr":                1e-4,    # Learning rate for ICM networks
    "intrinsic_coef":    0.02,    # Weight of intrinsic reward vs extrinsic
    "forward_loss_coef": 0.2,     # Weight for forward model loss
    "inverse_loss_coef": 0.8,     # Weight for inverse model loss (higher to learn good features)
}

# GMC hyper-parameters (see Table 3 in the paper)
GMC_CONFIG = {
    "beta_dot":       (0.990, 0.990),  # Momentum and second-moment decay for GMC
    "intrinsic_coef": 0.02,            # Weight of GMC intrinsic reward (same as ICM for comparability)
    "lr_dynamics":    1e-4,            # Learning rate for dynamics model (same as ICM for comparability)
}

# Training
TRAINING_CONFIG = {
    "total_timesteps": 3_000_000,  # Total environment steps for training
    "log_interval":    1,          # log every N rollouts
    "sim_interval":    2,         # visualize every N rollouts (when --simulate)
}

# Directories
LOG_DIR   = "./logs_mini"
GRAPH_DIR = "./graphs_mini"
os.makedirs(LOG_DIR,   exist_ok=True)
os.makedirs(GRAPH_DIR, exist_ok=True)

# Filename tags derived from current config (used when saving logs and graphs)
_NOISY_TAG = NOISY_TV_MODE.replace("_", "")   # "none", "door", "upperleft"
_FULL_TAG  = "full" if FULLY_OBSERVABLE else "partial"

# Consistent colour palette for algorithms (used by all plotting functions)
_ALGO_COLORS = {
    "ppo":     "#1f77b4",
    "icm":     "#ff7f0e",
    "ppo_gmc": "#2ca02c",
    "icm_gmc": "#d62728",
}


# ============================================================================
# MODELS
# ============================================================================

class PolicyNetwork(nn.Module):
    """CNN policy for MiniGrid.  NOT extended with backpack."""

    def __init__(self, obs_shape, action_dim, hidden_dim=128):
        super().__init__()
        C, H, W = obs_shape
        self.conv = nn.Sequential(
            nn.Conv2d(C, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * H * W, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, obs):
        x = self.conv(obs)
        x = x.reshape(x.size(0), -1)
        return torch.softmax(self.fc(x), dim=-1)


class ValueNetwork(nn.Module):
    """CNN value function for MiniGrid."""

    def __init__(self, obs_shape, hidden_dim=128):
        super().__init__()
        C, H, W = obs_shape
        self.conv = nn.Sequential(
            nn.Conv2d(C, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * H * W, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs):
        x = self.conv(obs)
        x = x.reshape(x.size(0), -1)
        return self.fc(x)


class ForwardDynamicsModel(nn.Module):
    """Predicts next observation from (obs, action) in *raw* observation space.

    Used by ppo_gmc (Section 3.2). Extended with backpack so that BatchGrad
    can produce per-sample gradients for GMC computation (Equation 2).
    """

    def __init__(self, obs_shape, action_dim, hidden_dim=256):
        super().__init__()
        self.obs_flat_dim = obs_shape[0] * obs_shape[1] * obs_shape[2]
        self.net = nn.Sequential(
            nn.Linear(self.obs_flat_dim + action_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, self.obs_flat_dim),
        )

    def forward(self, obs_flat, action_onehot):
        return self.net(torch.cat([obs_flat, action_onehot], dim=1))


class ICMFeatureEncoder(nn.Module):
    """Encodes observations into a learned feature space for ICM."""

    def __init__(self, obs_shape, feature_dim=128):
        super().__init__()
        C, H, W = obs_shape
        self.conv = nn.Sequential(
            nn.Conv2d(C, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
        )
        self.fc = nn.Linear(64 * H * W, feature_dim)

    def forward(self, obs):
        x = self.conv(obs)
        x = x.reshape(x.size(0), -1)
        # Tanh to bound features in [-1, 1], which can help stabilize training of the forward model
        return torch.tanh(self.fc(x))


class ICMInverseModel(nn.Module):
    """Predicts action from (state_features, next_state_features)."""

    def __init__(self, feature_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim * 2, 256), nn.ReLU(),
            nn.Linear(256, action_dim),
        )

    def forward(self, features, next_features):
        return self.net(torch.cat([features, next_features], dim=1))


class ICMForwardModel(nn.Module):
    """Predicts next state features from (state_features, action_onehot).

    When used with icm_gmc, this model is extended with backpack so that
    GMC can be computed on its per-sample gradients.
    """

    def __init__(self, feature_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim + action_dim, 256), nn.ReLU(),
            nn.Linear(256, feature_dim),
        )

    def forward(self, features, action_onehot):
        return self.net(torch.cat([features, action_onehot], dim=1))


class NoisyTVObservation:
    """Concatenates a noise channel to observations based on a checker function.
    
    The checker function determines when to add noise based on the agent's state.
    This can be used to test if the agent can learn to ignore irrelevant observations.
    """

    def __init__(self, checker=None):
        self.checker = checker

    def augment(self, env, obs):
        if self.checker and self.checker(env):
            noise = np.random.randn(*obs.shape[:-1], 1).astype(np.float32)
        else:
            noise = np.zeros((*obs.shape[:-1], 1), dtype=np.float32)
        return np.concatenate([obs, noise], axis=-1)


# ============================================================================
# PPO TRAINER
# ============================================================================

class PPOTrainer:
    """Runs one (algorithm, seed) experiment.

    Supports: ppo, icm, ppo_gmc, icm_gmc.
    """

    # ---- CSV columns (order matters) ----
    LOG_COLUMNS = [
        "timestep", "rollout", "wall_time",
        "ep_reward_mean", "ep_reward_std",
        "ep_length_mean", "ep_length_std",
        "policy_loss", "value_loss", "entropy",
        "clip_fraction", "explained_variance",
        "intrinsic_reward_mean", "intrinsic_reward_std",
        "forward_dynamics_loss", "inverse_dynamics_loss",
        "gmc_mean", "gmc_std",
        "advantage_mean", "advantage_std",
        "extrinsic_reward_mean", "extrinsic_reward_std",
        "entropy_bonus",
    ]

    # ------------------------------------------------------------------ init
    def __init__(self, algorithm, seed, simulate=False):
        assert algorithm in ("ppo", "icm", "ppo_gmc", "icm_gmc"), \
            f"Unknown algorithm: {algorithm}"
        self.algorithm  = algorithm
        self.seed       = seed
        self.simulate   = simulate
        self.use_icm    = algorithm in ("icm", "icm_gmc")
        self.use_gmc    = algorithm in ("ppo_gmc", "icm_gmc")
        self.use_dynamics = algorithm != "ppo"

        np.random.seed(seed)
        torch.manual_seed(seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # --- Environment ---
        self.env = gym.make(ENV_NAME)
        if FULLY_OBSERVABLE:
            self.env = FullyObsWrapper(self.env)

        self.env.reset(seed=seed)
        self.noisy_tv = NoisyTVObservation(checker=_create_noisy_tv_checker(NOISY_TV_MODE))

        obs, _ = self.env.reset()
        obs = self.noisy_tv.augment(self.env, obs["image"])
        self.obs_shape  = (obs.shape[2], obs.shape[0], obs.shape[1])  # CHW
        self.action_dim = self.env.action_space.n
        self.current_obs = obs

        # --- Core networks (always present) ---
        self.policy    = PolicyNetwork(self.obs_shape, self.action_dim).to(self.device)
        self.value_net = ValueNetwork(self.obs_shape).to(self.device)
        self.policy_optimizer = optim.Adam(self.policy.parameters(),
                                           lr=PPO_CONFIG["lr_policy"])
        self.value_optimizer  = optim.Adam(self.value_net.parameters(),
                                           lr=PPO_CONFIG["lr_value"])

        # --- Dynamics models (algorithm-dependent) ---
        self.forward_dynamics    = None   # raw forward model  (ppo_gmc)
        self.dynamics_optimizer  = None
        self.dynamics_loss_fn    = None

        self.icm_encoder  = None          # ICM components     (icm / icm_gmc)
        self.icm_inverse  = None
        self.icm_forward  = None
        self.icm_optimizer = None

        self._init_dynamics_models()

        # --- GMC momentum state ---
        self.gmc_state = {}

        # --- Episode bookkeeping ---
        self.episode_rewards = []
        self.episode_lengths = []
        self._ep_reward = 0.0
        self._ep_length = 0
        self.start_time = time.time()

        # --- Logging ---
        self.log_filename = os.path.join(LOG_DIR, f"minigrid_{algorithm}_{_NOISY_TAG}_{_FULL_TAG}_{seed}.csv")
        # self.log_file = open(self.log_filename, "w")
        self.log_file = open(self.log_filename, "x") # For exclusive lock
        self.log_file.write(",".join(self.LOG_COLUMNS) + "\n")
        self.log_file.flush()

    # ------------------------------------------------ dynamics initialisation
    def _init_dynamics_models(self):
        if self.algorithm == "ppo":
            return  # nothing needed

        if self.algorithm == "ppo_gmc":
            self.forward_dynamics = ForwardDynamicsModel(
                self.obs_shape, self.action_dim).to(self.device)
            extend(self.forward_dynamics)                        # backpack
            self.dynamics_loss_fn = extend(nn.MSELoss(reduction="none"))
            self.dynamics_optimizer = optim.Adam(
                self.forward_dynamics.parameters(),
                lr=GMC_CONFIG["lr_dynamics"])
            return

        # icm  or  icm_gmc
        self.icm_encoder = ICMFeatureEncoder(
            self.obs_shape, ICM_CONFIG["feature_dim"]).to(self.device)
        self.icm_inverse = ICMInverseModel(
            ICM_CONFIG["feature_dim"], self.action_dim).to(self.device)
        self.icm_forward = ICMForwardModel(
            ICM_CONFIG["feature_dim"], self.action_dim).to(self.device)

        if self.use_gmc:
            extend(self.icm_forward)                            # backpack
            self.dynamics_loss_fn = extend(nn.MSELoss(reduction="none"))

        icm_params = (list(self.icm_encoder.parameters()) +
                      list(self.icm_inverse.parameters()) +
                      list(self.icm_forward.parameters()))
        self.icm_optimizer = optim.Adam(icm_params, lr=ICM_CONFIG["lr"])

    # ------------------------------------------------ tensor helpers
    def _obs_to_tensor(self, obs):
        """HWC numpy -> 1xCxHxW float tensor on device."""
        return torch.tensor(
            np.transpose(obs, (2, 0, 1)),
            dtype=torch.float32, device=self.device
        ).unsqueeze(0)

    def _action_onehot(self, actions):
        """Long tensor of action indices -> float one-hot tensor."""
        if actions.dim() == 0:
            actions = actions.unsqueeze(0)
        oh = torch.zeros(actions.size(0), self.action_dim, device=self.device)
        oh.scatter_(1, actions.unsqueeze(1), 1.0)
        return oh

    # ================================================================ ROLLOUT
    def collect_rollout(self):
        obs_list, next_obs_list = [], []
        actions, rewards, values, log_probs, dones = [], [], [], [], []

        for _ in range(PPO_CONFIG["rollout_length"]):
            obs_t = self._obs_to_tensor(self.current_obs)
            with torch.no_grad():
                probs = self.policy(obs_t)
                val   = self.value_net(obs_t)
            dist   = Categorical(probs)
            action = dist.sample()

            obs_list.append(self.current_obs)
            actions.append(action.item())
            values.append(val.item())
            log_probs.append(dist.log_prob(action).item())

            raw_next, reward, terminated, truncated, _ = self.env.step(action.item())
            reward *= PPO_CONFIG["reward_scaling"]
            done    = terminated or truncated
            nxt_obs = self.noisy_tv.augment(self.env, raw_next["image"])

            rewards.append(reward)
            dones.append(float(done))
            next_obs_list.append(nxt_obs)

            self._ep_reward += reward
            self._ep_length += 1
            if done:
                self.episode_rewards.append(self._ep_reward)
                self.episode_lengths.append(self._ep_length)
                self._ep_reward = 0.0
                self._ep_length = 0
                rst, _ = self.env.reset()
                self.current_obs = self.noisy_tv.augment(self.env, rst["image"])
            else:
                self.current_obs = nxt_obs

        with torch.no_grad():
            nxt_val = self.value_net(
                self._obs_to_tensor(self.current_obs)).item()

        return dict(
            observations=obs_list,
            next_observations=next_obs_list,
            actions=actions,
            extrinsic_rewards=rewards,
            values=values,
            log_probs=log_probs,
            dones=dones,
            next_value=nxt_val,
        )

    # ========================================= INTRINSIC REWARD + DYNAMICS
    def compute_intrinsic_and_train_dynamics(self, rollout):
        """Dispatch to the appropriate dynamics handler.

        Returns a dict with keys:
            intrinsic_rewards  - list[float], per-step
            forward_loss       - float
            inverse_loss       - float  (ICM variants only)
            gmc_mean       - float  (GMC variants only)
            gmc_std        - float  (GMC variants only)
        """
        result = dict(
            intrinsic_rewards=[0.0] * len(rollout["actions"]),
            forward_loss=0.0,
            inverse_loss=0.0,
            gmc_mean=0.0,
            gmc_std=0.0,
        )
        if self.algorithm == "ppo":
            return result

        # Build shared tensors
        obs_batch      = torch.stack([self._obs_to_tensor(o).squeeze(0)
                                      for o in rollout["observations"]])
        next_obs_batch = torch.stack([self._obs_to_tensor(o).squeeze(0)
                                      for o in rollout["next_observations"]])
        actions_batch  = torch.tensor(rollout["actions"],
                                      dtype=torch.long, device=self.device)
        action_oh      = self._action_onehot(actions_batch)

        if self.algorithm == "ppo_gmc":
            return self._dynamics_ppo_gmc(
                obs_batch, next_obs_batch, action_oh, result)
        elif self.algorithm == "icm":
            return self._dynamics_icm(
                obs_batch, next_obs_batch, actions_batch, action_oh, result)
        elif self.algorithm == "icm_gmc":
            return self._dynamics_icm_gmc(
                obs_batch, next_obs_batch, actions_batch, action_oh, result)
        return result

    # ---- ppo_gmc: raw forward dynamics + GMC ----
    def _dynamics_ppo_gmc(self, obs_b, nobs_b, act_oh, result):
        obs_flat      = obs_b.reshape(obs_b.size(0), -1)
        next_obs_flat = nobs_b.reshape(nobs_b.size(0), -1)

        self.dynamics_optimizer.zero_grad()
        pred = self.forward_dynamics(obs_flat, act_oh)

        per_sample = self.dynamics_loss_fn(pred, next_obs_flat.detach())
        per_sample = per_sample.mean(dim=1)          # (batch,)
        loss       = per_sample.mean()               # scalar

        with backpack(BatchGrad()):
            loss.backward()

        gmc_rew = self._extract_gmc(
            self.forward_dynamics.parameters(), "raw_dyn"
        ).flatten().detach()

        raw = gmc_rew.cpu().numpy()
        result["intrinsic_rewards"] = (GMC_CONFIG["intrinsic_coef"] * raw).tolist()
        result["forward_loss"]  = loss.item()
        result["gmc_mean"]  = gmc_rew.mean().item()
        result["gmc_std"]   = gmc_rew.std().item()

        nn.utils.clip_grad_norm_(
            self.forward_dynamics.parameters(), PPO_CONFIG["max_grad_norm"])
        self.dynamics_optimizer.step()
        return result

    # ---- icm: standard ICM (prediction-error intrinsic reward) ----
    def _dynamics_icm(self, obs_b, nobs_b, acts, act_oh, result):
        self.icm_optimizer.zero_grad()

        feats      = self.icm_encoder(obs_b)
        next_feats = self.icm_encoder(nobs_b)
        nf_target  = next_feats.detach()

        # inverse model
        pred_act     = self.icm_inverse(feats, next_feats)
        inverse_loss = nn.CrossEntropyLoss()(pred_act, acts)

        # forward model  (detach input features so encoder learns from
        # inverse loss, not from making the forward model's job easier)
        pred_nf        = self.icm_forward(feats.detach(), act_oh)
        fwd_per_sample = 0.5 * nn.MSELoss(reduction="none")(
            pred_nf, nf_target).mean(dim=1)
        fwd_loss       = fwd_per_sample.mean()

        # intrinsic reward = forward prediction error
        raw = fwd_per_sample.detach().cpu().numpy()
        result["intrinsic_rewards"] = (ICM_CONFIG["intrinsic_coef"] * raw).tolist()
        result["forward_loss"]  = fwd_loss.item()
        result["inverse_loss"]  = inverse_loss.item()

        total_loss = (ICM_CONFIG["inverse_loss_coef"] * inverse_loss +
                      ICM_CONFIG["forward_loss_coef"] * fwd_loss)
        total_loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.icm_encoder.parameters()) +
            list(self.icm_inverse.parameters()) +
            list(self.icm_forward.parameters()),
            PPO_CONFIG["max_grad_norm"])
        self.icm_optimizer.step()
        return result

    # ---- icm_gmc: ICM architecture + GMC intrinsic reward ----
    def _dynamics_icm_gmc(self, obs_b, nobs_b, acts, act_oh, result):
        # --- Step 1: GMC intrinsic reward (backpack on forward model) ---
        # Encode with no_grad so only forward-model params get grad_batch
        with torch.no_grad():
            feats_gmc  = self.icm_encoder(obs_b)
            nfeats_gmc = self.icm_encoder(nobs_b)

        # Zero before the backpack pass
        self.icm_optimizer.zero_grad()

        pred_nf = self.icm_forward(feats_gmc, act_oh)
        per_sample = self.dynamics_loss_fn(
            pred_nf, nfeats_gmc.detach()).mean(dim=1)
        loss = per_sample.mean()

        with backpack(BatchGrad()):
            loss.backward()

        gmc_rew = self._extract_gmc(
            self.icm_forward.parameters(), "icm_dyn"
        ).flatten().detach()

        raw = gmc_rew.cpu().numpy()
        result["intrinsic_rewards"] = (GMC_CONFIG["intrinsic_coef"] * raw).tolist()
        result["gmc_mean"] = gmc_rew.mean().item()
        result["gmc_std"]  = gmc_rew.std().item()

        # --- Step 2: Normal ICM training (no backpack) ---
        self.icm_optimizer.zero_grad()

        feats      = self.icm_encoder(obs_b)
        next_feats = self.icm_encoder(nobs_b)
        nf_target  = next_feats.detach()

        pred_act     = self.icm_inverse(feats, next_feats)
        inverse_loss = nn.CrossEntropyLoss()(pred_act, acts)

        pred_nf2  = self.icm_forward(feats.detach(), act_oh)
        fwd_loss  = 0.5 * nn.MSELoss()(pred_nf2, nf_target)

        total_loss = (ICM_CONFIG["inverse_loss_coef"] * inverse_loss +
                      ICM_CONFIG["forward_loss_coef"] * fwd_loss)
        total_loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.icm_encoder.parameters()) +
            list(self.icm_inverse.parameters()) +
            list(self.icm_forward.parameters()),
            PPO_CONFIG["max_grad_norm"])
        self.icm_optimizer.step()

        result["forward_loss"] = fwd_loss.item()
        result["inverse_loss"] = inverse_loss.item()
        return result

    # ================================================================ GAE
    def compute_gae(self, rewards, values, dones, next_value):
        advantages = []
        gae = 0.0
        n = len(rewards)
        for t in reversed(range(n)):
            nxt_v = next_value if t == n - 1 else values[t + 1]
            mask  = 1.0 - dones[t]
            delta = rewards[t] + PPO_CONFIG["gamma"] * nxt_v * mask - values[t]
            gae   = delta + PPO_CONFIG["gamma"] * PPO_CONFIG["gae_lambda"] * mask * gae
            advantages.insert(0, gae)
        advantages = torch.tensor(advantages, dtype=torch.float32,
                                  device=self.device)
        returns = advantages + torch.tensor(
            values, dtype=torch.float32, device=self.device)
        return advantages, returns

    # ============================================================ PPO UPDATE
    def ppo_update(self, rollout, intrinsic_result):
        """Standard PPO policy + value update.  Returns metrics dict."""

        total_rewards = [
            r + ir for r, ir in zip(
                rollout["extrinsic_rewards"],
                intrinsic_result["intrinsic_rewards"])
        ]

        advantages, returns = self.compute_gae(
            total_rewards, rollout["values"],
            rollout["dones"], rollout["next_value"])

        adv_mean = advantages.mean()
        adv_std  = advantages.std()
        adv_normed = (advantages - adv_mean) / (adv_std + 1e-8)

        obs_batch  = torch.stack(
            [self._obs_to_tensor(o).squeeze(0) for o in rollout["observations"]])
        act_batch  = torch.tensor(
            rollout["actions"], dtype=torch.long, device=self.device)
        old_lp     = torch.tensor(
            rollout["log_probs"], dtype=torch.float32, device=self.device)

        p_losses, v_losses, entropies, clip_fracs = [], [], [], []

        for _ in range(PPO_CONFIG["ppo_epochs"]):
            idx = np.random.permutation(len(rollout["actions"]))
            for start in range(0, len(idx), PPO_CONFIG["mini_batch_size"]):
                mb = idx[start:start + PPO_CONFIG["mini_batch_size"]]

                mb_obs  = obs_batch[mb]
                mb_act  = act_batch[mb]
                mb_olp  = old_lp[mb]
                mb_adv  = adv_normed[mb]
                mb_ret  = returns[mb]

                # ---- policy ----
                self.policy_optimizer.zero_grad()
                probs   = self.policy(mb_obs)
                dist    = Categorical(probs)
                lp      = dist.log_prob(mb_act)
                entropy = dist.entropy().mean()
                ratio   = torch.exp(lp - mb_olp)
                s1      = ratio * mb_adv
                s2      = torch.clamp(
                    ratio,
                    1.0 - PPO_CONFIG["clip_epsilon"],
                    1.0 + PPO_CONFIG["clip_epsilon"]) * mb_adv
                p_loss  = -torch.min(s1, s2).mean() \
                          - PPO_CONFIG["entropy_coef"] * entropy
                p_loss.backward()
                nn.utils.clip_grad_norm_(
                    self.policy.parameters(), PPO_CONFIG["max_grad_norm"])
                self.policy_optimizer.step()

                # ---- value ----
                self.value_optimizer.zero_grad()
                v_pred = self.value_net(mb_obs).squeeze(-1)
                v_loss = nn.MSELoss()(v_pred, mb_ret)
                v_loss.backward()
                nn.utils.clip_grad_norm_(
                    self.value_net.parameters(), PPO_CONFIG["max_grad_norm"])
                self.value_optimizer.step()

                # ---- track ----
                p_losses.append(p_loss.item())
                v_losses.append(v_loss.item())
                entropies.append(entropy.item())
                with torch.no_grad():
                    cf = ((ratio - 1.0).abs() > PPO_CONFIG["clip_epsilon"]
                          ).float().mean().item()
                clip_fracs.append(cf)

        # explained variance
        with torch.no_grad():
            all_v  = self.value_net(obs_batch).squeeze(-1).cpu().numpy()
            ret_np = returns.cpu().numpy()
            var_y  = np.var(ret_np)
            ev = 1.0 - np.var(ret_np - all_v) / (var_y + 1e-8) \
                 if var_y > 1e-8 else 0.0

        return dict(
            policy_loss=np.mean(p_losses),
            value_loss=np.mean(v_losses),
            entropy=np.mean(entropies),
            clip_fraction=np.mean(clip_fracs),
            explained_variance=ev,
            advantage_mean=adv_mean.item(),
            advantage_std=adv_std.item(),
        )

    # ====================================================== GMC CORE
    def _extract_gmc(self, params, label, inner_abs=True, cosine=False):
        """Compute per-sample GMC intrinsic reward from backpack grad_batch.

        Args
        ----
        params : iterable of nn.Parameter  (must have .grad_batch)
        label  : str key for tracking momentum state
        inner_abs : use |g| @ |m| (preferred voting mechanism)
        cosine    : use cosine similarity instead

        Returns
        -------
        Tensor of shape (batch_size, 1).
        """
        if label not in self.gmc_state:
            self.gmc_state[label] = {}
        state = self.gmc_state[label]
        if "step" not in state:
            state["step"] = 0
        step = state["step"]
        b0, b1 = GMC_CONFIG["beta_dot"]

        gmc_corr     = None
        total_params = 0
        pidx         = 0

        for param in params:
            if not hasattr(param, "grad_batch"):
                continue
            g = param.grad_batch.detach()
            bsz = g.shape[0]
            g = g.reshape(bsz, -1)
            n_p = g.shape[1]

            if pidx not in state:
                state[pidx] = dict(
                    m=torch.zeros(n_p, device=self.device),
                    v=torch.zeros(n_p, device=self.device),
                )
            gs = state[pidx]
            pidx += 1

            m, v = gs["m"], gs["v"]
            if step > 0:
                m_hat = m / (1 - b0 ** step)
                v_hat = v / (1 - b1 ** step)
            else:
                m_hat, v_hat = m, v
            norm_mom = m_hat / torch.clamp(v_hat, min=1e-8)

            if cosine:
                dp = (torch.sum(g * norm_mom, dim=1)
                      / (torch.norm(g, dim=1) * torch.norm(norm_mom) + 1e-8))
                dp = dp.abs().unsqueeze(-1)
            elif inner_abs:
                dp = (g.abs() @ norm_mom.abs()).unsqueeze(-1)
            else:
                dp = (g @ norm_mom).abs().unsqueeze(-1)

            gmc_corr = dp if gmc_corr is None else gmc_corr + dp
            total_params += n_p

            g_mean = g.mean(dim=0)
            gs["m"].mul_(b0).add_(g_mean * (1 - b0))
            gs["v"].mul_(b1).add_((g_mean ** 2) * (1 - b1))

        if gmc_corr is None:
            return torch.zeros(1, device=self.device)
        gmc_corr = gmc_corr / np.sqrt(total_params)
        state["step"] += 1
        return gmc_corr

    # ============================================================ LOGGING
    def log_metrics(self, timestep, rollout_count, ppo_m, intr_r, rollout):
        """Write one row to CSV + print summary to console."""
        wt = time.time() - self.start_time
        ep_r = self.episode_rewards[-10:] if self.episode_rewards else [0.0]
        ep_l = self.episode_lengths[-10:] if self.episode_lengths else [0]
        ir = intr_r["intrinsic_rewards"]
        ext_r = rollout["extrinsic_rewards"]
        entropy_bonus = PPO_CONFIG["entropy_coef"] * ppo_m["entropy"]

        row = dict(
            timestep=timestep,
            rollout=rollout_count,
            wall_time=f"{wt:.1f}",
            ep_reward_mean=f"{np.mean(ep_r):.4f}",
            ep_reward_std=f"{np.std(ep_r):.4f}",
            ep_length_mean=f"{np.mean(ep_l):.1f}",
            ep_length_std=f"{np.std(ep_l):.1f}",
            policy_loss=f"{ppo_m['policy_loss']:.6f}",
            value_loss=f"{ppo_m['value_loss']:.6f}",
            entropy=f"{ppo_m['entropy']:.6f}",
            clip_fraction=f"{ppo_m['clip_fraction']:.4f}",
            explained_variance=f"{ppo_m['explained_variance']:.4f}",
            intrinsic_reward_mean=f"{np.mean(ir):.6f}",
            intrinsic_reward_std=f"{np.std(ir):.6f}",
            forward_dynamics_loss=f"{intr_r['forward_loss']:.6f}",
            inverse_dynamics_loss=f"{intr_r['inverse_loss']:.6f}",
            gmc_mean=f"{intr_r['gmc_mean']:.6f}",
            gmc_std=f"{intr_r['gmc_std']:.6f}",
            advantage_mean=f"{ppo_m['advantage_mean']:.6f}",
            advantage_std=f"{ppo_m['advantage_std']:.6f}",
            extrinsic_reward_mean=f"{np.mean(ext_r):.6f}",
            extrinsic_reward_std=f"{np.std(ext_r):.6f}",
            entropy_bonus=f"{entropy_bonus:.6f}",
        )
        self.log_file.write(
            ",".join(str(row[c]) for c in self.LOG_COLUMNS) + "\n")
        self.log_file.flush()

        # ---- console ----
        parts = [
            f"[{self.algorithm}|s{self.seed}]",
            f"t={timestep:>7d}",
            f"R={float(row['ep_reward_mean']):>7.3f}",
            f"L={float(row['ep_length_mean']):>6.1f}",
            f"pi_L={ppo_m['policy_loss']:>8.5f}",
            f"v_L={ppo_m['value_loss']:>8.5f}",
            f"H={ppo_m['entropy']:.4f}",
            f"clip={ppo_m['clip_fraction']:.3f}",
            f"EV={ppo_m['explained_variance']:.3f}",
        ]
        if self.use_dynamics:
            parts.append(f"intr={np.mean(ir):.5f}")
            parts.append(f"fwd_L={intr_r['forward_loss']:.5f}")
        if self.use_icm:
            parts.append(f"inv_L={intr_r['inverse_loss']:.5f}")
        if self.use_gmc:
            parts.append(f"gmc={intr_r['gmc_mean']:.5f}")
        print(" | ".join(parts))

    # ======================================================== VISUALISATION
    def visualize_agent(self, num_steps=1000, infinite=False):
        try:
            viz_env = gym.make(ENV_NAME, render_mode="human")
            if FULLY_OBSERVABLE:
                viz_env = FullyObsWrapper(viz_env)
            viz_env.reset(seed=self.seed)
            obs, _ = viz_env.reset()
            obs = self.noisy_tv.augment(viz_env, obs["image"])

            total, ep_r, ep_s = 0, 0.0, 0
            while True:
                with torch.no_grad():
                    probs = self.policy(self._obs_to_tensor(obs))
                action = Categorical(probs).sample().item()

                nxt, reward, term, trunc, _ = viz_env.step(action)
                done = term or trunc
                ep_r += reward
                ep_s += 1
                total += 1

                if done:
                    print(f"  viz ep: R={ep_r:.2f}, L={ep_s}")
                    obs, _ = viz_env.reset()
                    obs = self.noisy_tv.augment(viz_env, obs["image"])
                    ep_r, ep_s = 0.0, 0
                else:
                    obs = self.noisy_tv.augment(viz_env, nxt["image"])
                if not infinite and total >= num_steps:
                    break
            viz_env.close()
        except Exception as e:
            print(f"Visualization error: {e}")
            import traceback; traceback.print_exc()

    # ========================================================= MAIN LOOP
    def run(self):
        total_ts = 0
        rollout_n = 0

        while total_ts < TRAINING_CONFIG["total_timesteps"]:
            rollout       = self.collect_rollout()
            intrinsic_res = self.compute_intrinsic_and_train_dynamics(rollout)
            ppo_metrics   = self.ppo_update(rollout, intrinsic_res)

            rollout_n += 1
            total_ts  += PPO_CONFIG["rollout_length"]

            if rollout_n % TRAINING_CONFIG["log_interval"] == 0:
                self.log_metrics(total_ts, rollout_n,
                                 ppo_metrics, intrinsic_res, rollout)

            if self.simulate \
               and rollout_n % TRAINING_CONFIG["sim_interval"] == 0:
                print(f"  Running visualization (100 steps)...")
                self.visualize_agent(num_steps=100, infinite=False)

        if self.simulate:
            print("\n=== Training Complete ===")
            self.visualize_agent(num_steps=0, infinite=True)

    def close(self):
        self.log_file.close()


# ============================================================================
# GRAPHING / PLOTTING
# ============================================================================

def plot_experiment_results(log_dir=LOG_DIR, graph_dir=GRAPH_DIR,
                            smoothing=0.9):
    """Read all CSV log files and produce one comparison figure per seed.

    Each figure is a grid of subplots (one per metric); every algorithm for
    that seed is overlaid on the same subplot.
    """
    # Formatter to display timesteps as M (millions)
    def millions_formatter(x, pos):
        return f'{x/1e6:.2g}M' # Rounded to 2 significant digits
    log_files = sorted(glob.glob(os.path.join(log_dir, "minigrid_*.csv")))
    if not log_files:
        print(f"No log files found in {log_dir}")
        return

    # ---- parse ----
    experiments = {}   # (seed, noisy_tag, full_tag) -> {algo: data_dict}
    for path in log_files:
        fname = os.path.basename(path).replace(".csv", "")
        parsed = _parse_log_filename(fname)
        if parsed is None:
            continue
        algo, seed, noisy_tag, full_tag = parsed
        data = _read_csv(path)
        if data is None:
            continue
        exp_key = (seed, noisy_tag, full_tag)
        experiments.setdefault(exp_key, {})[algo] = data

    # ---- metrics to plot  (title, csv column) ----
    metrics = [
        ("Episode Reward (mean)",    "ep_reward_mean"),
        ("Episode Length (mean)",    "ep_length_mean"),
        ("Policy Loss",             "policy_loss"),
        ("Value Loss",              "value_loss"),
        ("Entropy",                 "entropy"),
        ("Clip Fraction",           "clip_fraction"),
        ("Explained Variance",      "explained_variance"),
        ("Intrinsic Reward (mean)", "intrinsic_reward_mean"),
        ("Forward Dynamics Loss",   "forward_dynamics_loss"),
        ("Inverse Dynamics Loss",   "inverse_dynamics_loss"),
        ("GMC (mean)",  "gmc_mean"),
        ("Advantage (mean)",        "advantage_mean"),
    ]

    for (seed, noisy_tag, full_tag), algo_data in experiments.items():
        total_subplots = len(metrics) + 1  # +1 for reward components comparison
        ncols = 4
        nrows = (total_subplots + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(5 * ncols, 4 * nrows),
                                 squeeze=False)
        cfg_label = f"noisy_tv={noisy_tag} | obs={full_tag}"
        fig.suptitle(
            f"MiniGrid {ENV_NAME}  --  seed {seed} | {cfg_label}",
            fontsize=14, fontweight="bold")

        for idx, (title, col) in enumerate(metrics):
            ax = axes[idx // ncols][idx % ncols]
            has_data = False
            for algo in sorted(algo_data):
                d = algo_data[algo]
                if col not in d or not d[col]:
                    continue
                y = np.array(d[col], dtype=float)
                if np.all(y == 0):
                    continue
                x = np.array(d["timestep"], dtype=float)
                color = _ALGO_COLORS.get(algo)
                ax.plot(x, y, alpha=0.15, color=color)
                ax.plot(x, _ema(y, smoothing),
                        label=algo, color=color, alpha=0.9)
                has_data = True
            ax.set_title(title, fontsize=10)
            ax.set_xlabel("Timesteps")
            ax.xaxis.set_major_formatter(FuncFormatter(millions_formatter))
            ax.grid(True, alpha=0.3)
            if has_data:
                ax.legend(fontsize=8)

        # ---- Reward Components Comparison subplot ----
        comp_idx = len(metrics)
        ax_comp = axes[comp_idx // ncols][comp_idx % ncols]
        comp_cols   = ["extrinsic_reward_mean", "intrinsic_reward_mean",
                       "entropy_bonus"]
        comp_styles = ["-", "--", ":"]
        comp_names  = ["Extrinsic", "Intrinsic", "Entropy Bonus"]
        has_comp = False
        for algo in sorted(algo_data):
            d = algo_data[algo]
            color = _ALGO_COLORS.get(algo)
            for col, ls, cname in zip(comp_cols, comp_styles, comp_names):
                if col not in d or not d[col]:
                    continue
                y = np.array(d[col], dtype=float)
                if np.all(y == 0):
                    continue
                x = np.array(d["timestep"], dtype=float)
                ax_comp.plot(x, _ema(y, smoothing),
                             label=f"{algo} {cname}",
                             color=color, linestyle=ls, alpha=0.9)
                has_comp = True
        ax_comp.set_title("Reward Components (per step)", fontsize=10)
        ax_comp.set_xlabel("Timesteps")
        ax_comp.xaxis.set_major_formatter(FuncFormatter(millions_formatter))
        ax_comp.grid(True, alpha=0.3)
        if has_comp:
            ax_comp.legend(fontsize=7)

        for idx in range(total_subplots, nrows * ncols):
            axes[idx // ncols][idx % ncols].set_visible(False)

        plt.tight_layout()
        out = os.path.join(
            graph_dir,
            f"minigrid_{noisy_tag}_{full_tag}_seed{seed}.png")
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved graph -> {out}")
        plt.close(fig)


def plot_seed_averaged_reward(log_dir=LOG_DIR, graph_dir=GRAPH_DIR,
                              smoothing=0.9, agg_n=10):
    """Episodic reward averaged over all seeds per algo, with ±1-std shading.

    One figure per (noisy_tv, fully_obs) config combination found in log_dir.
    Each figure plots mean ± std across seeds after block-averaging every
    ``agg_n`` rollouts for smoothness.
    """
    def millions_formatter(x, pos):
        return f'{x/1e6:.2g}M'

    log_files = sorted(glob.glob(os.path.join(log_dir, "minigrid_*.csv")))
    if not log_files:
        print(f"No log files found in {log_dir}")
        return

    # Collect: {(noisy_tag, full_tag, algo): {seed: data}}
    grouped = {}
    for path in log_files:
        fname = os.path.basename(path).replace(".csv", "")
        parsed = _parse_log_filename(fname)
        if parsed is None:
            continue
        algo, seed, noisy_tag, full_tag = parsed
        data = _read_csv(path)
        if data is None or "ep_reward_mean" not in data:
            continue
        key = (noisy_tag, full_tag, algo)
        grouped.setdefault(key, {})[seed] = data

    # One figure per (noisy_tag, full_tag) combination
    config_keys = sorted({(nt, ft) for nt, ft, _ in grouped})
    for (noisy_tag, full_tag) in config_keys:
        fig, ax = plt.subplots(figsize=(10, 6))
        cfg_label = f"noisy_tv={noisy_tag} | obs={full_tag}"
        # fig.suptitle(f"Episodic Reward – seed-averaged\n{ENV_NAME} | {cfg_label}", fontsize=13, fontweight="bold")

        algos = sorted(k[2] for k in grouped
                       if k[0] == noisy_tag and k[1] == full_tag)
        for algo in algos:
            seed_dict = grouped[(noisy_tag, full_tag, algo)]
            color = _ALGO_COLORS.get(algo)

            arrays, ts_ref = [], None
            for seed, data in seed_dict.items():
                y = np.array(data["ep_reward_mean"], dtype=float)
                arrays.append(y)
                if ts_ref is None:
                    ts_ref = np.array(data["timestep"], dtype=float)

            if not arrays or ts_ref is None:
                continue

            min_len = min(len(a) for a in arrays)
            arrays  = [a[:min_len] for a in arrays]
            ts_ref  = ts_ref[:min_len]

            # Block-aggregate every agg_n rollouts
            T_agg = min_len // agg_n
            if T_agg == 0:
                continue
            stacked = np.stack(arrays)[:, :T_agg * agg_n]           # (S, T')
            stacked = stacked.reshape(stacked.shape[0], T_agg, agg_n).mean(axis=2)  # (S, T_agg)
            ts_agg  = ts_ref[:T_agg * agg_n].reshape(T_agg, agg_n).mean(axis=1)

            mean_c = stacked.mean(axis=0)
            std_c  = stacked.std(axis=0)
            n_seeds = stacked.shape[0]

            t_crit = scipy_stats.t.ppf(0.975, df=n_seeds - 1)  # 95% CI, t-distribution
            ci95 = t_crit * std_c / np.sqrt(n_seeds)
            ax.plot(ts_agg, mean_c, label=f"{algo}", color=color, linewidth=2)
            ax.fill_between(ts_agg,
                            mean_c - ci95, mean_c + ci95,
                            alpha=0.2, color=color)

        ax.set_xlabel("Timesteps")
        ax.set_ylabel("Episode Reward (mean)")
        ax.xaxis.set_major_formatter(FuncFormatter(millions_formatter))
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        plt.tight_layout()

        out = os.path.join(
            graph_dir,
            f"minigrid_reward_avg_{noisy_tag}_{full_tag}.png")
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved seed-averaged reward graph -> {out}")
        plt.close(fig)


def print_pairwise_ttest_reward(log_dir=LOG_DIR, agg_n=10):
    """Pairwise Welch's t-tests on mean episodic reward (AUC) across seeds.

    For each (noisy_tag, full_tag) condition, computes per-seed AUC of
    episodic reward and tests each algorithm pair.
    """
    log_files = sorted(glob.glob(os.path.join(log_dir, "minigrid_*.csv")))
    if not log_files:
        print(f"No log files found in {log_dir}")
        return

    grouped = {}
    for path in log_files:
        fname = os.path.basename(path).replace(".csv", "")
        parsed = _parse_log_filename(fname)
        if parsed is None:
            continue
        algo, seed, noisy_tag, full_tag = parsed
        data = _read_csv(path)
        if data is None or "ep_reward_mean" not in data:
            continue
        key = (noisy_tag, full_tag, algo)
        grouped.setdefault(key, {})[seed] = data

    config_keys = sorted({(nt, ft) for nt, ft, _ in grouped})

    print(f"\n{'='*80}")
    print("Pairwise Welch's t-tests: mean episodic reward AUC across seeds")
    print(f"{'='*80}")

    for (noisy_tag, full_tag) in config_keys:
        cfg_label = f"noise={noisy_tag} | obs={full_tag}"
        print(f"\n--- {cfg_label} ---")

        algos = sorted(k[2] for k in grouped
                       if k[0] == noisy_tag and k[1] == full_tag)

        # Compute per-seed AUC (sum of reward over time)
        algo_seed_aucs = {}
        for algo in algos:
            seed_dict = grouped[(noisy_tag, full_tag, algo)]
            arrays = []
            for seed, data in seed_dict.items():
                y = np.array(data["ep_reward_mean"], dtype=float)
                arrays.append(y)

            if not arrays:
                continue

            min_len = min(len(a) for a in arrays)
            arrays = [a[:min_len] for a in arrays]

            T_agg = min_len // agg_n
            if T_agg == 0:
                continue
            stacked = np.stack(arrays)[:, :T_agg * agg_n]
            stacked = stacked.reshape(stacked.shape[0], T_agg, agg_n).mean(axis=2)
            # Per-seed AUC = sum of block-averaged rewards
            algo_seed_aucs[algo] = stacked.sum(axis=1)  # shape (n_seeds,)

        header = f"  {'Algo A':<12} {'Algo B':<12} {'mean_A':>8} {'mean_B':>8} {'t-stat':>8} {'p-value':>10} {'sig':>5}"
        print(header)
        print("  " + "-" * 70)

        for i, a1 in enumerate(algos):
            for a2 in algos[i+1:]:
                if a1 not in algo_seed_aucs or a2 not in algo_seed_aucs:
                    continue
                aucs1 = algo_seed_aucs[a1]
                aucs2 = algo_seed_aucs[a2]
                t_stat, p_val = scipy_stats.ttest_ind(aucs1, aucs2, equal_var=False)
                if p_val < 0.001:
                    stars = "***"
                elif p_val < 0.01:
                    stars = "**"
                elif p_val < 0.05:
                    stars = "*"
                else:
                    stars = "ns"
                print(f"  {a1:<12} {a2:<12} {aucs1.mean():>8.1f} {aucs2.mean():>8.1f} {t_stat:>8.3f} {p_val:>10.4f} {stars:>5}")

    print(f"\n{'='*80}")
    print("Significance: *** p<0.001, ** p<0.01, * p<0.05, ns not significant\n")


def _read_csv(path):
    """Read a CSV log file into {column: [values]}."""
    try:
        with open(path) as f:
            header = f.readline().strip().split(",")
            data = {c: [] for c in header}
            for line in f:
                vals = line.strip().split(",")
                if len(vals) != len(header):
                    continue
                for c, v in zip(header, vals):
                    try:
                        data[c].append(float(v))
                    except ValueError:
                        data[c].append(v)
        return data
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return None


def _ema(values, alpha=0.9):
    """Exponential moving average."""
    s = np.empty_like(values, dtype=float)
    s[0] = values[0]
    for i in range(1, len(values)):
        s[i] = alpha * s[i - 1] + (1 - alpha) * values[i]
    return s


def _parse_log_filename(fname):
    """Parse a minigrid log filename (without .csv) into (algo, seed, noisy_tag, full_tag).

    Expected format: ``minigrid_{algo}_{noisy_tag}_{full_tag}_{seed}``
    Returns None if the filename cannot be parsed.
    """
    parts = fname.split("_")
    # Need at least: minigrid + algo + noisy_tag + full_tag + seed = 5 parts
    if len(parts) < 5:
        return None
    seed      = parts[-1]
    full_tag  = parts[-2]
    noisy_tag = parts[-3]
    algo      = "_".join(parts[1:-3])
    return algo, seed, noisy_tag, full_tag


# ============================================================================
# HELPERS
# ============================================================================

def _create_noisy_tv_checker(mode="door"):
    """Return a function that determines when to add noise to observations.
    
    Args:
        mode: One of "none", or "door"
            - "none":       Never add noise
            - "door":       Add noise when agent is next to the door
    """
    if mode == "none":
        return lambda env: False
    
    elif mode == "door":
        def checker(env):
            agent_pos = env.unwrapped.agent_pos
            for obj in env.unwrapped.grid.grid:
                if obj is not None and obj.type == "door":
                    dp = obj.cur_pos
                    if abs(agent_pos[0] - dp[0]) + abs(agent_pos[1] - dp[1]) <= 1:
                        return True
            return False
        return checker
    
    else:
        raise ValueError(f"Unknown noisy TV mode: {mode}. Must be 'none' or 'door'.")


# ============================================================================
# EXPERIMENT RUNNER
# ============================================================================

def _run_experiment(algorithm, seed, simulate=False):
    """Entry-point for a spawned process."""
    trainer = None
    try:
        trainer = PPOTrainer(algorithm, seed, simulate=simulate)
        trainer.run()
        print(f"\nCompleted: {algorithm} seed={seed} -> {trainer.log_filename}")
    except Exception as e:
        print(f"ERROR ({algorithm} seed={seed}): {e}")
        import traceback; traceback.print_exc()
    finally:
        if trainer:
            trainer.close()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def run_all_experiments(simulate=False):
    """Run every (algorithm, seed) pair in a fresh process.

    Uses ``fork`` (not ``spawn``) so that child processes inherit the
    parent's already-imported module state rather than re-importing the
    .py file from disk.  This means edits made to the file in a *second*
    terminal session cannot bleed into the child processes of a
    *first* session that is still running.

    ``fork`` is safe here because the parent process never initialises a
    CUDA context (that only happens inside ``_run_experiment`` after the
    fork), so each child still gets its own independent CUDA context and
    GPU memory is fully isolated between experiments.
    """
    ctx = multiprocessing.get_context("fork")
    for seed in EXPERIMENT_CONFIG["seeds"]:
        for algo in EXPERIMENT_CONFIG["algorithms"]:
            print(f"\n{'=' * 60}")
            print(f"  Starting: {algo}  seed={seed}")
            print(f"{'=' * 60}")
            proc = ctx.Process(target=_run_experiment,
                               args=(algo, seed, simulate))
            proc.start()
            proc.join()
            if proc.exitcode != 0:
                print(f"WARNING: {algo} seed={seed} exited with "
                      f"code {proc.exitcode}")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    simulate   = False
    graph = False

    for arg in sys.argv[1:]:
        if arg in ("--simulate", "-s"):
            simulate = True
        elif arg in ("--graph", "-g"):
            graph = True
        else:
            print(f"Unknown argument: {arg}")
            print("Usage: python minigrid_ppo_gmc.py [--simulate|-s] [--graph|-g]")
            sys.exit(1)

    if graph:
        plot_experiment_results()
        plot_seed_averaged_reward()
        print_pairwise_ttest_reward()
    else:
        run_all_experiments(simulate=simulate)

# Duration averages:

# Experiment: minigrid_icm_door_full
# Average duration: 2:33:32

# Experiment: minigrid_icm_gmc_door_full
# Average duration: 2:22:56

# Experiment: minigrid_icm_gmc_none_full
# Average duration: 2:20:26

# Experiment: minigrid_icm_none_full
# Average duration: 2:20:45

# Experiment: minigrid_ppo_door_full
# Average duration: 2:07:45

# Experiment: minigrid_ppo_gmc_door_full
# Average duration: 2:30:40

# Experiment: minigrid_ppo_gmc_none_full
# Average duration: 2:21:26

# Experiment: minigrid_ppo_none_full
# Average duration: 2:02:04
