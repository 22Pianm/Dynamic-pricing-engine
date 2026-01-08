# training/train_ppo.py

import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

from simulation.pricing_env import PricingEnv


# =====================================================
# Gym Wrapper (same as SAC for fair comparison)
# =====================================================
class PricingGymEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()

        self.env = PricingEnv()

        # Action: price multiplier ∈ [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32
        )

        # Observation: 8D continuous state
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(8,),
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        obs = self.env.reset()
        return obs, {}

    def step(self, action):
        action = float(action[0])

        obs, reward, done, info = self.env.step(action)

        if done:
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)

        # Gymnasium API
        return obs, float(reward), done, False, info


# =====================================================
# Train PPO
# =====================================================
def train():
    os.makedirs("models", exist_ok=True)

    env = DummyVecEnv([lambda: PricingGymEnv()])

    checkpoint_cb = CheckpointCallback(
        save_freq=50_000,
        save_path="models/checkpoints",
        name_prefix="ppo_pricing"
    )

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=1,
        device="auto"
    )

    model.learn(
        total_timesteps=160_000,
        callback=checkpoint_cb,
        progress_bar=True
    )

    model.save("models/ppo_pricing_policy")

    print("\n✅ PPO pricing policy saved → models/ppo_pricing_policy.zip")


if __name__ == "__main__":
    train()
