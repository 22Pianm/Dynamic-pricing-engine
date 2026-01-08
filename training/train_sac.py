# training/train_sac.py

import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces


from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

from simulation.pricing_env import PricingEnv


# =====================================================
# Gym Wrapper
# =====================================================
class PricingGymEnv(gym.Env):
    """
    Gymnasium wrapper around PricingEnv
    """

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

        self.state = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = self.env.reset()
        return self.state, {}

    def step(self, action):
        action = float(action[0])

        next_state, reward, done, info = self.env.step(action)

        if done:
            next_state = np.zeros(self.observation_space.shape, dtype=np.float32)

        self.state = next_state

        # Gymnasium API: obs, reward, terminated, truncated, info
        return next_state, float(reward), done, False, info


# =====================================================
# Train SAC
# =====================================================
def train():
    os.makedirs("models", exist_ok=True)

    env = DummyVecEnv([lambda: PricingGymEnv()])

    checkpoint_cb = CheckpointCallback(
        save_freq=50_000,
        save_path="models/checkpoints",
        name_prefix="sac_pricing"
    )

    model = SAC(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        buffer_size=200_000,
        learning_starts=5_000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        ent_coef="auto",
        target_update_interval=1,
        verbose=1,
        device="auto"
    )

    model.learn(
        total_timesteps=200_000,
        callback=checkpoint_cb,
        progress_bar=True
    )

    model.save("models/sac_pricing_policy")

    print("\n✅ SAC pricing policy saved → models/sac_pricing_policy.zip")


if __name__ == "__main__":
    train()
