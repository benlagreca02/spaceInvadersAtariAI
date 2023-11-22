import gymnasium as gym
import numpy as np
from gymnasium import spaces

# Using custom environments official docs
# https://stable-baselines3.readthedocs.io/en/master/guide/custom_env.html


# OH MY LORD CROP IT LATER  I THINK
# this implies it will take in the *entire* screen
# really the right and left edges are useless, because it's a 4:3 game on 16:9(?) monitor
# will probably switch to grayscale which means N_CHANNELS := 1
N_CHANNELS = 3
WIDTH = 1920
HEIGHT = 1080

# one action, placeholder to get it to verify the legitness of our env
N_DISCRETE_ACTIONS = 1


# in theory... could change this to load *any* n64 rom...
# but then the action space is messed up
class SpaceInvadersN64Env(gym.Env):
    """Custom Environment that follows gym interface."""

    # metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self):
        super().__init__()
        self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)
        # Example for using image as input (channel-first; channel-last also works):
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(N_CHANNELS, HEIGHT, WIDTH), dtype=np.uint8)

    def step(self, action):
        # for now, just make a random image
        observation = self.observation_space.sample()
        reward = 0
        terminated = False
        truncated = False
        info = dict()
        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        # just take a random sample of observation space possibility (expecting noisy 1920x1080 pic)
        observation = self.observation_space.sample()
        info = dict()
        return observation, info

    def render(self):
        pass

    def close(self):
        pass
