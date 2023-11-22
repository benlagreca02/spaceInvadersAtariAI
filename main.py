import gymnasium as gym

# for verifying we made the custom environment right
from stable_baselines3.common.env_checker import check_env

from SpaceInvadersN64Env import SpaceInvadersN64Env
from utils.ScreenRipper import ScreenRipper
from stable_baselines3 import DQN, A2C
# does all the nice wrapping and prettying up for us
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

if __name__ == "__main__":
    custom_env = SpaceInvadersN64Env()
    print("Checking ")
    check_env(custom_env)
    print("Done! Congrats gamer!")
