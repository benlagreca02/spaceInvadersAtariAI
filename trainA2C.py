import stable_baselines3
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_atari_env
import numpy as np
import os


# could put this in its own class, but copy paste is easier lol
class SaveOnBestTrainingRewardCallback(BaseCallback):
    def __init__(self, check_freq: int, save_path: str, verbose):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path

    def _init_callback(self) -> None:
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(self.save_path, 'best_model_'.format(self.n_calls))
            self.model.save(model_path)
        return True


if __name__ == "__main__":
    CHECKPOINT_DIR = './trainA2C/'
    LOG_DIR = './logsA2C/'
    callback = SaveOnBestTrainingRewardCallback(check_freq=10000, save_path=(CHECKPOINT_DIR), verbose=0)

    env = make_atari_env('SpaceInvadersNoFrameskip-v4', n_envs=1, seed=0)
    env = VecFrameStack(env, n_stack=4)

    model = A2C("CnnPolicy", env, verbose=1, tensorboard_log=LOG_DIR)

    model.learn(total_timesteps=1000000, callback=callback, progress_bar=True)
