# I'm lazy and don't want to do proper command line processing

# loads a model spec. by MODEL_TO_LOAD string and plays it live for a few thousand iterations
# just so you can watch it play for a bit for fun

import time
from stable_baselines3 import DQN, A2C, PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

BUFFER_SIZE = 100000
VEC_STACK = 4

MODEL_TO_LOAD = "PPO_4M_steps"

if __name__ == "__main__":

    env = make_atari_env('SpaceInvadersNoFrameskip-v4')
    env = VecFrameStack(env, n_stack=VEC_STACK)

    # TODO: change me based on the model being loaded
    model = PPO.load(MODEL_TO_LOAD, env=env)

    vec_env = model.get_env()
    obs = vec_env.reset()
    for _ in range(5000):
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)
        vec_env.render("human")
        time.sleep(0.0416)
