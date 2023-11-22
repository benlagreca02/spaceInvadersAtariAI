
import time
from stable_baselines3 import DQN, A2C
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

ENV_NAME = 'ALE/SpaceInvaders-v5'

TOTAL_TIMESTEPS = 1_000_000
BUFFER_SIZE = 100_000
VEC_STACK=4

# pick between MlpPolicy, CnnPolicy, and MultiInputPolicy
# docs say to use Cnn with image inputs
DQN_POLICY = 'CnnPolicy'


if __name__ == "__main__":

    env = make_atari_env('SpaceInvadersNoFrameskip-v4')
    env = VecFrameStack(env, n_stack=VEC_STACK)

    model = DQN.load(f'{DQN_POLICY}_ReplaySize{BUFFER_SIZE}_NumTimesteps{TOTAL_TIMESTEPS}_VecStack{VEC_STACK}', env=env)

    vec_env = model.get_env()
    obs = vec_env.reset()
    for _ in range(5000):
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)
        vec_env.render("human")
        time.sleep(0.1)
