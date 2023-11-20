
import time
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3 import DQN, A2C


# just a copy paste from the other file...
# could be cleaned up a lot I know, but this is still just 
# the 'playing aorund with things' phase of stuff

TOTAL_TIMESTEPS = 500_000
ENV_NAME = 'ALE/SpaceInvaders-v5'

# pick between MlpPolicy, CnnPolicy, and MultiInputPolicy
# docs say to use Cnn with image inputs
DQN_POLICY = 'CnnPolicy'
NUM_ENVS = 1

# this is the 'replay buffer' size, too large and we won't be able to
# instantiate the agent
BUFFER_SIZE = 100_000

if __name__ == "__main__":

    env = make_atari_env('SpaceInvadersNoFrameskip-v4', n_envs=NUM_ENVS)

    model = DQN.load(f'{DQN_POLICY}_ReplaySize{BUFFER_SIZE}_NumTimesteps{TOTAL_TIMESTEPS}', env=env)

    vec_env = model.get_env()
    obs=vec_env.reset()
    for _ in range(5000):
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)
        vec_env.render("human")
        time.sleep(0.1)
