import time

import gymnasium as gym
from stable_baselines3 import DQN, A2C
# does all the nice wrapping and prettying up for us
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecFrameStack

if __name__ == '__main__':

    TOTAL_TIMESTEPS = 500_000
    ENV_NAME = 'ALE/SpaceInvaders-v5'

    # pick between MlpPolicy, CnnPolicy, and MultiInputPolicy
    # docs say to use Cnn with image inputs
    DQN_POLICY = 'CnnPolicy'
    NUM_ENVS = 1
    # this is the 'replay buffer' size
    BUFFER_SIZE = 100_000

    print(f"Making environment: {ENV_NAME}")

    # DON'T DO ATARI THIS WAY!!!
    # env = gym.make(ENV_NAME, render_mode='rgb_array')

    # we use 'make_atari_env' rather than 'gym.make()' because make_atari does some
    # preprocessing on the env. to make it less data intensive
    env = make_atari_env('SpaceInvadersNoFrameskip-v4', n_envs=NUM_ENVS)

    # instantiate agent
    print("Instantiating agent")

    # default buffer size is 1_000_000 and that tries to allocate 93 GB
    # This is where 'hyperparameter tuning' will come into play seriously
    model = DQN(DQN_POLICY, env, verbose=0, buffer_size=BUFFER_SIZE, )

    # train the agent w/ prog. bar
    print("Training...")
    model.learn(total_timesteps=TOTAL_TIMESTEPS, progress_bar=True)

    print("Saving model")
    model.save(f'{DQN_POLICY}_ReplaySize{BUFFER_SIZE}_NumTimesteps{TOTAL_TIMESTEPS}')

    # for evaluating
    # mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

    print("Enjoying the model...")
    # "enjoy trained agent"
    vec_env = model.get_env()
    obs = vec_env.reset()
    for _ in range(5000):
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)
        vec_env.render("human")
        time.sleep(0.1)
