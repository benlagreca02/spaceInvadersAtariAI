import time
from stable_baselines3 import DQN, A2C, PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecFrameStack


DQN_MODEL = "DQN_1M_steps"
A2C_MODEL = "A2C_1M_steps"
PPO_MODEL = "PPO_4M_steps"

VEC_STACK = 4

# INCREASE THIS A LOT
EVAL_EPS = 10

if __name__ == '__main__':
    env = make_atari_env('SpaceInvadersNoFrameskip-v4')
    env = VecFrameStack(env, n_stack=VEC_STACK)
    # Evaluate the agent
    print("Loading environments...")
    dqn_model = DQN.load(DQN_MODEL, env=env)
    a2c_model = A2C.load(A2C_MODEL, env=env)

    print("Evaluating...")
    mean_reward, std_reward = evaluate_policy(dqn_model, dqn_model.get_env(), n_eval_episodes=EVAL_EPS)
    print(f"DQN avg reward: {mean_reward}")
    mean_reward, std_reward = evaluate_policy(a2c_model, a2c_model.get_env(), n_eval_episodes=EVAL_EPS)
    print(f"A2C avg reward: {mean_reward}")
