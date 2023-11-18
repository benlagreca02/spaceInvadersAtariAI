import gymnasium as gym
from stable_baselines3 import DQN, A2C
from stable_baselines3.common.evaluation import evaluate_policy

if __name__ == '__main__':

    TOTAL_TIMESTEPS = 500000

    # make space invaders environment
    print("Making environment")
    env = gym.make('LunarLander-v2', render_mode='rgb_array')

    # instantiate agent
    print("Instantiating agent")
    model = DQN("MlpPolicy", env, verbose=0)

    # train the agent w/ prog. bar
    print("Training...")
    model.learn(total_timesteps=TOTAL_TIMESTEPS, progress_bar=True)

    model.save(f'dqn_spaceInvaders_{TOTAL_TIMESTEPS}_steps')

    mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

    # "enjoy trained agent"
    vec_env = model.get_env()
    obs = vec_env.reset()
    for _ in range(5000):
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)
        vec_env.render("human")
