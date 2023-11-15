import gym

if __name__ == '__main__':
    # render mode makes it so we see it
    # env = gym.make('ALE/SpaceInvaders-v5', render_mode='human')
    env = gym.make('LunarLander-v2', render_mode='human')

    # no idea
    env.action_space.seed(69)

    observation, info = env.reset(seed=69)

    for _ in range(1000):
        observation, reward, terminated, truncated, info = env.step(env.action_space.sample())
        if terminated or truncated:
            observation, info = env.reset()

    env.close()