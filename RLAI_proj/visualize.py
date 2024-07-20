import gymnasium as gym

def visualize_model(model):
    env = gym.make('Humanoid-v4', render_mode="human")
    obs = env.reset()
    for _ in range(1000):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()
        if dones:
            obs = env.reset()
    env.close()