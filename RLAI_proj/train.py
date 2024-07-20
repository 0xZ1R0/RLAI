import torch
import torch.nn as nn
import torch.optim as optim
from stable_baselines3 import PPO
import gymnasium as gym
import os
import json

def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    env = gym.make('Humanoid-v4', render_mode=None)
    env = gym.wrappers.RecordEpisodeStatistics(env)  # To record episode statistics

    model_folder = "./models"
    model_path = os.path.join(model_folder, "filhinhu.zip")
    generation_path = os.path.join(model_folder, "generation.json")

    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    if os.path.exists(generation_path):
        with open(generation_path, 'r') as f:
            generation = json.load(f)
    else:
        generation = 0

    if os.path.exists(model_path):
        print(f"Loading existing model from generation {generation}...")
        model = PPO.load(model_path, env=env, device=device)
    else:
        print("Training new model...")
        model = PPO("MlpPolicy", env, verbose=1, device=device)

    model.learn(total_timesteps=10000)

    model.save(model_path)
    generation += 1
    with open(generation_path, 'w') as f:
        json.dump(generation, f)
    print(f"Model trained and saved for generation {generation}.")

    return model