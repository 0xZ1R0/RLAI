import torch
import torch.nn as nn
import torch.optim as optim
from stable_baselines3 import PPO
import gymnasium as gym
import os
import json

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create the environment
env = gym.make('Humanoid-v4')
env = gym.wrappers.RecordEpisodeStatistics(env)  # To record episode statistics

# Define the model path with the folder
model_folder = "./models"
model_path = os.path.join(model_folder, "filhinhu.zip")
generation_path = os.path.join(model_folder, "generation.json")

# Create the models folder if it doesn't exist
if not os.path.exists(model_folder):
    os.makedirs(model_folder)

# Load or initialize the generation count
if os.path.exists(generation_path):
    with open(generation_path, 'r') as f:
        generation = json.load(f)
else:
    generation = 0

# Check if the model already exists
if os.path.exists(model_path):
    print(f"Loading existing model from generation {generation}...")
    model = PPO.load(model_path, env=env, device=device)
else:
    print("Training new model...")
    # Initialize the PPO agent
    model = PPO("MlpPolicy", env, verbose=1, device=device)

# Train the agent
model.learn(total_timesteps=200000)

# Save the model
model.save(model_path)
generation += 1
with open(generation_path, 'w') as f:
    json.dump(generation, f)
print(f"Model trained and saved for generation {generation}.")