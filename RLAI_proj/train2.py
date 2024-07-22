# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
import gymnasium as gym
import os
import json

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create the environment
env = gym.make('Humanoid-v4', render_mode=None)
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
    # Initialize the PPO agent with adjusted hyperparameters
    model = PPO("MlpPolicy", env, verbose=1, device=device, 
                learning_rate=0.0005028079767203933, 
                batch_size=16, 
                gamma=0.976633446886447, 
                ent_coef=1.1370504201084368e-08, 
                clip_range=0.3990038259701256)
                #TIRAL 29 Mean Reward: 539.9328608512878

# Define a function to adjust the learning rate
def adjust_learning_rate(model, initial_lr, step, gamma):
    new_lr = initial_lr * (gamma ** step)
    for param_group in model.policy.optimizer.param_groups:
        param_group['lr'] = new_lr
    return new_lr

# Train the agent
total_timesteps = 3000000
initial_lr = 0.0005028079767203933
gamma = 0.1
step_size = 30000  # Adjust this value based on your needs

for step in range(0, total_timesteps, step_size):
    model.learn(total_timesteps=step_size)
    new_lr = adjust_learning_rate(model, initial_lr, step // step_size, gamma)
    print(f"Learning rate adjusted to {new_lr}")

# Save the model
model.save(model_path)
generation += 1
with open(generation_path, 'w') as f:
    json.dump(generation, f)
print(f"Model trained and saved for generation {generation}.")

# Evaluate the agent
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"Mean reward: {mean_reward} +/- {std_reward}")