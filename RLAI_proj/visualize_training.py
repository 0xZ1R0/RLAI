import os
import torch
from stable_baselines3 import PPO
import gymnasium as gym

def visualize_training():
    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Define the model path with the folder
    model_folder = "./models"
    model_path = os.path.join(model_folder, "filhinhu.zip")

    # Check if the model exists
    if not os.path.exists(model_path):
        print("Model file does not exist. Please train the model first.")
        return

    # Create the environment with render mode set to 'human' for visualization
    env = gym.make('Humanoid-v4', render_mode='human')

    # Load the trained model
    model = PPO.load(model_path, device=device)

    # Run the model in the environment to visualize
    obs, _ = env.reset()  # Extract the observation from the tuple
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, _, _ = env.step(action)  # Extract the observation from the tuple
        env.render()
        if dones:
            obs, _ = env.reset()  # Extract the observation from the tuple

if __name__ == "__main__":
    visualize_training()