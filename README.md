# **RLAI_proj**

This repository serves as a cloud backup for a **reinforcement learning** project focused on training, tuning, and evaluating an AI model using the **PPO (Proximal Policy Optimization)** algorithm in a `Humanoid-v4` environment. The project includes scripts for managing training iterations, adjusting hyperparameters, and saving model generations.

## **Project Structure**
- **Training**:
  - `main.py`: Main entry point for model training, allowing users to set the number of training iterations.
  - `train2.py`: Contains the core training loop for the reinforcement learning agent, with hyperparameter adjustments and model checkpoints.
  
- **Hyperparameter Tuning**:
  - `tune_hyperparameters.py`: Uses **Optuna** to optimize key hyperparameters for improved model performance.

## **Dependencies**
- **Python 3.x**
- **Stable Baselines3** for reinforcement learning models
- **PyTorch** for neural network operations
- **Optuna** for hyperparameter optimization

## **Usage**
Clone this repository and use the provided scripts to train or tune the model. This setup is intended for **experimentation** and **improvement** of reinforcement learning model training.
