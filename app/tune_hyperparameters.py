import optuna
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
import gymnasium as gym
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def objective(trial):
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
    gamma = trial.suggest_float('gamma', 0.9, 0.9999)
    ent_coef = trial.suggest_float('ent_coef', 1e-8, 1e-1, log=True)
    clip_range = trial.suggest_float('clip_range', 0.1, 0.4)

    env = gym.make('Humanoid-v4', render_mode=None)
    env = gym.wrappers.RecordEpisodeStatistics(env)  # To record episode statistics

    model = PPO("MlpPolicy", env, verbose=0, learning_rate=learning_rate, batch_size=batch_size, gamma=gamma, ent_coef=ent_coef, clip_range=clip_range)
    model.learn(total_timesteps=100000)

    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=5)
    logger.info(f"Trial {trial.number} - Params: {trial.params} - Mean Reward: {mean_reward}")
    return mean_reward

def tune_hyperparameters():
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100, n_jobs=-1)

    logger.info("Best trial:")
    trial = study.best_trial
    logger.info(f"  Value: {trial.value}")
    logger.info("  Params: ")
    for key, value in trial.params.items():
        logger.info(f"    {key}: {value}")

    # Save the best hyperparameters to a file
    with open('best_hyperparameters.txt', 'w') as f:
        f.write(f"Best trial:\n")
        f.write(f"  Value: {trial.value}\n")
        f.write("  Params:\n")
        for key, value in trial.params.items():
            f.write(f"    {key}: {value}\n")

if __name__ == "__main__":
    tune_hyperparameters()