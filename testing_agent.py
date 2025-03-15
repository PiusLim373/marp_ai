#!/usr/bin/python3

from marp_ai_gym import *
import numpy as np
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import os

def ppo_test():
    env = MarpAIGym(render_flag=True)  # Create the environment
    gym_env = DummyVecEnv([lambda: env])  # Use DummyVecEnv for vectorized environments
    cwd = os.getcwd()
    model_path = os.path.join(cwd, "ppo_marp_ai_model.zip")
    model = PPO.load(model_path)  # Load the trained model
    # Testing the agent
    n_episodes = 10  # Number of episodes to run
    scores = []  # Initialize a list to store episode scores

    for episode in range(n_episodes):
        observation = gym_env.reset()  # Reset the environment for a new episode
        done = False
        score = 0

        while not done:
            action, _states = model.predict(observation)  # Get the action from the model
            observation, reward, done, info = gym_env.step(action)  # Step the environment
            score += reward  # Accumulate score
            if done:
                print(f"Episode {episode + 1}: Reward= {score}")
                input("Press Enter to continue")

        scores.append(score)  # Add episode score to the scores list
        print(f"Episode {episode + 1}: Reward= {score}")

    # Calculate and print average reward
    average_reward = np.mean(scores)
    print(f"Average Reward over {n_episodes} episodes: {average_reward}")

    # End of the testing
    gym_env.close()  # Close the environment window if applicable

if __name__ == "__main__":
    ppo_test()
