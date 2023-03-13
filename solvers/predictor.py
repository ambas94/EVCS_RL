import gym
import smart_nanogrid_gym

import gym
import numpy as np
import os
import argparse

from smart_nanogrid_gym.utils.config import solvers_files_directory_path

from stable_baselines3 import PPO
import time
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument("--env", default="SmartNanogridEnv-v0")
parser.add_argument("--reset_flag", default=1, type=int)
args = parser.parse_args()
env = gym.make(args.env)

# Define which model to load
model_name_ppo = "PPO-1676639715"
models_dir_ppo = f"{solvers_files_directory_path}\\RL\\models\\{model_name_ppo}"
model_path_ppo = f"{models_dir_ppo}\\9800"
model_ppo = PPO.load(model_path_ppo, env=env)

# Prediction has only 1 episode
episodes = 1

final_reward_PPO = [0]*episodes

for ep in range(episodes):
    print("episode = " + str(ep))
    rewards_list_PPO = []
    actions_list_PPO = []

    # PPO
    obs = env.reset(generate_new_initial_values=False)
    done = False
    step_counter = 0
    while not done:
        action, _states = model_ppo.predict(obs)
        obs, reward_PPO, done, info = env.step(action)
        rewards_list_PPO.append(reward_PPO)
        # print(f'step = {step_counter}')
        actions_list_PPO.append(action.tolist())
        step_counter += 1
    final_reward_PPO[ep] = sum(rewards_list_PPO)
    print(f"PPO rewards list: {rewards_list_PPO}")
    print(f"PPO actions list: {actions_list_PPO}")
# env.close()

Mean_reward_PPO = np.mean(final_reward_PPO)

print(f"Final PPO 2 reward: {final_reward_PPO}")

print(f"Mean PPO 2 reward: {Mean_reward_PPO}")

plt.rcParams["figure.figsize"] = (15, 10)
plt.rcParams.update({'font.size': 18})
plt.plot(final_reward_PPO, 'ro')
# plt.bar(final_reward_PPO)
plt.xlabel('Prediction episode')
plt.ylabel('Reward')
plt.legend(['PPO'])
plt.grid()

file_time = time.time()
plt.savefig(f"saved_figures\\figure_PPO_{int(file_time)}.png")

plt.show()

a = 1



