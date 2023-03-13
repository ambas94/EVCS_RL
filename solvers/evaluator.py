import gym
import smart_nanogrid_gym

import gym
import numpy as np
import os
import argparse
from solvers.RBC.rbc import RBC

from stable_baselines3 import PPO
import time
import matplotlib.pyplot as plt
from smart_nanogrid_gym.utils.config import solvers_files_directory_path


parser = argparse.ArgumentParser()
parser.add_argument("--env", default="SmartNanogridEnv-v0")
parser.add_argument("--reset_flag", default=1, type=int)
args = parser.parse_args()
env = gym.make(args.env)

# Define which model to load
# PPO Model 1
model_name1 = "PPO-1676639715"
models_dir1 = f"{solvers_files_directory_path}\\RL\\models\\{model_name1}"
model_path1 = f"{models_dir1}\\9800"
model1 = PPO.load(model_path1, env=env)

# How many evaluations
# episodes = 150
episodes = 50

final_reward_PPO_1 = [0]*episodes
final_reward_rbc = [0]*episodes

for ep in range(episodes):
    # print("episode = " + str(ep))
    rewards_list_PPO_1 = []
    rewards_list_rbc = []

    # PPO
    obs = env.reset(generate_new_initial_values=True)
    done = False
    while not done:
        action, _states = model1.predict(obs)
        obs, reward_PPO, done, info = env.step(action)
        rewards_list_PPO_1.append(reward_PPO)
    final_reward_PPO_1[ep] = sum(rewards_list_PPO_1)

    # RBC case
    obs = env.reset(generate_new_initial_values=False)
    done = False
    while not done:
        action_rbc = RBC.select_action(env.env, obs)
        obs, rewards_rbc, done, _ = env.step(action_rbc)
        rewards_list_rbc.append(rewards_rbc)
    final_reward_rbc[ep] = sum(rewards_list_rbc)

    if ep == episodes - 1:
        print(f"PPO 1 rewards list: {rewards_list_PPO_1}")
        print(f"RBC rewards list: {rewards_list_rbc}")
env.close()

Mean_reward_PPO_1 = np.mean(final_reward_PPO_1)
Mean_reward_RBC = np.mean(final_reward_rbc)

print(f"Final PPO 1 reward: {final_reward_PPO_1}")
print(f"Final RBC reward: {final_reward_rbc}")

print(f"Mean PPO 1 reward: {Mean_reward_PPO_1}")
print(f"Mean RBC reward: {Mean_reward_RBC}")

plt.rcParams["figure.figsize"] = (15, 10)
plt.rcParams.update({'font.size': 18})
plt.plot(final_reward_PPO_1)
plt.plot(final_reward_rbc)
plt.xlabel('Evaluation episodes')
plt.ylabel('Reward')
plt.legend(['PPO_1', 'RBC'])
plt.grid()

file_time = time.time()
plt.savefig(f"saved_figures\\figure_PPO_RBC_{int(file_time)}.png")

plt.show()

a = 1



