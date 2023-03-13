import smart_nanogrid_gym
import argparse

import gym
import numpy as np
import os

from stable_baselines3.ddpg.policies import MlpPolicy
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3 import DDPG
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_checker import check_env
import time

env_variants = [
    {
        'variant_name': 'basic-',
        'config': {
            'vehicle_to_everything': False,
            'pv_system_available_in_model': False,
            'battery_system_available_in_model': False
        }},
    {
        'variant_name': 'b-pv-',
        'config': {
            'vehicle_to_everything': False,
            'pv_system_available_in_model': True,
            'battery_system_available_in_model': True
        }},
    {
        'variant_name': 'v2x-',
        'config': {
            'vehicle_to_everything': True,
            'pv_system_available_in_model': False,
            'battery_system_available_in_model': False
        }},
    {
        'variant_name': 'v2x-b-pv-',
        'config': {
            'vehicle_to_everything': True,
            'pv_system_available_in_model': True,
            'battery_system_available_in_model': True
        }}
]
current_env = env_variants[3]
current_env_name = current_env['variant_name']

models_dir = f"models/DDPG-{current_env_name}{int(time.time())}"
logdir = f"logs/DDPG-{current_env_name}{int(time.time())}"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)

current_env_configuration = current_env['config']
env = gym.make('SmartNanogridEnv-v0', **current_env_configuration)

# It will check your custom environment and output additional warnings if needed
check_env(env)
# the noise objects for DDPG
n_actions = env.action_space.shape[-1]
param_noise = None
action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))

model = DDPG(MlpPolicy, env, verbose=1, action_noise=action_noise, tensorboard_log=logdir)

TIMESTEPS = 20000
start = time.time()
for i in range(1, 50):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="DDPG")
    model.save(f"{models_dir}/{TIMESTEPS * i}")

env.close

end = time.time()

seconds = end - start
minutes = seconds / 60
hours = minutes // 60
minutes = (minutes/60 - hours) * 60

print(f'Training started: {start}\nTraining ended: {end}\nTraining lasted: {hours} h and {minutes} min')
# del model # remove to demonstrate saving and loading

# model = DDPG.load("ddpg_Chargym", env=env)
#
# mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
#
# # Enjoy trained agent
# obs = env.reset()
# for i in range(24):
#     action, _states = model.predict(obs, deterministic=True)
#     obs, rewards, dones, info = env.step(action)
#     # env.render(
