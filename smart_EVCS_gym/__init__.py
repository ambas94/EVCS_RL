from gym.envs.registration import registry, register, make, spec


register(
     id='SmartNanogridEnv-v0',
     entry_point='smart_nanogrid_gym.envs:SmartNanogridEnv',
     max_episode_steps=200,
)
