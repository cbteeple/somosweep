import sys

from gym.envs.registration import register

default_max_episode_steps = 100

register(
    id="SoMoGymExampleEnv-v0",
    entry_point="environments.SoMoGymExampleEnv.SoMoGymExampleEnv:SoMoGymExampleEnv",
    max_episode_steps=default_max_episode_steps,
)
