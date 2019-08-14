import numpy as np
import gym
from gym.spaces import Box

env = gym.make('gym_battery:battery-v0')

env.set_standard_system()

env.set_stochastic_generator()
env.episode_type = 'count_days'
env.run_N_episodes=30
env.load.DF = env.load.DF[76:]
env.fit_load_to_space()
env.reset()
env.step(2)

# Set how to structure the environment. 'count_days' will generate the a single day as an episode. THe number of days
# given indicates how many differnet days to use.
# This needs to be changed so that it generates LONGER episodes, not DIFFERENT episodes, but this hasn't been done yet.
env.episode_type = 'count_days'
env.run_N_episodes = 1

env.observation_space = Box(np.array([0, 0, 3000, 1000]),np.array([24, env.bus.battery.capacity, 7000, 6500]))

env.reset()
env.step(0)
env.step(0)
env.step(0)


env.reset()
env.step(0)
env.step(0)
env.step(0)