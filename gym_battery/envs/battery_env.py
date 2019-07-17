import gym
from gym.spaces import Discrete, Box

from os import path
import sys
sys.path.append(path.abspath('C:/Users/Administrator/PycharmProjects/BatteryAgent'))

from gym_battery.envs.env_assets.Tariff import Tariff
from gym_battery.envs.env_assets.Battery import Battery
from gym_battery.envs.env_assets.Bus import Bus
from gym_battery.envs.env_assets.IntervalReading import IntervalReading

import numpy as np

def empty_generator():
    yield from ()

class BatteryEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, xml_location=None, N_hours=24, N_actions=5):
        self.load = IntervalReading(xml_location) if xml_location is not None else None
        self.tariff = Tariff()
        self.bus = Bus()
        self.time = min(self.load.DF.start) if self.load is not None else None
        self.policy = None
        self.dispatch_penalty = 0.1
        self.convergence_flag = False

        self.episode_types = ['first_day', 'first_month', 'all_days', 'all_months']
        self.episode_type = self.episode_types[-1]

        self.action_space = Discrete(N_actions)
        self._N_actions = N_actions
        self.observation_space = Box(np.array([0, 0, 0, 0]), np.array([24, 1, 1, 1]))

        self.DF_gen = empty_generator()
        self.episode_DF = None

        self.verbose = False

        self.action_mapping = {}

        self.set_standard_system()

    def set_standard_system(self):
        from pathlib import Path

        # Point a Path object at the GreenButton data
        fdir = Path().absolute()
        fname = "pge_electric_interval_data_2011-03-06_to_2012-04-06 A10S Med Business Large Usage.xml"
        fpath = fdir / "batterydispatch" / "resources" / fname
        # fname = "pge_electric_interval_data_2011-03-06_to_2012-04-06 A10S Med Business Heavy Usage.xml"
        self.bus.add_battery(Battery(capacity=10000, power=2000))
        self.set_load(fpath)
        self.fit_load_to_space()

    def set_load(self, xml_location):
        self.load = IntervalReading(xml_location)

    def fit_load_to_space(self, N_hours = 24):
        max_load = self.load.get_max_load()
        min_load = self.load.get_min_load()
        battery_capacity = self.bus.battery.capacity
        battery_power = self.bus.battery.power
        self.observation_space = gym.spaces.Box(np.array([0, 0, min_load, min_load-battery_power]),
                                                  np.array([24, battery_capacity, max_load, max_load+battery_power]))

        num_as = self._N_actions
        action_values = (np.arange(num_as)-round(num_as/2, 0)) / round(num_as/2) * self.bus.battery.power
        self.action_mapping = dict(zip(np.arange(num_as), action_values))

    def step(self, action):

        action_value = self.action_mapping[action]

        # Note calc-grid_flow takes care of battery discharge and affects the state of charge.
        period = self.step_row.duration_hrs
        net_flow, _ = self.bus.calc_grid_flow(self.step_row.value,
                                              action_value,
                                              period,
                                              affect_state=True)

        # IF the net_flow exceeds the demand, then update the episodic demand
        demand = max(net_flow, self.state[3])

        self.grid_flow.loc[self.step_ix, 'net_flow'] = net_flow
        self.grid_flow.loc[self.step_ix, 'load'] = self.step_row.value
        self.grid_flow.loc[self.step_ix, 'battery_action'] = action_value
        self.grid_flow.loc[self.step_ix, 'state_of_charge'] = self.bus.battery.charge
        self.grid_flow.loc[self.step_ix, 'state'] = str(self.state)

        if self.verbose:
            print("hour: {},  load(state): {}, soc:{}, load(actual): {}, demand(state): {}, battery action: {}"
                  .format(self.step_row.start.hour,
                          self.state.load,
                          self.bus.battery.charge,
                          self.step_row.value,
                          self.state.demand,
                          action_value), end=" | ")

        if action_value <= 0:
            reward = 0
        else:
            reward = -1 * self.dispatch_penalty

        # Step into the next state after taking action
        episode_over = False
        try:
            self.step_ix, self.step_row = next(self.episode_step_generator)
            self.state = np.array([self.step_row.start.hour,
                                   self.bus.battery.charge,
                                   self.step_row.value,
                                   demand])
        except StopIteration:
            episode_over = True
            demand_charge = self.tariff.calculate_demand_charge(self.grid_flow, 'net_flow')
            energy_charge = self.tariff.calculate_energy_charge(self.grid_flow, 'net_flow') * \
                            (1 if 'month' in self.episode_type else 30)
            reward -= (demand_charge + energy_charge)

        return (self.state, reward, episode_over, {})

    def reset(self, episode_type = None, random_charge = True):
        # set the initial state
        if random_charge:
            starting_charge = self.observation_space.sample()[1]
        else:
            starting_charge = 0
        self.state = np.array([self.observation_space.low[0],
                               starting_charge,
                               self.observation_space.low[2],
                               max(self.observation_space.low[3], 0)])

        self.bus.battery.charge = self.state[2]

        # Set the episode type if it has been provided to be changed:
        if episode_type is not None:
            self.episode_type = episode_type


        # Get the DF_epsidoe from teh generator function which is load for this episode.
        try:
            # If there are bill perios left in the DF_gen, then pop the next one
            self.episode_DF = next(self.DF_gen)
        except StopIteration:
            # If not, then reload the data based on the day/month parameter and whether doing the first or all.
            if 'day' in self.episode_type:
                self.DF_gen = self.load.get_daily_generator()
            elif 'month' in self.episode_type:
                self.DF_gen = self.load.get_month_generator()
            else:
                raise ValueError(f"{self.episode_type} not a reocgnized episode type from {self.episode_types}")

            self.episode_DF = next(self.DF_gen)

            # Check to see if we should just run the first and if so set the empty generator
            if 'first' in self.episode_type:
                self.DF_gen = empty_generator() # This will for recreation of the generator, repeating hte same day/month
            elif 'all' in self.episode_type:
                pass
            else:
                raise ValueError(
                    f"'first' and 'all' not found in {self.episode_type}, should be one of {self.episode_types}")

        # Set up the episode generators and trackers
        self.grid_flow = self.episode_DF.copy(deep=True)
        self.episode_step_generator = self.episode_DF.iterrows()
        self.step_ix, self.step_row = next(self.episode_step_generator)

        # Set the initial state
        self.state = np.array([self.step_row.start.hour,
                               starting_charge,
                               self.step_row.value,
                               self.step_row.value])

        return self.state


    def render(self, mode='human'):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError