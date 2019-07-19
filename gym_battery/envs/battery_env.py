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

import pkg_resources

def empty_generator():
    yield from ()

class BatteryEnv(gym.Env):
    '''This environment represents a load with a battery. Each episode is a period of load data; the length of an
    episode is determined by the episode_type parameter, which should be of the registered episode_types.
    In general, an episode lasts either a day or a month.

    The state space is [hour of the day, state of charge, load, and current demand], and is defined as a Box space.

    The action space is discreate, with N possible actions. Per gym convension, the Discrete mapping is index values;
    the actual battery instructions are interpreted by action_mapping.

    The same day/month can be repeated, if first_ is set as the type. This allows agents to rapidly learn but
    is over-fitting. If all_ is set as the type, all periods in the load data provided to the enviornment will be run,
    sequentially, as separate episodes. Finally, count_ can be set, which runs the number of episodes defined by
    the run_N_episodes parameters (i.e. setting run_N_episodes == 2 will run two days, as two separate episodes, each
    time reset is called.

    The system comprises of a battery and a load, which are joined by a bus. Based on the load and battery action
    during a given time step, the bus object manages the calculation of net flow (energy supplied by the grid), and
    dispatches the battery. The battery dispatch will affect the state of charge.

    For testing purposes, set_standard_system is built into the __init__ method.'''
    metadata = {'render.modes': ['human']}

    def __init__(self, xml_location=None, N_hours=24, N_actions=5):

        # Load, tariff, and bus manage the representation of the actual system.
        self.load = IntervalReading(xml_location) if xml_location is not None else None
        self.tariff = Tariff()
        self.bus = Bus()
        self.time = min(self.load.DF.start) if self.load is not None else None

        # Dispatch penalty can be applied to all discharge actions. This slightly discourages the agent from discharging,
        # all other things being equal.
        self.dispatch_penalty = 0.1

        # The episode_type, which should be one of the registered episode_types, controls how episodes are generated.
        self.episode_types = ['first_day', 'first_month', 'all_days', 'all_months', 'count_days', 'count_months']
        self.episode_type = self.episode_types[-1]

        # Define the gym action and observation spaces, and the action_mapping,
        # determines what a discrete action means in terms of battery discharge/charge in kW
        self.action_space = Discrete(N_actions)
        self._N_actions = N_actions
        self.action_mapping = {}
        self.observation_space = Box(np.array([0, 0, 0, 0]), np.array([24, 1, 1, 1]))

        # DF_gen is the load profile generator; it returns the next day or month of load, episode_DF,
        # depending on episode_type
        self.DF_gen = empty_generator()
        self.episode_DF = None

        # episode_step_generator is a pandas iterrows generator, which steps through episode_DF.
        self.episode_step_generator = empty_generator()
        self.step_row = None # pd.Series, returned by episode_step_generator
        self.step_ix = int # row index, returned by episode_ste_generator

        # if count_ episode_type is used, this tracks how many episodes have occured.
        self.count_episodes = -1
        self.run_N_episodes = None

        # When the episode is over, this defines what is returned as the 'state' following action A from state S.
        self.terminal_state = 'terminal'

        self.verbose = False

        # Set up the standard system - for testing.
        print("setting the standard system, A10S Med busines large usage with a 2,000kW/10,000kWh battery")
        self.set_standard_system()

    def set_standard_system(self):
        '''This method establishes the basic system used for the project, based on the A10S Med Business Large Usage
        load profile, and with a 10,000kWh, 2000kW battery.'''

        import os

        # Point a Path object at the GreenButton data
        fdir = os.path.normpath(os.path.dirname(__file__)) + "\\env_assets\\resources\\"
        fname = "pge_electric_interval_data_2011-03-06_to_2012-04-06 A10S Med Business Large Usage.xml"
        # fname = "pge_electric_interval_data_2011-03-06_to_2012-04-06 A10S Med Business Heavy Usage.xml"

        fpath = fdir + fname

        self.bus.add_battery(Battery(capacity=10000, power=2000))
        self.set_load(fpath)
        self.fit_load_to_space()

    def set_load(self, xml_location):
        '''This sets the load based on the xml GreenButton data, which should be available in the location provided.'''
        self.load = IntervalReading(xml_location)

    def fit_load_to_space(self, N_hours = 24):
        '''This function scales the observation space and action space to the parameters of the specific system,
        as defined by the bus, the battery, and the load.'''
        max_load = self.load.get_max_load()
        min_load = self.load.get_min_load()
        battery_capacity = self.bus.battery.capacity
        battery_power = self.bus.battery.power
        self.observation_space = gym.spaces.Box(np.array([0, 0, min_load, min_load]),
                                                  np.array([24, battery_capacity, max_load, max_load+battery_power]))

        num_as = self._N_actions
        action_values = (np.arange(num_as)-round(num_as/2, 0)) / round(num_as/2) * self.bus.battery.power
        self.action_mapping = dict(zip(np.arange(num_as), action_values))

    def step(self, action):
        '''Step takes an action in the environment, resolves the impact of that action, calculates the reward,
        and returns the next state.
        Takes: an action index (in the action_space and with interpretation defined in action_mapping).
        Gives: (new_state, reward, episode_over_flag, {details}'''

        action_value = self.action_mapping[action]

        # Note calc-grid_flow takes care of battery discharge and affects the state of charge.
        period = self.step_row.duration_hrs
        net_flow, _ = self.bus.calc_grid_flow(self.step_row.value,
                                              action_value,
                                              period,
                                              affect_state=True)

        # IF the net_flow exceeds the demand, then update the episodic demand
        demand = max(net_flow, self.state[3])

        # Record the behavior of the system for rendering
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

        # If the action_value is discharging, then apply the dispatch penalty if defined.
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
            return (self.state, reward, episode_over, {})

        except StopIteration:
            # Then the episode_step_generator is depleted, indicating the period, and the episode, is over
            episode_over = True

            # Calculate the final reward
            demand_charge = self.tariff.calculate_demand_charge(self.grid_flow, 'net_flow')
            energy_charge = self.tariff.calculate_energy_charge(self.grid_flow, 'net_flow') * \
                            (1 if 'month' in self.episode_type else 30)
            reward -= (demand_charge + energy_charge)
            return (self.terminal_state, reward, episode_over, {})



    def reset(self, episode_type = None, random_charge = True):
        '''Reset the environment to start a new episode.
        Takes: a new episode_type, if the episode_type should change, and whether to initialize with a random state of
        charge or 0. '''
        # set the initial state
        if random_charge:
            starting_charge = self.observation_space.sample()[1]
        else:
            starting_charge = self.observation_space.low[1]

        # Set the initial state
        self.state = np.array([self.observation_space.low[0],
                               starting_charge,
                               self.observation_space.low[2],
                               max(self.observation_space.low[3], 0)])

        # Set the bus battery charge based on the initial state
        self.bus.battery.charge = self.state[2]

        # Set the episode type if it has been provided to be changed:
        if episode_type is not None:
            self.episode_type = episode_type

        # Get the DF_episode from the episode load generator function, which gives the time and load for the period.
        # In order to generate episodes indefinitely, if the load generator function is empty, it is re-initialized
        # using the load generator defined by episode_type
        try:
            # If there are bill periods left in the DF_gen, then pop the next one
            self.episode_DF = next(self.DF_gen)
        except StopIteration:
            # If not, then reload the data based on the day/month parameter and whether doing the first or all.
            if 'day' in self.episode_type:
                self.DF_gen = self.load.get_daily_generator()
            elif 'month' in self.episode_type:
                self.DF_gen = self.load.get_month_generator()
            else:
                raise ValueError(f"{self.episode_type} not a reocgnized episode type from {self.episode_types}")

            # finally, we can get the episodid load
            self.episode_DF = next(self.DF_gen)

            # If the first_ flag is set, we want to go back to get a daily generator each episode, so we empty the
            # episode generator; this forces a StopIteration next reset.
            if 'first' in self.episode_type:
                self.DF_gen = empty_generator() # This will for recreation of the generator, repeating hte same day/month

        # If the user has set a count, then we monitor this and empty the generator when needed.
        if 'count' in self.episode_type:
            if self.run_N_episodes is None:
                raise ValueError("Set run_N_episodes before running count episode types.")
            if self.count_episodes == -1 or self.count_episodes >= self.run_N_episodes:
                self.DF_gen = empty_generator()
                self.count_episodes = 1
            else:
                self.count_episodes += 1

        # Set up the episode generators and trackers
        self.grid_flow = self.episode_DF.copy(deep=True)
        self.episode_step_generator = self.episode_DF.iterrows()

        # Start the episode.
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