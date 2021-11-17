import pandas as pd
import numpy as np
import random
from mushroom_rl.core.environment import Environment, MDPInfo
from mushroom_rl.utils.spaces import Discrete, Box
from scripts import network, objFunction
from scripts.dqn.logger import Plotter


class WaterNetworkEnvironment(Environment):

    def __init__(self,
                 town='anytown_pd',
                 state_vars=None,
                 action_vars=None,
                 duration=3600 * 24,
                 hyd_step=300,
                 pattern_step=3600,
                 pattern_files=None,
                 seed=None,
                 update_every=None,
                 bounds=None,
                 logger=None,
                 show_plot=False,
                 ):
        """

        :param town:
        :param state_vars:
        :param action_vars:
        :param duration:
        :param hyd_step:
        :param pattern_step:
        :param pattern_files:
        """
        self.town = town
        self.state_vars = state_vars
        self.action_vars = action_vars

        self.duration = duration
        self.hyd_step = hyd_step
        self.pattern_step = pattern_step
        self.patterns_train = pattern_files['train']
        self.patterns_train_full_range_csv = pattern_files['train_full_range']
        self.patterns_train_low_csv = pattern_files['train_low']
        self.patterns_test_csv = pattern_files['test']
        self.demand_moving_average = None
        self.seed = seed

        self.update_every = update_every
        self.on_eval = False
        self.logger = logger
        self.show_plot = show_plot

        if self.show_plot:
            self.train_plot = Plotter('Training Plot')
            self.eval_plot = Plotter('Evaluation Plot')

        self.wn = network.WaterDistributionNetwork(town + '.inp')
        self.wn.set_time_params(duration=duration, hydraulic_step=hyd_step, pattern_step=pattern_step)

        self.curr_time = None
        self.timestep = None
        self.done = False
        self.total_updates = 0
        self.dsr = 0

        # Two possible values for each pump: 2 ^ n_pumps
        action_space = Discrete(2 ** len(self.action_vars))

        # Current state
        self._state = None

        # Bounds for observation space
        lows = np.array([bounds[key]['min'] for key in bounds.keys()])
        highs = np.array([bounds[key]['max'] for key in bounds.keys()])

        # Observation space
        observation_space = Box(low=lows, high=highs, shape=(len(bounds),))

        # TODO: what is horizon?
        mdp_info = MDPInfo(observation_space, action_space, gamma=0.99, horizon=1000000)
        super().__init__(mdp_info)

    def reset(self, state=None):
        """
        Called at the beginning of each episode
        :param state:
        :return:
        """
        self.wn.reset()

        if self.on_eval:
            if self.patterns_test_csv:
                junc_demands = pd.read_csv(self.patterns_test_csv)

                if self.seed and not (self.seed >= len(junc_demands.columns)):
                    col = junc_demands.columns.values[self.seed]
                else:
                    col = random.choice(junc_demands.columns.values)
                print("col: ", col)
                self.wn.set_demand_pattern('junc_demand', junc_demands[col], self.wn.junctions)
        else:
            if self.patterns_train:
                # Set pattern file choosing randomly between full range or low demand pattern
                junc_demands = pd.read_csv(self.patterns_train)
                col = random.choice(junc_demands.columns.values)
                self.wn.set_demand_pattern('junc_demand', junc_demands[col], self.wn.junctions)
                print("col: ", col)

        self.demand_moving_average = junc_demands.rolling(window=6, min_periods=1).mean()

        self.wn.init_simulation()
        self.curr_time = 0
        self.timestep = 1
        self.wn.solved = False
        self.done = False
        self.total_updates = 0
        self._state = self.build_current_state(reset=True)
        return self._state

    def step(self, action):
        """

        :param action:
        :return:
        """
        n_updates = 0

        # We want to update pump status every a certain amount of time
        if self.update_every:
            if self.curr_time % self.update_every == 0:
                # Action translated in binary value with one bit for each pump status
                new_status_dict = {pump_id: 0 for pump_id in self.action_vars}
                bin_action = '{0:0{width}b}'.format(action[0], width=len(self.action_vars))

                for i, key in enumerate(new_status_dict.keys()):
                    new_status_dict[key] = int(bin_action[i])

                n_updates = self.wn.update_actuators_status(new_status_dict)
                self.total_updates += n_updates
        # We want to update pump status at each timestep limiting the number of updates
        else:
            # Action translated in binary value with one bit for each pump status
            new_status_dict = {pump_id: 0 for pump_id in self.action_vars}
            bin_action = '{0:0{width}b}'.format(action[0], width=len(self.action_vars))

            for i, key in enumerate(new_status_dict.keys()):
                new_status_dict[key] = int(bin_action[i])

            n_updates = self.wn.update_actuators_status(new_status_dict)
            self.total_updates += n_updates

        # Simulate the next hydraulic step
        self.timestep = self.wn.simulate_step(self.curr_time, get_state=False)
        self.curr_time += self.timestep

        # Retrieve current state and reward from the chosen action
        self._state = self.build_current_state()
        reward = self.compute_reward(n_updates)

        info = None

        if self.timestep == 0:
            self.done = True
            self.wn.solved = True
            self.dsr = self.evaluate()
            #reward = self.compute_reward(n_actuators_updates=0)
            self.logger.results(self.dsr, self.total_updates)

        return self._state, reward, self.done, info

    def render(self):
        if self.done and self.show_plot:
            if self.on_eval:
                self.eval_plot.update(self.dsr)
            else:
                self.train_plot.update(self.dsr)

    def build_current_state(self, reset=False):
        """
        Build current state list, which can be used as input of the nn saved_models
        :param reset:
        :return:
        """
        state = []

        if reset:
            # Appending time, day, tanks level and junction pressure of nodes specified in yaml file
            for key in self.state_vars.keys():
                if key == 'time' or key == 'day':
                    state.append(self.state_vars[key])
                if key == 'tanks':
                    state.extend([0 for tank_id in self.state_vars[key]])
                if key == 'junctions':
                    state.extend([0 for junc_id in self.state_vars[key]])
                if key == 'demand_SMA':
                    state.append(self.demand_moving_average.iloc[0, 0])
        else:
            # Appending current daily timestamp and day of the week
            seconds_per_day = 3600 * 24
            days_per_week = 7
            current_hour = (self.wn.times[-1] % self.duration) // 3600

            # Appending current time, day, tanks level and junction pressure of nodes specified in yaml file
            for key in self.state_vars.keys():
                if key == 'time':
                    state.append(self.wn.times[-1] % seconds_per_day)
                if key == 'day':
                    state.append(((self.wn.times[-1] // seconds_per_day) % days_per_week) + 1)
                if key == 'tanks':
                    state.extend([self.wn.tanks[tank_id].results['pressure'][-1] for tank_id in self.state_vars[key]])
                if key == 'junctions':
                    state.extend([self.wn.junctions[junc_id].results['pressure'][-1] for junc_id in self.state_vars[key]])
                if key == 'demand_SMA':
                    state.append(self.demand_moving_average.iloc[current_hour, 0])

        #print(state)

        state = [np.float32(i) for i in state]
        return state

    def check_overflow(self):
        """
        Check if the we have an overflow problem in the tanks. We have an overflow if after one hour we the tank is
        still at the maximum level.
        :return: penalty value
        """
        penalty = 1
        risk_percentage = 0.9

        for tank in self.wn.tanks:
            if tank.results['pressure'][-1] > tank.maxlevel * risk_percentage:
                out_bound = tank.results['pressure'][-1] - (tank.maxlevel * risk_percentage)
                # Normalization of the out_bound pressure
                multiplier = out_bound / (tank.maxlevel - tank.maxlevel * risk_percentage)
                return penalty * multiplier
        return 0
        #     if len(tank.results['pressure']) > 5:
        #         last_levels = tank.results['pressure'][-5:-1]
        #         is_overflow = True
        #
        #         for level in last_levels:
        #             if level != tank.maxlevel:
        #                 is_overflow = False
        #                 break
        #
        #         if is_overflow:
        #             print("OVERFLOW!!!")
        #             return 50
        #         else:
        #             return 0
        # return 0

    def compute_reward(self, n_actuators_updates):
        """
        TODO: understand how to compute reward
        Compute the reward for the current step. It depends on the step_DSR and on the number of actuators updates.
        :param n_actuators_updates:
        :return:
        """
        if self.done:
            if self.dsr < 0.9:
                return -100
            else:
                return 0
            #    return 10 ** (1/self.dsr)

        overflow_penalty = self.check_overflow()
        #if not self.on_eval:
        #    print("OVERFLOW RISK: ", overflow_penalty)
        dsr_ratio = objFunction.step_supply_demand_ratio(self.wn)
        if self.update_every:
            return dsr_ratio - overflow_penalty
        else:
            reward = -n_actuators_updates/2 + dsr_ratio - overflow_penalty
            return reward

    def evaluate(self):
        """

        :return:
        """
        return objFunction.supply_demand_ratio(self.wn)

    def get_state(self):
        return self._state

