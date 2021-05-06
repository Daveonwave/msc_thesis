import epynet
import pandas as pd
import datetime
import utils

actuators_update_dict = {}
step_count = 0


class WaterDistributionNetwork(epynet.Network):
    """Class of the network inherited from Epynet.Network"""
    def __init__(self, inpfile: str):
        super().__init__(inputfile=inpfile)
        self.df_nodes_report = None
        self.df_links_report = None
        self.times = []
        self.interactive = False
        self.network_state = pd.Series()

    def set_time_params(self, duration=None, hydraulic_step=None, pattern_step=None, report_step=None, start_time=None,
                        rule_step=None):
        """
        Set the time parameters before the simulation (unit: seconds)
        :param duration: EN_DURATION
        :param hydraulic_step: EN_HYDSTEP
        :param pattern_step: EN_PATTERNSTEP
        :param report_step: EN_REPORTSTEP
        :param start_time: EN_STARTTIME
        :param rule_step: EN_RULESTEP
        """
        if duration is not None:
            self.ep.ENsettimeparam(utils.get_time_param_code('EN_DURATION'), duration)
        if hydraulic_step is not None:
            self.ep.ENsettimeparam(utils.get_time_param_code('EN_HYDSTEP'), hydraulic_step)
        if pattern_step is not None:
            self.ep.ENsettimeparam(utils.get_time_param_code('EN_PATTERNSTEP'), pattern_step)
        if report_step is not None:
            self.ep.ENsettimeparam(utils.get_time_param_code('EN_REPORTSTEP'), report_step)
        if start_time is not None:
            self.ep.ENsettimeparam(utils.get_time_param_code('EN_STARTTIME'), start_time)
        if rule_step is not None:
            self.ep.ENsettimeparam(utils.get_time_param_code('EN_RULESTEP'), rule_step)

    def set_demand_pattern(self, uid: str, values=None):
        """
        Set a base-demand pattern for junctions if exists, otherwise it creates and set a new pattern
        :param uid: pattern id
        :param values: list of multipliers, None if already existing
        """
        if values is None:
            if uid in self.patterns.uid:
                for junc in self.junctions:
                    junc.pattern = uid
            else:
                raise KeyError("Chosen pattern id doesn't exist")
        else:
            self.add_pattern(uid, values)
            for junc in self.junctions:
                junc.pattern = uid

    def demand_model_summary(self):
        """
        Print information related to the current demand model
        """
        dm_type, pmin, preq, pexp = self.ep.ENgetdemandmodel()
        if dm_type == 0:
            print("Running a demand driven analysis...")
        else:
            print("Running a pressure driven analysis...")
            print("-> Minimum pressure: {:.2f}".format(pmin))
            print("-> Required pressure: {:.2f}".format(preq))
            print("-> Exponential pressure: {:.2f}".format(pexp))

    def run(self, interactive=False, status_dict=None):
        """
        Run method wrapper to set the interactivity option (and others in the future related to RL)
        :param interactive: to update the actuators with own values
        :param status_dict: dictionary with predefined updates (just to test, it will be removed)
        TODO: remove status_dict
        """
        global actuators_update_dict
        if status_dict and interactive:
            actuators_update_dict = status_dict
            self.interactive = interactive
        else:
            self.interactive = False
        self.__run()

    def __run(self):
        """
        Step by step simulation: the idea is to put inside this function the online RL algorithm
        """
        self.init_simulation()
        curr_time = 0
        timestep = 1

        # timestep becomes 0 the last hydraulic step
        while timestep > 0:
            timestep, state = self.simulate_step(curr_time=curr_time)
            print(state['PU2'])
            curr_time += timestep

            # update the status of actuators after the first step
            if timestep != 0 and self.interactive:
                self.update_actuators_status()

        self.ep.ENcloseH()
        self.create_df_reports()

    def init_simulation(self):
        """
         Initialiaze the network simulation
        """
        self.reset()
        self.times = []
        self.ep.ENopenH()
        self.ep.ENinitH(flag=0)

    def simulate_step(self, curr_time):
        """
        Simulation of one step from the given time
        :param curr_time: current simulation time
        :return: time until the next event, if 0 the simulation is going to end
        """
        self.ep.ENrunH()
        timestep = self.ep.ENnextH()
        self.times.append(curr_time)
        self.load_attributes(curr_time)
        return timestep, self.get_network_state()

    def update_actuators_status(self, new_status=None):
        """
        Set actuators (pumps and valves) status to a new current state
        :param new_status: will be used in future with RL
        TODO: update with new_status
        """
        global step_count
        for uid in self.pumps.uid:
            self.pumps[uid].status = actuators_update_dict[uid][step_count % 2]
        for uid in self.valves.uid:
            self.valves[uid].status = actuators_update_dict[uid][step_count % 2]
        step_count += 1

    def get_network_state(self):
        """
        Retrieve the current values of the network in the form a pandas series of dictionaries.
        The collected values are referred to:
            - tanks: {pressure}
            - junctions: {pressure}
            - pumps: {status, flow}
            - valves: {status, flow}
        :return: the series with the above enlisted values
        """
        network_state = pd.Series()
        for uid in self.tanks.results.index.append(self.junctions.results.index):
            nodes_dict = {key: self.nodes[uid].results[key][-1] for key in ['pressure']}
            network_state[uid] = nodes_dict
        for uid in self.pumps.results.index.append(self.valves.results.index):
            links_dict = {key: self.links[uid].results[key][-1] for key in ['status', 'flow']}
            network_state[uid] = links_dict
        return network_state

    def create_df_reports(self):
        """
        Create nodes and links report dataframes - 3 level dataframe
        How to access: df['node', 'id', 'property'] -> column
        TODO: create a unique 4 level dataframe with 0 level distinguishing between node and link
        """
        if self.df_nodes_report is not None:
            del self.df_nodes_report
        if self.df_links_report is not None:
            del self.df_links_report

        tanks_ids = [uid for uid in self.tanks.uid]
        junctions_ids = [uid for uid in self.junctions.uid]
        tanks_iterables = [['tanks'], tanks_ids, ['head', 'pressure', 'demand']]
        junct_iterables = [['junctions'], junctions_ids, ['head', 'pressure', 'demand']]
        tanks_indices = pd.MultiIndex.from_product(iterables=tanks_iterables, names=["node", "id", "properties"])
        junctions_indices = pd.MultiIndex.from_product(iterables=junct_iterables, names=["node", "id", "properties"])

        # We use timestamp as index for both nodes and links dataframes
        times = [datetime.timedelta(seconds=time) for time in self.times]

        # Nodes dataframes creation
        df_tanks = pd.DataFrame(columns=tanks_indices, index=times)
        df_junctions = pd.DataFrame(columns=junctions_indices, index=times)

        # Dataframe filling
        for i, j in zip(df_tanks.columns.get_level_values(1), df_tanks.columns.get_level_values(2)):
            df_tanks['tanks', i, j] = self.tanks.results[i][j]
        for i, j in zip(df_junctions.columns.get_level_values(1), df_junctions.columns.get_level_values(2)):
            df_junctions['junctions', i, j] = self.junctions.results[i][j]

        self.df_nodes_report = pd.concat([df_tanks, df_junctions], axis=1)

        # We can assume that there is always at least one pump in each network, since would be pointless to study a wds
        # without this kind of links.
        pumps_ids = [uid for uid in self.pumps.uid]
        pumps_iterables = [['pumps'], pumps_ids, ['flow', 'energy', 'status']]
        pumps_indices = pd.MultiIndex.from_product(iterables=pumps_iterables, names=["link", "id", "properties"])
        df_pumps = pd.DataFrame(columns=pumps_indices, index=times)

        # Pump dataframe filling and columns renaming
        for i, j in zip(df_pumps.columns.get_level_values(1), df_pumps.columns.get_level_values(2)):
            df_pumps['pumps', i, j] = self.pumps.results[i][j]

        self.df_links_report = df_pumps

        # We cannot do the same assumption for valves, as we can see in "anytown" network
        if self.valves:
            valves_ids = [uid for uid in self.valves.uid]
            valves_iterables = [['valves'], valves_ids, ['velocity', 'flow', 'status']]
            valves_indices = pd.MultiIndex.from_product(iterables=valves_iterables, names=["link", "id", "properties"])

            df_valves = pd.DataFrame(columns=valves_indices, index=times)

            # Valves dataframe filling and columns renaming
            for i, j in zip(df_valves.columns.get_level_values(1), df_valves.columns.get_level_values(2)):
                df_valves['valves', i, j] = self.valves.results[i][j]

            self.df_links_report = pd.concat([df_pumps, df_valves], axis=1)


if __name__ == '__main__':
    net = WaterDistributionNetwork("ctown.inp")
    net.set_time_params(duration=3600, hydraulic_step=300)

    status = [1.0, 0.0]
    actuators_update_dict = {uid: status for uid in net.pumps.uid.append(net.valves.uid)}

    net.run(interactive=True, status_dict=actuators_update_dict)

    # net.set_basedemand_pattern(2)
    # net.set_time_params(duration=3600)
    # for pump in net.pumps:
    #     print(pump.results['status', 'flow'][1])

    curr_results = pd.Series(index=net.pumps.index)
    # print(net.pumps.results.index)

    # print(net.df_links_report['pumps', 'PU1'])
    # print(net.df_links_report.iloc[:, net.df_links_report.columns.get_level_values(2) == 'status'])
    # print(net.tanks.pressure)
    # print(net.junctions.results['22']['demand'])






