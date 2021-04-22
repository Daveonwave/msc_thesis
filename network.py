import epynet
import pandas as pd
import datetime
import utils


class WaterDistributionNetwork(epynet.Network):
    """Class of the network inherited from Epynet.Network"""
    def __init__(self, inpfile: str):
        super().__init__(inputfile=inpfile)
        self.df_nodes_report = None
        self.df_links_report = None
        self.times = []

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

    # TODO: maybe useless
    def delete_control_rules(self):
        """
        Deletes, if any, all the rule-based controls retrieved from .inp file
        """
        return

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

    def run(self):
        """
        Step by step simulation: the idea is to put inside this function the online RL algorithm
        """
        self.reset()
        self.times = []
        self.ep.ENopenH()
        self.ep.ENinitH(flag=0)

        curr_time = 0
        timestep = 1

        # timestep becomes 0 the last hydraulic step
        while timestep > 0:
            # status of the links and nodes
            # TODO: manage the status/speed of pumps avoiding control rules
            self.ep.ENrunH()
            timestep = self.ep.ENnextH()
            self.times.append(curr_time)
            self.load_attributes(curr_time)
            curr_time += timestep

        self.ep.ENcloseH()
        self.create_df_reports()

    def load_attributes(self, simtime):
        """
        Override of the original method because it doesn't save pump status and pump speed for the current
        simulation time
        """
        for node in self.nodes:
            for property_name in node.properties.keys():
                if property_name not in node.results.keys():
                    node.results[property_name] = []
                # clear cached values
                node._values = {}
                node.results[property_name].append(node.get_property(node.properties[property_name]))
            node.times.append(simtime)

        for link in self.links:
            for property_name in link.properties.keys():
                if property_name not in link.results.keys():
                    link.results[property_name] = []
                # clear cached values
                link._values = {}
                link.results[property_name].append(link.get_property(link.properties[property_name]))

            for property_name in link.static_properties.keys():
                if property_name in ["speed", "initstatus"]:
                    if property_name not in link.results.keys():
                        link.results[property_name] = []
                    # clear cached values
                    link._values = {}
                    link.results[property_name].append(link.get_property(link.static_properties[property_name]))
            link.times.append(simtime)

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
        pumps_iterables = [['pumps'], pumps_ids, ['flow', 'energy', 'speed', 'initstatus']]
        pumps_indices = pd.MultiIndex.from_product(iterables=pumps_iterables, names=["link", "id", "properties"])
        df_pumps = pd.DataFrame(columns=pumps_indices, index=times)

        # Pump dataframe filling and columns renaming
        for i, j in zip(df_pumps.columns.get_level_values(1), df_pumps.columns.get_level_values(2)):
            df_pumps['pumps', i, j] = self.pumps.results[i][j]

        self.df_links_report = df_pumps

        # We cannot do the same assumption for valves, as we can see in "anytown" network
        if self.valves:
            valves_ids = [uid for uid in self.valves.uid]
            valves_iterables = [['valves'], valves_ids, ['velocity', 'flow', 'initstatus']]
            valves_indices = pd.MultiIndex.from_product(iterables=valves_iterables, names=["link", "id", "properties"])

            df_valves = pd.DataFrame(columns=valves_indices, index=times)

            # Valves dataframe filling and columns renaming
            for i, j in zip(df_valves.columns.get_level_values(1), df_valves.columns.get_level_values(2)):
                df_valves['valves', i, j] = self.valves.results[i][j]

            self.df_links_report = pd.concat([df_pumps, df_valves], axis=1)


if __name__ == '__main__':
    net = WaterDistributionNetwork("anytown_pd.inp")
    net.set_time_params(duration=3600, rule_step=3600)
    net.demand_model_summary()
    net.ep.ENsetdemandmodel(0, 0, 0, 0)
    net.demand_model_summary()
    net.ep.ENsetdemandmodel(1, 0, 0.5, 0.5)
    net.demand_model_summary()
    net.run()


    # net.set_basedemand_pattern(2)
    # net.set_time_params(duration=3600)

    print(net.df_nodes_report)
    # print(net.tanks.pressure)
    # print(net.junctions.results['22']['demand'])






