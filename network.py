import epynet
import pandas as pd
import datetime
import utils


class WaterDistributionNetwork(epynet.Network):
    """Class of the network inherited from Epynet.Network"""
    def __init__(self, inpfile: str):
        super().__init__(inputfile=inpfile)
        self.df_junctions_report = None
        self.df_pumps_report = None
        self.df_pumps_report = None
        self.times = []

    def set_time_params(self, duration=None, hydraulic_step=None, pattern_step=None, report_step=None, start_time=None):
        """
        Set the time parameters before the simulation (unit: seconds)
        :param duration: EN_DURATION
        :param hydraulic_step: EN_HYDSTEP
        :param pattern_step: EN_PATTERNSTEP
        :param report_step: EN_REPORTSTEP
        :param start_time: EN_STARTTIME
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

    def set_basedemand_pattern(self, uid: str, values=None):
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

    def run(self):
        """
        Step by step simulation: the idea is to put inside this function the online RL algorithm
        """
        self.reset()
        self.ep.ENopenH()
        self.ep.ENinitH(flag=0)

        curr_time = 0
        timestep = 1

        # timestep becomes 0 the last hydraulic step
        while timestep > 0:
            self.ep.ENrunH()
            timestep = self.ep.ENnextH()
            self.times.append(curr_time)
            self.load_attributes(curr_time)
            curr_time += timestep

        self.ep.ENcloseH()
        self.create_junctions_report()
        self.create_pumps_report()

    def create_junctions_report(self):
        """Build a dataframe of the report of junctions properties"""
        iterables = [self.junctions.uid, ['head', 'pressure', 'demand']]
        cols_index = pd.MultiIndex.from_product(iterables=iterables, names=["id", "properties"])

        times = [datetime.timedelta(seconds=time) for time in self.times]
        self.df_junctions_report = pd.DataFrame(0, index=times, columns=cols_index)

        for i, j in zip(self.df_junctions_report.columns.get_level_values(0),
                        self.df_junctions_report.columns.get_level_values(1)):
            self.df_junctions_report[i, j] = self.junctions.results[i][j]

    def create_pumps_report(self):
        """Build a dataframe of the report of junctions properties"""
        iterables = [self.pumps.uid, ['flow', 'energy']]
        cols_index = pd.MultiIndex.from_product(iterables=iterables, names=["id", "properties"])

        times = [datetime.timedelta(seconds=time) for time in self.times]
        self.df_pumps_report = pd.DataFrame(0, index=times, columns=cols_index)

        for i, j in zip(self.df_pumps_report.columns.get_level_values(0),
                        self.df_pumps_report.columns.get_level_values(1)):
            self.df_pumps_report[i, j] = self.pumps.results[i][j]


if __name__ == '__main__':
    net = WaterDistributionNetwork("anytown.inp")
    # net.set_basedemand_pattern(2)
    net.set_time_params(duration=3600)
    net.run()

    print(net.df_junctions_report)
    #print(net.junctions.results['22']['demand'])




