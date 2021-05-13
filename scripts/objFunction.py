from . import network
import datetime


def supply_demand_ratio(wds: network.WaterDistributionNetwork):
    """
    Compute the supply/demand objective function at the end of the simulation (for DE algorithm)
    :param wds: network object
    :return: value of the objective function for the current simulation
    """
    if not wds.solved:
        return -1

    # Requested and delivered total water
    tot_r = 0
    tot_d = 0

    # Transform wds.times (which are in seconds) to timedelta values
    times = [datetime.timedelta(seconds=time) for time in wds.times]
    for time in times:
        # TODO: check if we need actually the demand or something else
        r_demand = sum([wds.df_nodes_report.loc[time]['junctions', junc_id, 'demand'] for junc_id in wds.junctions.uid])
        d_demand = sum([wds.df_links_report.loc[time][:, link_id, 'flow'] for link_id in wds.links.uid])
        tot_r = tot_r + r_demand
        tot_d = tot_d + d_demand

    # TODO: do we want to maximize the ratio between delivered / requested water?
    ratio = tot_d / tot_r
    return ratio


def energy_consumption(wds: network.WaterDistributionNetwork):
    """
    Compute the total energy consumption at the end of the simulation
    :param wds: network object
    :return: total energy consumed in the current simulation
    """
    if not wds.solved:
        return -1
    total_energy = sum([wds.df_links_report[:, link_id, 'energy'].sum() for link_id in wds.links.uid])
    return total_energy
