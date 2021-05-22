from . import network
import datetime


def supply_demand_ratio(wds: network.WaterDistributionNetwork, driven_links):
    """
    Computes the supply/demand objective function at the end of the simulation (for DE algorithm)
    :param wds: network object
    :param driven_links: TODO: understand if we need to consider all links or only some pumps (PU1, PU2)
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
        r_demand = sum([wds.df_nodes_report.loc[time]['junctions', junc_id, 'demand'].sum() for junc_id in wds.junctions.uid])
        d_demand = sum([wds.df_links_report.loc[time][:, link_id, 'flow'].sum() for link_id in driven_links])
        #print(r_demand)
        #print(d_demand)
        tot_r = tot_r + r_demand
        tot_d = tot_d + d_demand

    print(tot_r)
    print(tot_d)
    ratio = tot_d / tot_r
    return ratio


def supply_demand_ratio2(wds: network.WaterDistributionNetwork, driven_links):
    if not wds.solved:
        return -1

    # TODO: check if we need actually the demand or something else
    tot_requested = sum([wds.df_nodes_report['junctions', junc_id, 'demand'].sum() for junc_id in wds.junctions.uid])
    tot_delivered = sum([wds.df_links_report['pumps', link_id, 'flow'].sum() for link_id in driven_links])

    #print(tot_requested)
    #print(tot_delivered)
    ratio = tot_delivered / tot_requested
    return ratio


def energy_consumption(wds: network.WaterDistributionNetwork):
    """
    Computes the total energy consumption at the end of the simulation
    :param wds: network object
    :return: total energy consumed in the current simulation
    """
    if not wds.solved:
        return -1
    total_energy = sum([wds.df_links_report[:, link_id, 'energy'].sum() for link_id in wds.links.uid])
    return total_energy


def tanks_feed_ratio(wds: network.WaterDistributionNetwork):
    """
    Computes the ratio between (junctions_demand) / (junctions_demand + total_tanks_flow), where the total_tanks_flow
    is determined by the sum of the absolute value of inflow and outflow for every tank.
    The objective function is taken from: https://github.com/BME-SmartLab/rl-wds
    :param wds: network object
    :return: feed ratio of the current simulation
    """
    if not wds.solved:
        return -1

    # TODO: understand if it makes sense to be used for the whole simulation as in DE and not only online (PROBABLY NOT)
    total_feed_ratio = 0
    # Transform wds.times (which are in seconds) to timedelta values
    times = [datetime.timedelta(seconds=time) for time in wds.times]
    for time in times:
        total_demand = sum([wds.df_nodes_report.loc[time]['junctions', junc_id, 'demand'] for junc_id in wds.junctions.uid])
        # TODO: problem with inflow-outflow and step-by-step simulation
        # total_tank_flow = sum([tank.inflow + tank.outflow for tank in wds.tanks])
        # total_feed_ratio += total_demand / (total_demand  + total_tank_flow)
    return total_feed_ratio
