if __name__ == '__main__':
    from scripts import network, epynetUtils, objFunction, differentialEvolution
    import pandas as pd
    import datetime

    net = network.WaterDistributionNetwork("ctown_pd.inp")
    # Put report step equal to hyd_step because if it is lower it leads the timestep interval
    net.set_time_params(duration=3600*12, hydraulic_step=1200, report_step=1200)

    pattern_csv = "../demand_patterns/demands_anytown.csv"
    junc_demands = pd.read_csv(pattern_csv, names=['junc_demand'])
    pattern_dict = {col_name: junc_demands[col_name].values for col_name in junc_demands.columns}
    # net.set_demand_pattern('junc_demand', junc_demands.values, net.junctions)

    net.run()
    print(net.df_nodes_report['junctions'])

    ratio = objFunction.average_demand_deficit(net)
    print(ratio)
    exit(0)

    net.demand_model_summary()

    #for junc in net.junctions:
    #    print("basedemand: " + str(junc.basedemand) + ";    demand: " + str(junc.emitter))
    #exit(0)

    status = [1.0, 0.0]
    if net.valves:
        actuators_update_dict = {uid: status for uid in net.pumps.uid.append(net.valves.uid)}
    else:
        actuators_update_dict = {uid: status for uid in net.pumps.uid}

    net.run(interactive=True, status_dict=actuators_update_dict)

    print(net.tanks['T1'].inflow)
    exit(0)

    pumps_energy = sum([net.df_links_report['pumps', pump_id, 'energy'].sum() for pump_id in net.pumps.uid])
    print(pumps_energy)
    exit(0)

    times = [datetime.timedelta(seconds=time) for time in net.times]
    # net.set_basedemand_pattern(2)
    # net.set_time_params(duration=3600)
    # print(net.pumps.results.index)

    # print(net.df_links_report.iloc[:, net.df_links_report.columns.get_level_values(2) == 'status'])
    for time in times[:5]:
        print(str(time) + ": " +
              str(net.df_nodes_report.loc[time]['junctions', 'J147', 'demand']))
