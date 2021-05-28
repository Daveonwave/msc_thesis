if __name__ == '__main__':
    from scripts import network, epynetUtils, objFunction, differentialEvolution
    import datetime

    # Trials modifying init tanks level
    net = network.WaterDistributionNetwork("anytown_pd.inp")
    net.set_time_params(duration=86400, hydraulic_step=600)
    # print(net.tanks.tanklevel)
    net.run()
    objFunction.step_supply_demand_ratio(net, ['78','79'])
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
