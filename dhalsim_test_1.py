import sys

# Davide's epynet
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, 'scripts')

import network
import utils

import pandas as pd
import numpy as np
import csv

net = network.WaterDistributionNetwork("ctown.inp")
step = 300
duration = 3600
net.set_time_params(duration=duration, hydraulic_step=step)  # duration=24h, hydstep=10min

status = [1.0, 0.0]
actuators_status_dict = {uid: status for uid in net.pumps.uid.append(net.valves.uid)}
net.run(interactive=True, status_dict=actuators_status_dict)

no_step_nodes_report = net.df_nodes_report
no_step_links_report = net.df_links_report
no_step_df_tanks = no_step_nodes_report.iloc[:, no_step_nodes_report.columns.get_level_values(2)=='pressure']['tanks']
no_step_df_tanks = no_step_df_tanks.droplevel('properties', axis=1)
no_step_df_tanks = no_step_df_tanks.reset_index()
no_step_df_tanks.rename(columns={"index": "Time"})
no_step_df_tanks.to_csv('script_no_step_df_tanks.csv', index=False)

net = network.WaterDistributionNetwork("ctown.inp")
step = 300
duration = 3600
net.set_time_params(duration=duration, hydraulic_step=step)  # duration=24h, hydstep=10min

status = [1.0, 0.0]
actuators_status_dict = {uid: status for uid in net.pumps.uid.append(net.valves.uid)}

timestep = 1

simulation_time = 0
sim_values = []

net.init_simulation(interactive=True)

while timestep > 0:
    timestep, network_state = net.simulate_step(simulation_time, actuators_status_dict)
    simulation_time = simulation_time + timestep

net.create_df_reports()
step_nodes_results = net.df_nodes_report
step_links_results = net.df_links_report

step_df_tanks = step_nodes_results.iloc[:, step_nodes_results.columns.get_level_values(2)=='pressure']['tanks']

step_df_tanks = step_df_tanks.droplevel('properties', axis=1)
step_df_tanks = step_df_tanks.reset_index()
step_df_tanks.rename(columns={"index": "Time"})
step_df_tanks.to_csv('script_step_df_tanks.csv', index=False)