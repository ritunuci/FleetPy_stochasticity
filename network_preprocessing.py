import os
from click import option
import pandas as pd
import csv
from typing import Dict, Tuple

from scipy.stats import lognorm

network_dir = "D:/Ritun/Currently Developing/FleetPy_sto_tt_can_no_show_CALIBRATION_ver_sto_pudo_final/" \
              "data/networks/example_network"
scs_path = os.path.join(os.path.dirname(__file__), "studies", "example_study", "scenarios")
sc = os.path.join(scs_path, "example_depot.csv")
scenario_file_df = pd.read_csv(sc)

# op_mrng_peak = int(scenario_file_df["morning_hr_start"].iloc[0])
# op_evng_peak = int(scenario_file_df["evening_hr_start"].iloc[0])
#
# mrng_peak_tt_file_path = os.path.join(network_dir, f"time_{op_mrng_peak}", "operator_exp_tt.csv")
# evng_peak_tt_file_path = os.path.join(network_dir, f"time_{op_evng_peak}", "operator_exp_tt.csv")
#
# op_tt_buffers = scenario_file_df["operator_tt_buffers"][0].split(';')
# op_tt_buffers_list = [float(x) for x in op_tt_buffers]
#
# mrng_peak_tt_df = pd.read_csv(mrng_peak_tt_file_path)
# evng_peak_tt_df = pd.read_csv(evng_peak_tt_file_path)
#
# mrng_peak_tt_df["travel_time"] = mrng_peak_tt_df["travel_time"]*(1 + op_tt_buffers_list[0]/100)
# evng_peak_tt_df["travel_time"] = evng_peak_tt_df["travel_time"]*(1 + op_tt_buffers_list[1]/100)
#
# mrng_peak_tt_df.to_csv(mrng_peak_tt_file_path, index=False)
# evng_peak_tt_df.to_csv(evng_peak_tt_file_path, index=False)

tt_multipliers = scenario_file_df["operator_tt_buffers"][0].split(';')
tt_multipliers_list = [float(x) for x in tt_multipliers]

network_file_path = os.path.join(network_dir, "base", "edges.csv")
network_df = pd.read_csv(network_file_path)
mean_tt_list = []
for _, row in network_df.iterrows():
    scale = row["scale"] * tt_multipliers_list[0]
    shape = row["shape"] * tt_multipliers_list[1]
    loc = row["loc"]
    dist = lognorm(shape, loc=loc, scale=scale)
    mean = dist.mean()
    mean_tt_list.append(mean)

network_df["travel_time"] = mean_tt_list
network_df.to_csv(network_file_path, index=False)