import ast
from colorama import Fore, Style
import time
import csv
import os
import subprocess
import glob
import logging
from pathlib import Path
import json

import numpy as np
import pandas as pd
from collections import defaultdict
import torch
import matplotlib.pyplot as plt

from ax.service.ax_client import AxClient, ObjectiveProperties
from ax.service.utils.report_utils import exp_to_df
from ax.generation_strategy.generation_strategy import GenerationStrategy, GenerationStep
from ax.adapter.registry import Generators
# from ax.storage.json_store.save import save_experiment

# ----------------------------------------------------------------------------------------------------------------------
# Paths and setup
# ----------------------------------------------------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
calibration_study_folder_name = "calibration_study_final_sc_3_BO_try_1"

fleetpy_input_file_path = PROJECT_ROOT / "studies/example_study/scenarios/example_depot.csv"
fleetpy_const_config_input_file_path = PROJECT_ROOT / "studies/example_study/scenarios/constant_config_depot_cali_sc_1.csv"
fleetpy_demand_folder_path_main = PROJECT_ROOT / "data/demand/example_demand/matched/example_network"
fleetpy_edge_input_file_path = PROJECT_ROOT / "data/networks/example_network/base/edges.csv"
# fleetpy_output_folder_path = PROJECT_ROOT / "studies/example_study/results/example_depot_time_pool_irsonly_sc_1"
fleetpy_replication_output_path = PROJECT_ROOT / "studies/example_study/results/"
raw_requests_folder_path = PROJECT_ROOT / "raw_rq_files_3_3_ODT_bins"           # Change this

cali_output_folder_path = PROJECT_ROOT / "calibration_outputs" / calibration_study_folder_name
os.makedirs(cali_output_folder_path, exist_ok=True)

ODT_bins_std_folder_path = cali_output_folder_path / "files_for_ODT_bins_std_cal"
os.makedirs(ODT_bins_std_folder_path, exist_ok=True)

# ax_checkpoint_path = cali_output_folder_path / "ax_experiment_checkpoint.json"
ax_client_checkpoint_path = cali_output_folder_path / "ax_client_checkpoint.json"
optimization_intermediate_path = cali_output_folder_path / "optimization_result_intermediate.csv"
result_intermediate_path = cali_output_folder_path / "result_intermediate.csv"

veh_folder = PROJECT_ROOT / "data/fleetctrl"
active_fleetsize_path = PROJECT_ROOT / "data/fleetctrl/elastic_fleet_size/example_active_fleetsize.csv"
real_data_dir = PROJECT_ROOT / "output_for_calibration/real_data"

days = ["tuesday", "wednesday", "thursday"]
# days = ["tuesday"]
city_id = 77            # 77 for Cupertino
num_replications = 10   # Variable not used

# # Get unique bin lists
# rq_df = pd.read_csv("D:/Ritun/Cupertino_travel_time_calibration_files/2024_rq_file_with_3_by_3_ODT_bins.csv")
# unique_bins = rq_df["odt_bin"].unique()
# unique_bins_1 = list(unique_bins)
# # Sort the list using the evaluated tuple as the key
# unique_bins_1.sort(key=ast.literal_eval)

# unique_bins_2 = []
# for bin in unique_bins_1:
#     unique_bins_2.append(bin + '_mean')
#     unique_bins_2.append(bin + '_std')

KPI_COLS = [
    "percentage_offer_created",
    "percentage_of_offer_declined_by_rider",
    "rides_per_veh_rev_hr",
    "wait_time_morning_peak",
    "wait_time_evening_peak",
    "cancellation_rate_percent",
    # "median_cancellation_dur",
]                            # Here put the KPI column names same as in the real data files
# KPI_COLS = KPI_COLS + unique_bins_2

# ----------------------------------------------------------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------------------------------------------------------
log_file_path = cali_output_folder_path / "run_log.txt"
log_file_path.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file_path, mode="a"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------------------------------------------------
# Optimization settings
# ----------------------------------------------------------------------------------------------------------------------
N_TRAIN_SOBOL = 35           # Number of Sobol sample trial #25
N_BATCH = 50                 # number of BO trials after Sobol warm start #75
TOTAL_TRIALS = N_TRAIN_SOBOL + N_BATCH
BATCH_SIZE = 1               # keep sequential generation/evaluation

# ----------------------------------------------------------------------------------------------------------------------
# Convergence monitoring settings
# ----------------------------------------------------------------------------------------------------------------------
# Because the Ax objective is total_negative_ES and minimize=False, larger values are better.
# Equivalently, the original Energy Score is improving when total_negative_ES increases.

CONVERGENCE_TOL = 3e-2       # minimum meaningful improvement in total_negative_ES # 1e-2
CONVERGENCE_PATIENCE = 10    # number of BO trials allowed without meaningful improvement # 5
ENABLE_EARLY_STOPPING = False # set False if you only want to monitor convergence without stopping

# RANDOM_SEEDS_FOR_FLEETPY = [42]   # Just 1 replication
RANDOM_SEEDS_FOR_FLEETPY = []
for i in range(42, 47):
    RANDOM_SEEDS_FOR_FLEETPY.append(i)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

param_names = [
    "op_max_wait_time",
    "op_detour_time_factor",
    "const_PUDO_duration",
    "cancel_wait_time_bound_lower",
    "max_cancel_decision_time",
    "random_term_variance",
    "decision_threshold",
    "no_show_wait_time",
    "CV_group_1_peak",
    "CV_group_1_offpeak",
    "CV_group_2_peak",
    "CV_group_2_offpeak",
    "CV_group_3_peak",
    "CV_group_3_offpeak"
]

param_bounds = {
    "op_max_wait_time": (1500.0, 4500.0),
    "op_detour_time_factor": (20.0, 100.0),
    "const_PUDO_duration": (30.0, 600.0),
    "cancel_wait_time_bound_lower": (200.0, 700.0),
    "max_cancel_decision_time": (20.0, 180.0),
    "random_term_variance": (1.5, 6.0),
    "decision_threshold": (30.0, 100.0),
    "no_show_wait_time": (20.0, 180.0),
    "CV_group_1_peak": (0.05, 0.5),
    "CV_group_1_offpeak": (0.01, 0.3),
    "CV_group_2_peak": (0.05, 0.6),
    "CV_group_2_offpeak": (0.01, 0.25),
    "CV_group_3_peak": (0.05, 1.0),
    "CV_group_3_offpeak": (0.01, 0.4),
}


ax_parameters = [
    {"name": name, "type": "range", "bounds": [lb, ub], "value_type": "float"}
    for name, (lb, ub) in param_bounds.items()
]

# ----------------------------------------------------------------------------------------------------------------------
# Bookkeeping
# ----------------------------------------------------------------------------------------------------------------------
all_inputs = []
all_objectives = []
objective_cache = {}

# Per-component raw tracking (all negative, for Ax maximisation)
raw_metrics_cache = {}    # input_tuple → (weekly_ES_neg, pooled_tt_ES_neg, ODT_err_neg)

# Sobol-phase normalisation
sobol_raw_metrics = []    # list of (weekly_ES_neg, pooled_tt_ES_neg, ODT_err_neg) per Sobol trial
sobol_stds = None         # {"weekly_ES": std, "pooled_tt_ES": std, "ODT_std_err": std}
sobol_stds_path = cali_output_folder_path / "sobol_stds.json"
ax_client_rebuilt_flag_path = cali_output_folder_path / "ax_client_rebuilt.flag"

# Convergence trace objects. These are updated once per completed Ax trial.
convergence_log = []
best_so_far = -np.inf
no_improve_count = 0

timing_log_path = cali_output_folder_path / "indiv_call_timings_eval.csv"
convergence_log_path = cali_output_folder_path / "convergence_log.csv"

# ----------------------------------------------------------------------------------------------------------------------
# Extra utility functions for Scenario 2 calibration
# ----------------------------------------------------------------------------------------------------------------------
def safe_mean(series, default=600):
    val = series.mean()
    return default if pd.isna(val) else val

def safe_std(series, default=120):
    val = series.std()
    return default if pd.isna(val) else val

# ----------------------------------------------------------------------------------------------------------------------
# Simulation / KPI functions
# ----------------------------------------------------------------------------------------------------------------------
def run_fleetpy_replications(file_name, day, active_fleetsize_path):
    # runs one FleetPy replication and returns raw KPI vector
    try:
        subprocess.run(["/data/homezvol2/rituns/.conda/envs/fleetpy_hpc/bin/python", "run_examples.py"], check=True)
        # subprocess.run(["C:/Users/ritun_uci/.conda/envs/fleetpy_Ax_client/python", "run_examples.py"], check=True)
    except Exception as e:
        # print(f"{Fore.GREEN}FleetPy failed for {str(file_name)} of {day}: {e},\n "
        #         f"Retrying with fleet fraction = 0.5 for all times...{Style.RESET_ALL}")
        logger.info(f"{Fore.GREEN}FleetPy failed for {str(file_name)} of {day}: {e},\n "
                f"Retrying with fleet fraction = 0.5 for all times...{Style.RESET_ALL}")
        df_fix = pd.read_csv(active_fleetsize_path)
        df_fix.iloc[:, 1] = 0.5
        df_fix.to_csv(active_fleetsize_path, index=False)
        time.sleep(1)
        subprocess.run(["/data/homezvol2/rituns/.conda/envs/fleetpy_hpc/bin/python", "run_examples.py"], check=True)
        # subprocess.run(["C:/Users/ritun_uci/.conda/envs/fleetpy_Ax_client/python", "run_examples.py"], check=True)
        logger.info(f"{Fore.GREEN}FleetPy retry successful for {str(file_name)} of {day} with fleet fraction = 0.5 for all times{Style.RESET_ALL}")
    time.sleep(1)



def read_FleetPy_replications_outputs_and_build_kpi_vec(pooled_tt_mean_dict, file_name):
    # reads the outputs of all replications for a date and builds a array of KPI vectors
    file_Z_samples_raw = []
    for rs in RANDOM_SEEDS_FOR_FLEETPY:
        df = pd.read_csv(fleetpy_replication_output_path / f"example_depot_time_pool_irsonly_rs_{rs}" / "standard_eval.csv", index_col=0, header=0)
        df_user_stats = pd.read_csv(fleetpy_replication_output_path / f"example_depot_time_pool_irsonly_rs_{rs}" / "1_user-stats.csv")
        # Save one seed with a unique suffix so calculate_ODT_std_error pools all trips.
        if rs == 42:
            df_user_stats.to_csv(
                ODT_bins_std_folder_path / f"{Path(file_name).stem}_rs_{rs}.csv",
                index=False,
            )
        df_user_stats["travel_time"] = df_user_stats["dropoff_time"] - df_user_stats["pickup_time"]

        pooled_tt_mean_dict[rs]["sim"].append((len(df_user_stats["travel_time"]), df_user_stats["travel_time"].sum()))

        total_demand = len(df_user_stats["request_id"])
        num_riders_declined = df_user_stats["rider_declined"].sum()
        num_offers_created = total_demand * float(df.loc["% created offers"]["MoD_0"])/100.0
        per_of_offers_declined_by_rider = (num_riders_declined / num_offers_created)*100 if num_offers_created > 0 else 0.0
        
        kpi_vec = np.array([
            float(df.loc["% created offers"]["MoD_0"]),               # % offer created
            per_of_offers_declined_by_rider,
            float(df.loc["rides per veh rev hours"]["MoD_0"]),        # rides per veh revenue hr
            float(df.loc["waiting time for morning"]["MoD_0"]) if not pd.isna(df.loc["waiting time for morning"]["MoD_0"]) else 800,        # wait time morning peak
            float(df.loc["waiting time for evening"]["MoD_0"]) if not pd.isna(df.loc["waiting time for evening"]["MoD_0"]) else 800,        # wait time evening peak
            float((df_user_stats["diffusion_cancelled"].sum()/num_offers_created)*100) if num_offers_created > 0 else 0.0,        # % of offers cancelled
        ], dtype=float)
        file_Z_samples_raw.append(kpi_vec)

    return file_Z_samples_raw, pooled_tt_mean_dict



def calculate_pooled_tt_energy_score(pooled_tt_mean_dict, real_tt_mean_dict):
    tt_Z_samples_sim = []
    for rs in RANDOM_SEEDS_FOR_FLEETPY:
        total_travel_time = 0
        num_travelers = 0
        for item in pooled_tt_mean_dict[rs]["sim"]:
            num_travelers += item[0]  # number of travelers from simulation
            total_travel_time += item[1]  # total travel time from simulation

        tt_kpi_vec = np.array([float(total_travel_time/num_travelers)], dtype=float)   # average travel time per traveler
        tt_Z_samples_sim.append(tt_kpi_vec)
    tt_Z_samples_sim = np.vstack(tt_Z_samples_sim)       # From the simulated replications
    
    total_real_travel_time = 0
    num_real_travelers = 0
    for key, value in real_tt_mean_dict.items():
        num_real_travelers += value[0]  # number of travelers from real data
        total_real_travel_time += value[1]
    tt_real_vec = np.array([float(total_real_travel_time/num_real_travelers)], dtype=float)   # average travel time per traveler

    tt_energy_score = compute_energy_score_file(tt_Z_samples_sim, tt_real_vec)

    return tt_energy_score



def calculate_ODT_std_error(ODT_bins_std_folder_path):
    pooled_tt_ODT_bin_dict = defaultdict(lambda: {"sim": [], "real": []})
    csv_files = glob.glob(str(ODT_bins_std_folder_path / "*.csv"))
    for file in csv_files:
        df = pd.read_csv(file)
        df["travel_time"] = df["dropoff_time"] - df["pickup_time"]
        df = df.dropna(subset=["travel_time", "odt_bin"])
        for _, row in df.iterrows():
            odt_bin = row["odt_bin"]
            tt = row["travel_time"]
            pooled_tt_ODT_bin_dict[odt_bin]["sim"].append(tt)

    for day in days:
        fleetpy_demand_folder_path = fleetpy_demand_folder_path_main / day
        demand_csv_files = glob.glob(str(fleetpy_demand_folder_path / "*.csv"))
        for file in demand_csv_files:
            p = Path(file)
            file_name = p.name
            df = pd.read_csv(file)
            df = df.dropna(subset=["actual_travel_time", "odt_bin"])
            for _, row in df.iterrows():
                odt_bin = row["odt_bin"]
                tt = row["actual_travel_time"]
                pooled_tt_ODT_bin_dict[odt_bin]["real"].append(tt)

    abs_diff_sum = 0.0
    num_bins = 0
    for odt_bin, data in pooled_tt_ODT_bin_dict.items():
        if data["sim"] and data["real"]:
            sim_std  = np.std(data["sim"],  ddof=1) if len(data["sim"])  > 1 else 0.0
            real_std = np.std(data["real"], ddof=1) if len(data["real"]) > 1 else 0.0
            abs_diff_sum += abs(sim_std - real_std)
            num_bins += 1

    if num_bins == 0:
        return None

    return abs_diff_sum / num_bins



def compute_day_wise_standardized_mu_sigma_from_real(day):
    """
    mu_k, sigma_k computed over all observed (real) dates for a particualr day of the week (for exmaple: for all the dates having Monday), KPI-wise.
    Returns (mu, sigma) arrays of shape (4,).
    """
    rows = []
    df_real = pd.read_csv(real_data_dir / f"{day}_real.csv")
    mat = df_real[KPI_COLS].to_numpy(dtype=float)
    # rows.append(mat)
    # Y = numpy.vstack(rows)  # (num_dates_total, 4)
    Y = mat
    mu = np.nanmean(Y, axis=0)
    sigma = np.nanstd(Y, axis=0, ddof=1)  # sample std with ddof=1
    sigma[sigma < 1e-12] = 1.0  # avoid divide-by-zero
    
    return mu, sigma



def zscore_standardize(vec, mu, sigma):
    vec = np.asarray(vec, dtype=float)
    mu = np.asarray(mu, dtype=float).reshape(-1)
    sigma = np.asarray(sigma, dtype=float).reshape(-1)
    
    return (vec - mu) / sigma



def compute_energy_score_file(Z_samples, z_obs):
    """
    Z_samples: array (S, d) standardized simulated KPI vectors for day/date i
    z_obs:     array (d,) standardized observed KPI vector for day/date i
    Implements Eq (7) in the doc
    """
    Z = np.asarray(Z_samples, dtype=float)  # (S,d)
    z = np.asarray(z_obs, dtype=float)      # (d,)

    # drop any rows with NaNs
    mask = np.isfinite(Z).all(axis=1) & np.isfinite(z).all()
    if not mask.any():
        raise ValueError("No valid Z_samples for Energy Score")
    Z = Z[mask]
    S = Z.shape[0]
    if S < 2:
        raise ValueError("Energy Score needs at least S>=2 replications")

    # Term1 = (1/S) * sum ||Z_s - z||
    term1 = np.mean(np.linalg.norm(Z - z[None, :], axis=1))

    # Term2 = (1/(2*S*(S-1))) * sum_{s!=t} ||Z_s - Z_t||
    # vectorized pairwise distances
    diff = Z[:, None, :] - Z[None, :, :]           # (S,S,d)
    D = np.linalg.norm(diff, axis=2)               # (S,S)
    term2 = D.sum() / (2.0 * S * (S - 1))          # includes diagonal zeros, ok

    return float(term1 - term2)


# def compute_error_for_file(Z_samples, z_obs):           # This function is specifically used only in Scenario 2 calibration
#     """
#     Z_samples: array (S, d) standardized simulated KPI vectors for day/date i
#     z_obs:     array (d,) standardized observed KPI vector for day/date i
#     Implements Eq (7) in the doc
#     """
#     Z = np.asarray(Z_samples, dtype=float)  # (S,d)
#     z = np.asarray(z_obs, dtype=float)      # (d,)

#     # Term1 = (1/S) * sum ||Z_s - z||
#     term1 = np.mean(np.linalg.norm(Z - z[None, :], axis=1))

#     return float(term1)


def run_program_day(day, pooled_tt_mean_dict, real_tt_mean_dict):
    logger.info(f"Running program for {day}")
    day_Energy_Score_list = []
    mu, sigma = compute_day_wise_standardized_mu_sigma_from_real(day)
    df_real = pd.read_csv(real_data_dir / f"{day}_real.csv")
    day_real_dict = {}
    for _, row in df_real.iterrows():
        date_key = pd.to_datetime(row["date"]).date()
        day_real_dict[date_key] = row[KPI_COLS].to_numpy(dtype=float)

    fleetpy_demand_folder_path = fleetpy_demand_folder_path_main / day
    fleetpy_input_df = pd.read_csv(fleetpy_const_config_input_file_path, index_col=0)
    fleetpy_input_df.loc["day_dir_name", "Parameter_Value"] = str(day)
    fleetpy_input_df.to_csv(fleetpy_const_config_input_file_path)
    time.sleep(1)

    demand_csv_files = glob.glob(str(fleetpy_demand_folder_path / "*.csv"))
    for file in demand_csv_files:
        # file_Z_samples_raw = []
        p = Path(file)
        file_name = p.name
        file_name_without_ext = p.stem
        date = pd.to_datetime(file_name_without_ext).date()
        veh_file = veh_folder / f"{city_id}_unique_vehid" / day / file_name
        veh_file_df = pd.read_csv(veh_file)
        veh_list = []
        active_fleetsize_dict = {}
        for _, row in veh_file_df.iterrows():
            for i in list(range(7, 19)):   # from 7am to 6pm
                veh_list.append(max(int(row[f"Hour_{i}"]), 1))
                active_fleetsize_dict[i*3600] = int(row[f"Hour_{i}"])
        fleet_size = max(veh_list)
        for key, value in active_fleetsize_dict.items():
            active_fleetsize_dict[key] = value/fleet_size
        active_fleetsize_df = pd.DataFrame({"time": active_fleetsize_dict.keys(),
                                            "share_active_fleet_size": active_fleetsize_dict.values()})
        active_fleetsize_df.to_csv(active_fleetsize_path, index=False)
        fleetpy_input_file_df = pd.read_csv(fleetpy_const_config_input_file_path, index_col=0)
        fleetpy_input_file_df.loc["rq_file", "Parameter_Value", ] = str(file_name)
        fleetpy_input_file_df.loc["op_fleet_composition", "Parameter_Value", ] = f"default_vehtype:{fleet_size}"
        fleetpy_input_file_df.to_csv(fleetpy_const_config_input_file_path)
        print(f"{Fore.GREEN}Running Fleetpy for: {str(file_name)} file of {day} {Style.RESET_ALL}")
        # logger.info(f"Running Fleetpy for: {str(file_name)} file of {day}")			# Comment this later
        time.sleep(1)
        run_fleetpy_replications(file_name, day, active_fleetsize_path)
        file_Z_samples_raw, pooled_tt_mean_dict = read_FleetPy_replications_outputs_and_build_kpi_vec(pooled_tt_mean_dict, file_name)
        file_Z_samples_raw = np.vstack(file_Z_samples_raw)       # From the simulated replications
        file_z_real_raw = day_real_dict[date]
        # Standardize both simulated and observed KPI vectors
        file_Z_samples_standardized = zscore_standardize(file_Z_samples_raw, mu, sigma)
        file_z_real_standardized = zscore_standardize(file_z_real_raw, mu, sigma)
        Energy_Score_file = compute_energy_score_file(file_Z_samples_standardized, file_z_real_standardized)
        day_Energy_Score_list.append(Energy_Score_file)
        # weekly_energy_score_list.append(day_Energy_Score_list)
        # df_raw_rq = pd.read_csv(raw_requests_folder_path / day / file_name)
        df_demand = pd.read_csv(fleetpy_demand_folder_path / file_name)
        real_tt_mean_dict[file_name_without_ext] = (len(df_demand["actual_travel_time"]), df_demand["actual_travel_time"].sum())
    return day_Energy_Score_list, pooled_tt_mean_dict, real_tt_mean_dict


def update_csv_with_train_x(values):
    # op_max_wait, detour_factor, pudo_dur, cancel_wait_time_bound_upper, cancel_wait_time_bound_lower, max_cancel_decision_time, random_term_variance, decision_threshold, no_show_wait_time, cv1_pk, cv1_off, cv2_pk, cv2_off, cv3_pk, cv3_off = values
    op_max_wait, detour_factor, pudo_dur, cancel_wait_time_bound_lower, max_cancel_decision_time, random_term_variance, decision_threshold, no_show_wait_time, cv1_pk, cv1_off, cv2_pk, cv2_off, cv3_pk, cv3_off = values

    input_csv_path = fleetpy_const_config_input_file_path
    input_csv_df = pd.read_csv(input_csv_path, index_col=0)
    input_csv_df.loc["op_max_wait_time", "Parameter_Value"] = float(op_max_wait)
    input_csv_df.loc["op_max_detour_time_factor", "Parameter_Value"] = float(detour_factor)
    input_csv_df.loc["op_const_boarding_time", "Parameter_Value"] = float(pudo_dur)
    input_csv_df.loc["waiting_time_upper_bound", "Parameter_Value"] = float(op_max_wait)
    input_csv_df.loc["waiting_time_lower_bound", "Parameter_Value"] = float(cancel_wait_time_bound_lower)
    input_csv_df.loc["max_decision_time", "Parameter_Value"] = float(max_cancel_decision_time)
    input_csv_df.loc["random_term_variance", "Parameter_Value"] = float(random_term_variance)
    input_csv_df.loc["decision_threshold_upper", "Parameter_Value"] = float(decision_threshold)
    input_csv_df.loc["decision_threshold_lower", "Parameter_Value"] = -float(decision_threshold)
    input_csv_df.loc["no_show_wait_time", "Parameter_Value"] = float(no_show_wait_time)

    edge_csv_df = pd.read_csv(fleetpy_edge_input_file_path)
    peak_cv_list = []
    offpeak_cv_list = []
    for _, row in edge_csv_df.iterrows():
        if row["highway_group"] == "group_1":
            peak_cv_list.append(cv1_pk)
            offpeak_cv_list.append(cv1_off)
        elif row["highway_group"] == "group_2":
            peak_cv_list.append(cv2_pk)
            offpeak_cv_list.append(cv2_off)
        else:
            peak_cv_list.append(cv3_pk)
            offpeak_cv_list.append(cv3_off)
    
    edge_csv_df["peak_cv"] = peak_cv_list
    edge_csv_df["offpeak_cv"] = offpeak_cv_list
    edge_csv_df.to_csv(fleetpy_edge_input_file_path, index=False)
    input_csv_df.to_csv(input_csv_path)
    time.sleep(1)


def collect_output():
    # Remove all files in ODT_bins_std_folder_path
    for f in glob.glob(str(ODT_bins_std_folder_path / "*")):
        try:
            os.remove(f)
        except Exception as e:
            logger.warning(f"Could not remove file {f}: {e}")

    weekly_energy_score_list = []
    pooled_tt_mean_dict = defaultdict(lambda: {"sim": []})
    real_tt_mean_dict = {}
    for day in days:
        day_Energy_Score_list, pooled_tt_mean_dict, real_tt_mean_dict = run_program_day(day, pooled_tt_mean_dict, real_tt_mean_dict)
        day_average_Energy_Score = np.mean(day_Energy_Score_list)
        weekly_energy_score_list.append(day_average_Energy_Score)
    pooled_tt_energy_score = calculate_pooled_tt_energy_score(pooled_tt_mean_dict, real_tt_mean_dict)
    ODT_std_error = calculate_ODT_std_error(ODT_bins_std_folder_path)
    ODT_std_error_safe = ODT_std_error if ODT_std_error is not None else 0.0
    return (-float(np.mean(weekly_energy_score_list)), -pooled_tt_energy_score, -ODT_std_error_safe)


def send_instruction(train_x):
    train_x_array = train_x.numpy()
    all_outputs_list = []
    # other_metrics = []
    for values in train_x_array:
        update_csv_with_train_x(values)
        # output = collect_output()
        # all_outputs_list.append(output)
        weekly_energy_score, pooled_tt_energy_score, ODT_std_error = collect_output()

    return (weekly_energy_score, pooled_tt_energy_score, ODT_std_error)


def compute_and_save_sobol_stds():
    """
    Compute per-component std from all completed Sobol trials and persist to disk.
    Called once, at the transition from Sobol to BO.
    Guards against constant components by replacing zero stds with 1.0.
    """
    global sobol_stds
    if len(sobol_raw_metrics) < 2:
        logger.warning("Fewer than 2 Sobol trials available; using unit stds for normalisation.")
        sobol_stds = {"weekly_ES": 1.0, "pooled_tt_ES": 1.0, "ODT_std_err": 1.0}
    else:
        arr = np.array(sobol_raw_metrics, dtype=float)   # (N_sobol, 3)
        stds = np.std(arr, axis=0, ddof=1)
        stds = np.where(stds < 1e-12, 1.0, stds)
        sobol_stds = {
            "weekly_ES":    float(stds[0]),
            "pooled_tt_ES": float(stds[1]),
            "ODT_std_err":  float(stds[2]),
        }
    with open(sobol_stds_path, "w") as fp:
        json.dump(sobol_stds, fp, indent=2)
    logger.info(f"Sobol-phase normalisation stds computed and saved: {sobol_stds}")


def rebuild_ax_client_for_bo():
    """
    Build a fresh BO-only AxClient and re-attach every Sobol point with its
    rescaled total_negative_error so the GP surrogate fits on a single
    consistent scale.

    Also rewrites in-memory state and disk artifacts atomically so that a
    crash between the rebuild and the first BO trial does not leave stale
    raw-sum values on disk:
      - objective_cache and all_objectives (in place)
      - result_intermediate.csv
      - convergence_log.csv  (best_so_far columns recomputed on new scale)
      - best_so_far global
    Creates a flag file so resume after this point skips the rebuild.
    """
    assert sobol_stds is not None, "sobol_stds must be computed before rebuild"

    gs_bo_only = GenerationStrategy(
        steps=[
            GenerationStep(
                generator=Generators.BOTORCH_MODULAR,
                num_trials=-1,
                max_parallelism=1,
            ),
        ]
    )
    new_ax_client = AxClient(generation_strategy=gs_bo_only, verbose_logging=True)
    new_ax_client.create_experiment(
        name="single_obj_exp",
        parameters=ax_parameters,
        objectives={"total_negative_error": ObjectiveProperties(minimize=False)},
    )

    # Rescale every Sobol observation and re-attach to the new client.
    # Deduplicate to avoid attaching the same parameter point twice (cache-hit duplicates).
    seen_keys = set()
    for x_sorted in all_inputs:
        key = tuple(x_sorted)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        if key not in raw_metrics_cache:
            logger.warning(f"rebuild: key {key} missing from raw_metrics_cache — skipping.")
            continue
        weekly_ES, pooled_tt_ES, ODT_err = raw_metrics_cache[key]
        scaled = (
            weekly_ES                                                       # This was initially devided by sobol_stds["weekly_ES"]
            + pooled_tt_ES / sobol_stds["pooled_tt_ES"]
            + ODT_err / sobol_stds["ODT_std_err"]
        )
        objective_cache[key] = scaled
        params_dict = {name: val for name, val in zip(param_names, x_sorted)}
        _, trial_idx = new_ax_client.attach_trial(parameters=params_dict)
        new_ax_client.complete_trial(
            trial_index=trial_idx,
            raw_data={"total_negative_error": (scaled, None)},
        )

    # Rewrite all_objectives in place so result_intermediate.csv reflects scaled values
    for k, x_sorted in enumerate(all_inputs):
        all_objectives[k] = objective_cache[tuple(x_sorted)]

    # Persist the rescaled all_objectives to result_intermediate.csv so that a
    # resume between rebuild and the first BO trial sees a consistent scale.
    _save_result_intermediate()

    # Rewrite convergence_log entries on the rescaled scale, recompute
    # best_so_far columns, and persist. This keeps best_so_far / plots
    # continuous across the Sobol→BO transition and prevents a stale
    # best_so_far from being loaded on a between-rebuild-and-BO resume.
    global best_so_far
    running_best = -np.inf
    for entry in convergence_log:
        x_sorted = [entry[p] for p in param_names]
        key = tuple(x_sorted)
        if key in raw_metrics_cache:
            weekly_ES, pooled_tt_ES, ODT_err = raw_metrics_cache[key]
            scaled = (
                weekly_ES                                                   # This was initially devided by sobol_stds["weekly_ES"]
                + pooled_tt_ES / sobol_stds["pooled_tt_ES"]
                + ODT_err / sobol_stds["ODT_std_err"]
            )
            entry["objective_total_negative_error"] = scaled
            entry["energy_score"] = -scaled
            if scaled > running_best:
                running_best = scaled
            entry["best_so_far_total_negative_error"] = running_best
            entry["best_so_far_energy_score"] = -running_best
            entry["improvement_over_previous_best"] = np.nan  # ambiguous post-rescale; clear it
    best_so_far = running_best if np.isfinite(running_best) else -np.inf
    pd.DataFrame(convergence_log).to_csv(convergence_log_path, index=False)
    logger.info(
        f"Rewrote {len(convergence_log)} convergence_log entries on rescaled scale. "
        f"best_so_far reset to {best_so_far:.6f}."
    )

    new_ax_client.save_to_json_file(filepath=str(ax_client_checkpoint_path))
    ax_client_rebuilt_flag_path.touch()
    logger.info(
        f"AxClient rebuilt with {len(seen_keys)} rescaled Sobol observations. "
        f"Flag written to {ax_client_rebuilt_flag_path}."
    )
    return new_ax_client


# ----------------------------------------------------------------------------------------------------------------------
# Ax-facing evaluation wrapper
# ----------------------------------------------------------------------------------------------------------------------
def evaluate_objectives(x_sorted, iteration_type):
    start_time = time.time()
    input_data = torch.tensor(x_sorted, dtype=torch.double).unsqueeze(0)

    logger.info(f"{Fore.RED}send_instruction is being called for input {input_data}{Style.RESET_ALL}")
    logger.info(f"{iteration_type}: evaluating input {x_sorted}")

    weekly_ES, pooled_tt_ES, ODT_err = send_instruction(input_data)

    duration = time.time() - start_time
    file_exists = os.path.isfile(timing_log_path)

    with open(timing_log_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["iteration_type", "evaluation_time_(sec)"])
        writer.writerow([iteration_type, duration])

    logger.info(
        f"{iteration_type}: weekly_ES={weekly_ES:.6f}, pooled_tt_ES={pooled_tt_ES:.6f}, "
        f"ODT_err={ODT_err:.6f}, eval_time_sec={duration:.2f}"
    )
    return float(weekly_ES), float(pooled_tt_ES), float(ODT_err)


def _save_result_intermediate():
    df_inputs_live = pd.DataFrame(all_inputs, columns=param_names)
    raw = [raw_metrics_cache.get(tuple(inp), (np.nan, np.nan, np.nan)) for inp in all_inputs]
    raw_arr = np.array(raw, dtype=float)
    df_obj_live = pd.DataFrame({
        "total_negative_error": all_objectives,
        "total_error_score":    [-v for v in all_objectives],
        "weekly_ES_neg":        raw_arr[:, 0],
        "pooled_tt_ES_neg":     raw_arr[:, 1],
        "ODT_std_err_neg":      raw_arr[:, 2],
    })
    pd.concat([df_inputs_live, df_obj_live], axis=1).to_csv(
        cali_output_folder_path / "result_intermediate.csv", index=False
    )


def f_total(parameters, iteration_type):
    global sobol_raw_metrics

    x_sorted = [parameters[p_name] for p_name in param_names]
    input_tuple = tuple(x_sorted)

    is_sobol = iteration_type.startswith("training_itr_")

    if input_tuple not in objective_cache:
        logger.info(f"f_total evaluating: {input_tuple}")

        weekly_ES, pooled_tt_ES, ODT_err = evaluate_objectives(x_sorted, iteration_type)
        raw_metrics_cache[input_tuple] = (weekly_ES, pooled_tt_ES, ODT_err)

        if is_sobol:
            sobol_raw_metrics.append((weekly_ES, pooled_tt_ES, ODT_err))

        if sobol_stds is None:
            # Sobol phase: equal (unit) weights — Ax normalises internally anyway
            output = weekly_ES + pooled_tt_ES + ODT_err
        else:
            output = (
                weekly_ES                                               # This was initially devided by sobol_stds["weekly_ES"]
                + pooled_tt_ES / sobol_stds["pooled_tt_ES"]
                + ODT_err / sobol_stds["ODT_std_err"]
            )

        objective_cache[input_tuple] = output
        all_inputs.append(x_sorted)
        all_objectives.append(output)
        _save_result_intermediate()
        logger.info(
            f"f_total result: {output:.6f} "
            f"(weekly_ES={weekly_ES:.6f}, pooled_tt_ES={pooled_tt_ES:.6f}, ODT_err={ODT_err:.6f})"
        )
    else:
        logger.info(f"f_total retrieving from cache: {input_tuple}")
        weekly_ES, pooled_tt_ES, ODT_err = raw_metrics_cache[input_tuple]
        if sobol_stds is None:
            output = weekly_ES + pooled_tt_ES + ODT_err
        else:
            output = (
                weekly_ES                                               # This was initially devided by sobol_stds["weekly_ES"]
                + pooled_tt_ES / sobol_stds["pooled_tt_ES"]
                + ODT_err / sobol_stds["ODT_std_err"]
            )
        objective_cache[input_tuple] = output  # refresh in case scale changed after rebuild
        all_inputs.append(x_sorted)
        all_objectives.append(output)
        _save_result_intermediate()
        logger.info(f"f_total cached result (recomputed): {output:.6f}")

    return float(objective_cache[input_tuple])

# ----------------------------------------------------------------------------------------------------------------------
# Convergence monitoring utilities
# ----------------------------------------------------------------------------------------------------------------------
def update_convergence_log(
    trial_number,
    trial_index,
    stage_name,
    iteration_type,
    parameters,
    value,
):
    """
    Updates and saves the convergence trace for one completed trial.

    Since Ax is maximizing total_negative_ES, convergence is monitored using the
    best-so-far value of total_negative_ES. The corresponding Energy Score is
    simply -total_negative_ES, so lower Energy Score is better.
    """
    global best_so_far, no_improve_count

    previous_best = best_so_far
    improvement = value - previous_best

    # if value > best_so_far:          # always update the true incumbent
    #     best_so_far = value

    if value > previous_best + CONVERGENCE_TOL:
        best_so_far = value
        no_improve_count = 0
        improved = True
    else:
        no_improve_count += 1
        improved = False

    row = {
        "trial_number": trial_number,
        "trial_index": trial_index,
        "stage": stage_name,
        "iteration_type": iteration_type,
        "objective_total_negative_error": value,
        "energy_score": -value,
        "best_so_far_total_negative_error": best_so_far,
        "best_so_far_energy_score": -best_so_far,
        "improvement_over_previous_best": improvement,
        "improved": improved,
        "no_improve_count": no_improve_count,
    }
    row.update(parameters)
    convergence_log.append(row)

    df_conv = pd.DataFrame(convergence_log)
    df_conv.to_csv(convergence_log_path, index=False)

    print(
        f"{Fore.CYAN}"
        f"Convergence | trial={trial_number}, "
        f"stage={stage_name}, "
        f"objective={value:.6f}, "
        f"energy_score={-value:.6f}, "
        f"best_objective={best_so_far:.6f}, "
        f"best_energy_score={-best_so_far:.6f}, "
        f"improvement={improvement:.6f}, "
        f"no_improve_count={no_improve_count}"
        f"{Style.RESET_ALL}"
    )
    logger.info(
        f"Convergence | trial={trial_number}, stage={stage_name}, "
        f"objective={value:.6f}, energy_score={-value:.6f}, "
        f"best_objective={best_so_far:.6f}, best_energy_score={-best_so_far:.6f}, "
        f"improvement={improvement:.6f}, no_improve_count={no_improve_count}"
    )

    return no_improve_count


def plot_convergence_curves():
    """
    Saves convergence plots for both the Ax objective and the original Energy Score.
    The vertical line marks the transition from Sobol warm-start to BO.
    """
    if len(convergence_log) == 0:
        logger.warning("No convergence data available to plot.")
        return

    df_conv = pd.DataFrame(convergence_log)

    # Plot 1: Ax objective. Higher is better.
    plt.figure(figsize=(10, 8))
    plt.plot(
        df_conv["trial_number"],
        df_conv["objective_total_negative_error"],
        marker="o",
        linestyle="-",
        label="Observed objective",
    )
    plt.plot(
        df_conv["trial_number"],
        df_conv["best_so_far_total_negative_error"],
        linewidth=2,
        label="Best objective so far",
    )
    plt.axvline(x=N_TRAIN_SOBOL, linestyle="--", label="Start of BO")
    plt.xlabel("Trial number", fontsize=20)
    plt.ylabel("Objective: total_negative_error", fontsize=20)
    plt.tick_params(labelsize=20)
    # plt.title("Bayesian Optimization Convergence: Objective")
    plt.legend(fontsize=18)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(cali_output_folder_path / "bo_convergence_total_negative_error.png", dpi=300)
    plt.close()

    # Plot 2: original Energy Score. Lower is better.
    plt.figure(figsize=(10, 8))
    plt.plot(
        df_conv["trial_number"],
        df_conv["energy_score"],
        marker="o",
        linestyle="-",
        label="Observed Energy Score",
    )
    plt.plot(
        df_conv["trial_number"],
        df_conv["best_so_far_energy_score"],
        linewidth=2,
        label="Best Energy Score so far",
    )
    plt.axvline(x=N_TRAIN_SOBOL, linestyle="--", label="Start of BO")
    plt.xlabel("Trial number", fontsize=20)
    plt.ylabel("Energy Score", fontsize=20)
    plt.tick_params(labelsize=20)
    # plt.title("Bayesian Optimization Convergence: Energy Score")
    plt.legend(fontsize=18)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(cali_output_folder_path / "bo_convergence_energy_score.png", dpi=600)
    plt.close()

    logger.info("Saved BO convergence plots.")


def summarize_recent_bo_improvement(window=20):
    """
    Prints the improvement in the incumbent objective over the last `window` BO trials.
    A small value suggests that BO is becoming flat / converged.
    """
    if len(convergence_log) == 0:
        return

    df_conv = pd.DataFrame(convergence_log)
    bo_df = df_conv[df_conv["trial_number"] > N_TRAIN_SOBOL].copy()

    if len(bo_df) >= window:
        recent_best_change = (
            bo_df["best_so_far_total_negative_error"].iloc[-1]
            - bo_df["best_so_far_total_negative_error"].iloc[-window]
        )
        print(
            f"{Fore.YELLOW}"
            f"Best objective improvement over last {window} BO trials: "
            f"{recent_best_change:.6f}"
            f"{Style.RESET_ALL}"
        )
        logger.info(
            f"Best objective improvement over last {window} BO trials: "
            f"{recent_best_change:.6f}"
        )
# ----------------------------------------------------------------------------------------------------------------------
def save_live_progress(ax_client):
    """
    Saves everything needed to inspect and resume the BO run.

    Important:
    This should be called only after ax_client.complete_trial(...),
    not immediately after get_next_trial().
    """

    # 1. Save full AxClient state for true BO resume
    ax_client.save_to_json_file(filepath=str(ax_client_checkpoint_path))

    # 2. Save Ax trial dataframe for easy inspection
    df_intermediate = exp_to_df(ax_client.experiment).sort_values(by=["trial_index"])
    df_intermediate.to_csv(optimization_intermediate_path, index=False)

    # 3. Save manually tracked objective history (with raw component columns)
    if len(all_inputs) > 0:
        _save_result_intermediate()

    if sobol_stds is not None and not sobol_stds_path.exists():
        with open(sobol_stds_path, "w") as fp:
            json.dump(sobol_stds, fp, indent=2)

    logger.info(f"Saved live AxClient checkpoint to {ax_client_checkpoint_path}")

# ----------------------------------------------------------------------------------------------------------------------
# AxClient experiment with Sobol warm-start, then BO
# ----------------------------------------------------------------------------------------------------------------------
if ax_client_checkpoint_path.exists():
    print(
        f"{Fore.YELLOW}"
        f"Found existing AxClient checkpoint. Resuming from: {ax_client_checkpoint_path}"
        f"{Style.RESET_ALL}"
    )
    logger.info(f"Resuming AxClient from checkpoint: {ax_client_checkpoint_path}")

    ax_client = AxClient.load_from_json_file(
        filepath=str(ax_client_checkpoint_path)
    )

else:
    print(
        f"{Fore.GREEN}"
        f"No AxClient checkpoint found. Creating a new Ax experiment."
        f"{Style.RESET_ALL}"
    )
    logger.info("No AxClient checkpoint found. Creating a new Ax experiment.")

    gs = GenerationStrategy(
        steps=[
            GenerationStep(
                generator=Generators.SOBOL,
                num_trials=N_TRAIN_SOBOL,
                min_trials_observed=N_TRAIN_SOBOL,
                enforce_num_trials=True,
                max_parallelism=1,
            ),
            GenerationStep(
                generator=Generators.BOTORCH_MODULAR,
                num_trials=-1,
                max_parallelism=1,
            ),
        ]
    )

    ax_client = AxClient(generation_strategy=gs, verbose_logging=True)

    ax_client.create_experiment(
        name="single_obj_exp",
        parameters=ax_parameters,
        objectives={
            "total_negative_error": ObjectiveProperties(minimize=False),
        },
    )

# Clean up stale RUNNING trials after resume.
# This is useful if the HPC job was killed after get_next_trial()
# but before complete_trial()
for trial_index, trial in ax_client.experiment.trials.items():
    if trial.status.is_running:
        logger.warning(
            f"Found RUNNING trial {trial_index} on resume. "
            f"Marking it failed so optimization can continue."
        )
        ax_client.log_trial_failure(trial_index=trial_index)

ax_client.save_to_json_file(filepath=str(ax_client_checkpoint_path))

logger.info(f"Generation Strategy: {ax_client.generation_strategy}")

# ----------------------------------------------------------------------------------------------------------------------
# Reload manual bookkeeping if resuming
# ----------------------------------------------------------------------------------------------------------------------
if result_intermediate_path.exists():
    df_prev = pd.read_csv(result_intermediate_path)

    all_inputs = df_prev[param_names].values.tolist()
    all_objectives = df_prev["total_negative_error"].values.tolist()

    objective_cache = {
        tuple(row[param_names].astype(float).tolist()): float(row["total_negative_error"])
        for _, row in df_prev.iterrows()
    }

    # Reload per-component raw metrics cache
    raw_cols = ["weekly_ES_neg", "pooled_tt_ES_neg", "ODT_std_err_neg"]
    if all(c in df_prev.columns for c in raw_cols):
        for _, row in df_prev.iterrows():
            key = tuple(row[param_names].astype(float).tolist())
            raw_metrics_cache[key] = (
                float(row["weekly_ES_neg"]),
                float(row["pooled_tt_ES_neg"]),
                float(row["ODT_std_err_neg"]),
            )

    print(
        f"{Fore.YELLOW}"
        f"Reloaded {len(all_objectives)} previous objective evaluations "
        f"from {result_intermediate_path.name}."
        f"{Style.RESET_ALL}"
    )
    logger.info(
        f"Reloaded {len(all_objectives)} previous objective evaluations "
        f"from {result_intermediate_path}."
    )

if convergence_log_path.exists():
    df_conv_prev = pd.read_csv(convergence_log_path)

    convergence_log = df_conv_prev.to_dict("records")

    if len(df_conv_prev) > 0:
        best_so_far = float(df_conv_prev["best_so_far_total_negative_error"].iloc[-1])
        no_improve_count = int(df_conv_prev["no_improve_count"].iloc[-1])

    print(
        f"{Fore.YELLOW}"
        f"Reloaded convergence log with {len(convergence_log)} rows."
        f"{Style.RESET_ALL}"
    )
    logger.info(f"Reloaded convergence log with {len(convergence_log)} rows.")

# Reload Sobol-phase normalisation stds (present only after Sobol phase is complete)
if sobol_stds_path.exists():
    with open(sobol_stds_path) as fp:
        sobol_stds = json.load(fp)
    print(
        f"{Fore.YELLOW}"
        f"Reloaded Sobol normalisation stds: {sobol_stds}"
        f"{Style.RESET_ALL}"
    )
    logger.info(f"Reloaded Sobol normalisation stds: {sobol_stds}")
elif len(convergence_log) > 0:
    # Mid-Sobol resume: identify Sobol trials by their iteration_type label and look
    # up raw components from raw_metrics_cache (already loaded from result_intermediate.csv).
    # This is more robust than slicing result_intermediate.csv by position, which would
    # break if duplicate cache-hit rows shifted the row order.
    for entry in convergence_log:
        if not str(entry.get("iteration_type", "")).startswith("training_itr_"):
            continue
        key = tuple(float(entry[p]) for p in param_names)
        if key in raw_metrics_cache:
            sobol_raw_metrics.append(raw_metrics_cache[key])
    logger.info(
        f"Reconstructed {len(sobol_raw_metrics)} Sobol raw-metric rows for mid-Sobol resume."
    )

# ----------------------------------------------------------------------------------------------------------------------
# Unified Ax-managed optimization loop with resume support
# ----------------------------------------------------------------------------------------------------------------------
overall_start_time = time.time()

num_completed_trials = sum(
    1 for trial in ax_client.experiment.trials.values()
    if trial.status.is_completed
)

start_trial_number = num_completed_trials + 1

print(
    f"{Fore.YELLOW}"
    f"Completed trials already in Ax: {num_completed_trials}. "
    f"Starting from trial number {start_trial_number}."
    f"{Style.RESET_ALL}"
)
logger.info(
    f"Completed trials already in Ax: {num_completed_trials}. "
    f"Starting from trial number {start_trial_number}."
)

# --------------------------------------------------------------------------------------------------
# Important:
# no_improve_count is updated during Sobol too, but early stopping should only judge BO/SAASBO.
# Therefore, reset no_improve_count once when the BO/SAASBO phase begins.
# --------------------------------------------------------------------------------------------------
bo_patience_reset_done = start_trial_number > N_TRAIN_SOBOL + 1

for i in range(start_trial_number, TOTAL_TRIALS + 1):
    # --- Sobol → BO transition: compute stds and rebuild AxClient on consistent scale ---
    # The flag-file check ensures this runs exactly once; a post-rebuild resume skips it.
    if i > N_TRAIN_SOBOL and not ax_client_rebuilt_flag_path.exists():
        if sobol_stds is None:
            compute_and_save_sobol_stds()
        ax_client = rebuild_ax_client_for_bo()

    parameters, trial_index = ax_client.get_next_trial()

    if i <= N_TRAIN_SOBOL:
        iteration_type = f"training_itr_{i}"
        stage_name = "Sobol warm-start"
    else:
        if not bo_patience_reset_done:
            no_improve_count = 0
            bo_patience_reset_done = True
            logger.info(f"Reset no_improve_count to 0 at the start of BO/SAASBO phase.")
        iteration_type = f"optimization_itr_{i - N_TRAIN_SOBOL}"
        stage_name = "Bayesian optimization"

    logger.info(f"{stage_name} trial {i}/{TOTAL_TRIALS}: {parameters}")

    value = f_total(parameters, iteration_type=iteration_type)

    ax_client.complete_trial(
        trial_index=trial_index,
        raw_data={"total_negative_error": (value, None)},
    )

    update_convergence_log(
        trial_number=i,
        trial_index=trial_index,
        stage_name=stage_name,
        iteration_type=iteration_type,
        parameters=parameters,
        value=value,
    )

    save_live_progress(ax_client)

    if (
        ENABLE_EARLY_STOPPING
        and i > N_TRAIN_SOBOL
        and no_improve_count >= CONVERGENCE_PATIENCE
    ):
        print(
            f"{Fore.YELLOW}"
            f"Early stopping at trial {i}: no meaningful improvement "
            f"for {CONVERGENCE_PATIENCE} BO trials."
            f"{Style.RESET_ALL}"
        )
        logger.info(
            f"Early stopping at trial {i}: no meaningful improvement "
            f"for {CONVERGENCE_PATIENCE} BO trials."
        )
        break

overall_end_time = time.time()
overall_duration = overall_end_time - overall_start_time

# print(f"\n{Fore.GREEN}Total duration is {overall_duration:.2f} seconds{Style.RESET_ALL}")
logger.info(f"Total duration is {overall_duration:.2f} seconds")

# Save final convergence plots and print a compact late-stage BO improvement summary.
plot_convergence_curves()
summarize_recent_bo_improvement(window=20)

# ----------------------------------------------------------------------------------------------------------------------
# Save optimization summary
# ----------------------------------------------------------------------------------------------------------------------
df = exp_to_df(ax_client.experiment).sort_values(by=["trial_index"])
df.to_csv(cali_output_folder_path / "optimization_result.csv", index=False)

best_parameters, best_values = ax_client.get_best_parameters()

# print(f"\n{Fore.RED}Best parameters: {best_parameters}{Style.RESET_ALL}")
# print(f"\n{Fore.RED}Best objective: {best_values}{Style.RESET_ALL}")

logger.info(f"Best parameters: {best_parameters}")
logger.info(f"Best objective: {best_values}")

# ----------------------------------------------------------------------------------------------------------------------
# Save all evaluated points
# ----------------------------------------------------------------------------------------------------------------------
df_inputs = pd.DataFrame(all_inputs, columns=param_names)

df_objectives = pd.DataFrame(
    all_objectives,
    columns=["total_negative_error"]
)
df_objectives["total_error_score"] = -df_objectives["total_negative_error"]

df_total = pd.concat([df_inputs, df_objectives], axis=1)
df_total.to_csv(cali_output_folder_path / "result_old.csv", index=False)
# ----------------------------------------------------------------------------------------------------------------------