from colorama import Fore, Style
import pickle
import socket
import time
import csv
import os
import subprocess
import glob
import math
import logging
from ax import optimize
from ax.core import ParameterType, RangeParameter, SearchSpace, MultiObjective, Objective, ObjectiveThreshold
from ax.core.experiment import Experiment
from ax.core.optimization_config import (
    MultiObjectiveOptimizationConfig,
    ObjectiveThreshold,
)
from ax.runners.synthetic import SyntheticRunner
from ax.metrics.noisy_function import GenericNoisyFunctionMetric
from botorch.utils.sampling import draw_sobol_samples

import numpy
import pandas as pd
import torch
from ax.core.data import Data
from ax.core.experiment import Experiment
from ax.core.metric import Metric
from ax.core.objective import MultiObjective, Objective
from ax.core.optimization_config import (
    MultiObjectiveOptimizationConfig,
    ObjectiveThreshold,
)
from ax.core import ChoiceParameter
from ax.core.parameter import ParameterType, RangeParameter
from ax.core.search_space import SearchSpace
from ax.metrics.noisy_function import GenericNoisyFunctionMetric
from ax.modelbridge.cross_validation import compute_diagnostics, cross_validate

# Analysis utilities, including a method to evaluate hypervolumes
from ax.modelbridge.modelbridge_utils import observed_hypervolume

# Model registry for creating multi-objective optimization models.
from ax.modelbridge.registry import Models
from ax.models.torch.botorch_modular.surrogate import Surrogate
from ax.plot.contour import plot_contour
from ax.plot.diagnostic import tile_cross_validation
from ax.plot.pareto_frontier import plot_pareto_frontier
from ax.plot.pareto_utils import compute_posterior_pareto_frontier
from ax.runners.synthetic import SyntheticRunner
from ax.service.utils.report_utils import exp_to_df

# Plotting imports and initialization
from ax.utils.notebook.plotting import init_notebook_plotting, render
from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP
from botorch.test_functions.multi_objective import DTLZ2
from botorch.utils.multi_objective.box_decompositions.dominated import (
    DominatedPartitioning,
)
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
from botorch.utils.multi_objective.pareto import is_non_dominated
from ax.plot.pareto_frontier import plot_pareto_frontier
from ax.core.parameter_constraint import ParameterConstraint
from ax import SumConstraint
# add this import
from ax.core.optimization_config import OptimizationConfig

# ----------------------------------------------------------------------------------------------------------------------
fleetpy_input_file_path = "D:/Ritun/Currently Developing/FleetPy_sto_tt_can_no_show_CALIBRATION_ver_sto_pudo_final/studies/" \
                          "example_study/scenarios/example_depot.csv"
fleetpy_demand_folder_path_main = "D:/Ritun/Currently Developing/FleetPy_sto_tt_can_no_show_CALIBRATION_ver_sto_pudo_final/data/" \
                                  "demand/example_demand/matched/example_network"
fleetpy_output_folder_path = "D:/Ritun/Currently Developing/FleetPy_sto_tt_can_no_show_CALIBRATION_ver_sto_pudo_final/studies/" \
                             "example_study/results/example_depot_time_pool_irsonly_sc_1"
cali_output_folder_path = "D:/Ritun/Currently Developing/FleetPy_sto_tt_can_no_show_CALIBRATION_ver_sto_pudo_final/"
veh_folder = "D:/Ritun/Currently Developing/FleetPy_sto_tt_can_no_show_CALIBRATION_ver_sto_pudo_final/data/" \
                 "fleetctrl"
active_fleetsize_path = "D:/Ritun/Currently Developing/" \
                                    "FleetPy_sto_tt_can_no_show_CALIBRATION_ver_sto_pudo_final/data/fleetctrl/" \
                                    "elastic_fleet_size/example_active_fleetsize.csv"
# fleetpy_demand_folder_path = "D:/Ritun/Currently Developing/FleetPy_sto_tt_can_no_show_CALIBRATION_ver/data/" \
#                              "demand/example_demand/matched/example_network"
# std_eval_file_path = "D:/Ritun/Currently Developing/FleetPy_sto_tt_can_no_show_CALIBRATION_ver/
# studies/example_study/" \
#                      "results/example_depot_time_pool_irsonly_sc_1/standard_eval.csv"
# days = ["monday", "tuesday", "wednesday", "thursday", "friday"]
days = ["tuesday"]
city_id = 269

log_file_path = "D:/Ritun/Currently Developing/FleetPy_sto_tt_can_no_show_CALIBRATION_ver_sto_pudo_final/" \
                "calibration_other_results/single_obj/run_log.txt"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file_path, mode="a"),
        logging.StreamHandler()  # keep console output too
    ]
)

logger = logging.getLogger(__name__)


# def rms_error(list1, list2):
#     if len(list1) != len(list2):
#         raise ValueError("Lists must have the same length")
#     filtered_pairs = [(a, b) for a, b in zip(list1, list2) if not math.isnan(b)]
#     if not filtered_pairs:
#         raise ValueError("All values in list2 are NaN, cannot compute RMS error")
#     list1_filtered, list2_filtered = zip(*filtered_pairs)
#     squared_diffs = [(a - b) ** 2 for a, b in zip(list1_filtered, list2_filtered)]
#     mean_squared_diff = sum(squared_diffs) / len(squared_diffs)
#     return math.sqrt(mean_squared_diff)

def rms_error(list1, list2):
    a = numpy.asarray(list1, dtype=float)
    b = numpy.asarray(list2, dtype=float)
    mask = numpy.isfinite(a) & numpy.isfinite(b)
    if not mask.any():
        raise ValueError("No valid pairs for RMS error")
    diff = a[mask] - b[mask]
    return float(numpy.sqrt(numpy.mean(diff**2)))


def run_program():
    subprocess.run(["C:/Users/ritun_uci/.conda/envs/fleetpy_backup/python", "network_preprocessing.py"])
    print(f"{Fore.GREEN} Network Processed {Style.RESET_ALL}")
    time.sleep(1)

    # fleet_size_prob_date = []
    # fleet_size_prob_day = []

    for day in days:
        day_op_cancellation_rate_list = []
        day_wait_time_morning_peak_list = []
        day_wait_time_evening_peak_list = []
        day_rides_per_veh_rev_hr_list = []
        date_list = []
        fleetpy_demand_folder_path = os.path.join(fleetpy_demand_folder_path_main, day)
        with open(fleetpy_input_file_path, mode='r', newline='') as csvfile:
            reader = csv.reader(csvfile)
            rows = list(reader)
        rows[1][33] = str(day)
        with open(fleetpy_input_file_path, mode='w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(rows)
        time.sleep(1)

        demand_csv_files = glob.glob(os.path.join(fleetpy_demand_folder_path, "*.csv"))
        for file in demand_csv_files:
            file_name = os.path.splitext(os.path.basename(file))[0] + ".csv"
            file_name_without_ext = os.path.splitext(os.path.basename(file))[0]
            veh_file = os.path.join(veh_folder, f"{city_id}_unique_vehid", day, file_name)
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
            with open(fleetpy_input_file_path, mode='r', newline='') as csvfile:
                reader = csv.reader(csvfile)
                rows = list(reader)
            rows[1][2] = str(file_name)
            rows[1][3] = f"default_vehtype:{fleet_size}"
            with open(fleetpy_input_file_path, mode='w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerows(rows)
            print(f"{Fore.GREEN}Running Fleetpy for: {str(file_name)} file of {day} {Style.RESET_ALL}")
            time.sleep(1)
            try:
                subprocess.run(["C:/Users/ritun_uci/.conda/envs/fleetpy_backup/python", "run_examples.py"], check=True)
            except Exception as e:
                print(f"{Fore.GREEN}FleetPy failed for {str(file_name)} of {day}: {e},\n "
                      f"Retrying with fleet fraction = 0.5 for all times...{Style.RESET_ALL}")
                # fleet_size_prob_date.append(str(file_name))         #TODO:Remove later
                # fleet_size_prob_day.append(day)                     #TODO:Remove later
                df_fix = pd.read_csv(active_fleetsize_path)
                df_fix.iloc[:, 1] = 0.5
                df_fix.to_csv(active_fleetsize_path, index=False)
                time.sleep(1)
                subprocess.run(["C:/Users/ritun_uci/.conda/envs/fleetpy_backup/python", "run_examples.py"])
            time.sleep(1)
            df = pd.read_csv(os.path.join(fleetpy_output_folder_path, "standard_eval.csv"))
            day_op_cancellation_rate_list.append(100 - df["MoD_0"][13])
            day_wait_time_morning_peak_list.append(df["MoD_0"][21])
            day_wait_time_evening_peak_list.append(df["MoD_0"][27])
            day_rides_per_veh_rev_hr_list.append(df["MoD_0"][33])
            date_list.append(file_name_without_ext)
        df_day = pd.DataFrame({"date": date_list, "op_cancellation_rate": day_op_cancellation_rate_list,
                               "wait_time_morning_peak": day_wait_time_morning_peak_list,
                               "wait_time_evening_peak": day_wait_time_evening_peak_list,
                               "rides_per_veh_rev_hr": day_rides_per_veh_rev_hr_list})
        output_file = os.path.join(cali_output_folder_path, "output_for_calibration", "sim_data",
                                   f"{day}_simulated.csv")
        df_day.to_csv(output_file, index=False)
    # fleet_size_prob_df = pd.DataFrame({"Date": fleet_size_prob_date, "Day": fleet_size_prob_day})   #TODO: Remove later
    # fleet_size_prob_df.to_csv("D:/Ritun/Currently Developing/FleetPy_sto_tt_can_no_show_CALIBRATION_ver_sto_pudo_final/"
    #                           "data/fleetctrl/elastic_fleet_size/example_active_fleetsize_problems.csv", index=False)    #TODO: Remove later


def update_csv_with_train_x(values):
    a, b, c, d = values
    no_show_wait_time = float(a)
    max_rel_error_black_box_PUDO_dur = float(b)
    tt_mul_scale = float(c)
    tt_mul_shape = float(d)
    op_tt_buffers = f"{tt_mul_scale};{tt_mul_shape}"

    input_csv_path = fleetpy_input_file_path
    with open(input_csv_path, mode='r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        rows = list(reader)

    # Update the required values
    rows[1][25] = no_show_wait_time
    rows[1][22] = max_rel_error_black_box_PUDO_dur
    rows[1][14] = op_tt_buffers

    # Write the updated rows back to the CSV file
    with open(input_csv_path, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(rows)
    time.sleep(1)


def collect_output():
    real_data_dir = "D:/Ritun/Currently Developing/FleetPy_sto_tt_can_no_show_CALIBRATION_ver_sto_pudo_final/" \
                    "output_for_calibration/real_data"
    sim_data_dir = "D:/Ritun/Currently Developing/FleetPy_sto_tt_can_no_show_CALIBRATION_ver_sto_pudo_final/" \
                   "output_for_calibration/sim_data"

    day_rms_error_op_cancellation_rate_list = []
    day_rms_error_wait_time_morning_peak_list = []
    day_rms_error_wait_time_evening_peak_list = []
    day_rms_error_rides_per_veh_rev_hr_list = []

    for day in days:
        df_op_cancellation_rate_list_real = pd.read_csv(os.path.join(real_data_dir, f"{day}_real.csv"))
        df_op_cancellation_rate_list_sim = pd.read_csv(os.path.join(sim_data_dir, f"{day}_simulated.csv"))

        day_rms_error_op_cancellation_rate_list.append(rms_error(
            df_op_cancellation_rate_list_real["op_cancellation_rate"],
            df_op_cancellation_rate_list_sim["op_cancellation_rate"]))

        day_rms_error_wait_time_morning_peak_list.append(
            rms_error(df_op_cancellation_rate_list_real["wait_time_morning_peak"],
                      df_op_cancellation_rate_list_sim["wait_time_morning_peak"]))

        day_rms_error_wait_time_evening_peak_list.append(
            rms_error(df_op_cancellation_rate_list_real["wait_time_evening_peak"],
                      df_op_cancellation_rate_list_sim["wait_time_evening_peak"]))

        day_rms_error_rides_per_veh_rev_hr_list.append(
            rms_error(df_op_cancellation_rate_list_real["rides_per_veh_rev_hr"],
                      df_op_cancellation_rate_list_sim["rides_per_veh_rev_hr"]))

    obj_error_op_cancellation_rate = -float(numpy.mean(day_rms_error_op_cancellation_rate_list))
    obj_error_wait_time_morning_peak = -float(numpy.mean(day_rms_error_wait_time_morning_peak_list))
    obj_error_wait_time_evening_peak = -float(numpy.mean(day_rms_error_wait_time_evening_peak_list))
    obj_error_rides_per_veh_rev_hr = -float(numpy.mean(day_rms_error_rides_per_veh_rev_hr_list))

    return [obj_error_op_cancellation_rate, obj_error_wait_time_morning_peak, obj_error_wait_time_evening_peak,
            obj_error_rides_per_veh_rev_hr]


# Function for Exchanging Input and Output with the Simulation (Fleetpy)
def send_instruction(train_x):
    train_x_array = train_x.numpy()
    all_outputs = []
    # other_metrics = []
    for values in train_x_array:
        update_csv_with_train_x(values)
        run_program()
        output = collect_output()
        all_outputs.append(output)
        # other_metrics.append(other_metric)
    all_outputs_array = numpy.array(all_outputs)
    # other_metrics_array = numpy.array(other_metrics)

    received_output_list = all_outputs_array.tolist()[0]
    received_other_metrics_list = []

    return received_output_list, received_other_metrics_list


# Parameters for Ax experiment
N_BATCH = 3  # Number of iterations in optimization loop   # used 30 before
BATCH_SIZE = 1  # Number of candidate points to be generated   # used 2 before
n_samples = 2  # Initial sobol samples for training             # used 15 before
# dimension = 3   # Number of input parameters
tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}

num_samples = 256
warmup_steps = 512
# ----------------------------------------------------------------------------------------------------------------------
# TODO: Search Space Definition for Discrete variables (i.e., the variables can take only certain values) Ax also
#  supports categorical variables

param_no_show_wait_time = RangeParameter(name="no_show_wait_time", lower=30.0, upper=120.0,
                                         parameter_type=ParameterType.FLOAT)
param_max_rel_error_black_box_PUDO_dur = RangeParameter(name="max_rel_error_black_box_PUDO_dur", lower=5.0, upper=90.0,
                                          parameter_type=ParameterType.FLOAT)             ## In %
# param_op_tt_buffer_mrng_peak = RangeParameter(name="op_tt_buffer_mrng_peak", lower=0.5, upper=1.5,
#                                               parameter_type=ParameterType.FLOAT)
# param_op_tt_buffer_evening_peak = RangeParameter(name="op_tt_buffer_evening_peak", lower=0.5, upper=1.5,
#                                                  parameter_type=ParameterType.FLOAT)

tt_multiplier_for_scale = RangeParameter(name="tt_mul_scale", lower=1.0, upper=1.01,
                                              parameter_type=ParameterType.FLOAT)
tt_multiplier_for_shape = RangeParameter(name="tt_mul_shape", lower=1.0, upper=1.01,
                                                 parameter_type=ParameterType.FLOAT)

search_space = SearchSpace(
    parameters=[param_no_show_wait_time, param_max_rel_error_black_box_PUDO_dur, tt_multiplier_for_scale,
                tt_multiplier_for_shape]
)
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# Initialize lists to collect results
all_inputs = []
all_objectives = []
all_other_metrics = []

num_sobol_samples = n_samples
sobol_counter_f_total = 1  # track initial Sobol vs candidate logging
objective_cache = {}

# # Normalization/scoring config
# WEIGHTS = numpy.array([1.0, 1.0, 1.0, 1.0])   # tune if needed
# SCALES = numpy.array([10.0, 60.0, 60.0, 1.0])  #TODO: We have to discuss these values
# OFFSETS = numpy.array([0.0, 0.0, 0.0, 0.0])   # usually 0 for errors
# EPS = 1e-9


def evaluate_objectives(x_sorted):
    start_time = time.time()
    input_data = torch.tensor(x_sorted).unsqueeze(0)
    print(f"{Fore.RED}send_instruction is being called for input {input_data} {Style.RESET_ALL}")
    _last_output, _last_other_metrics = send_instruction(input_data)
    end_time = time.time()
    duration = end_time - start_time
    log_path = "D:/Ritun/Currently Developing/FleetPy_sto_tt_can_no_show_CALIBRATION_ver_sto_pudo_final/" \
               "calibration_other_results/single_obj" \
               "/indiv_call_timings_eval.csv"
    file_exists = os.path.isfile(log_path)
    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["evaluation_time_(sec)"])
        writer.writerow([duration])

    return _last_output, _last_other_metrics


param_names = ["no_show_wait_time", "max_rel_error_black_box_PUDO_dur", "tt_mul_scale",
               "tt_mul_shape", ]


# For single objective
def f_total(x) -> float:
    """
    Returns the SUM of the four (negative) error components.
    We maximize this (lower_is_better=False), which minimizes the total RMS error.
    """
    x_sorted = [x[p_name] for p_name in param_names]
    input_tuple = tuple(x_sorted)
    global sobol_counter_f_total

    if input_tuple not in objective_cache:
        if sobol_counter_f_total <= num_sobol_samples:
            print(f"{Fore.GREEN}f_total evaluating for Sobol input: {input_tuple}{Style.RESET_ALL}")
            logger.info(f"f_total evaluating for Sobol input: {input_tuple}")
        else:
            print(f"{Fore.GREEN}f_total evaluating for candidate input: {input_tuple}{Style.RESET_ALL}")
            logger.info(f"f_total evaluating for candidate input: {input_tuple}")

        # This runs FleetPy once and returns the 4 metrics (still negative RMS errors)
        output, _ = evaluate_objectives(x_sorted)
        objective_cache[input_tuple] = output

        # keep your bookkeeping
        all_inputs.append(x_sorted)
        all_objectives.append(output)

        print(f"{Fore.GREEN}f_total components: {objective_cache[input_tuple]}{Style.RESET_ALL}\n")
        logger.info(f"f_total components: {objective_cache[input_tuple]}")
    else:
        print(f"{Fore.GREEN}f_total retrieving from cache: {input_tuple}{Style.RESET_ALL}")
        logger.info(f"f_total retrieving from cache: {input_tuple}")
        print(f"{Fore.GREEN}f_total components (cached): {objective_cache[input_tuple]}{Style.RESET_ALL}\n")
        logger.info(f"f_total components (cached): {objective_cache[input_tuple]}")

        all_inputs.append(x_sorted)
        all_objectives.append(objective_cache[input_tuple])

    sobol_counter_f_total += 1

    # sum the four negative errors -> still negative; maximizing this minimizes total RMS error
    total = float(sum(objective_cache[input_tuple]))
    return total

    # raw = numpy.array(objective_cache[input_tuple], dtype=float)  # negative errors
    # pos = -raw  # convert to positive errors
    # norm = (pos - OFFSETS) / (SCALES + EPS)  # normalize to unitless
    # score = float(numpy.sum(WEIGHTS * norm))  # weighted sum (lower is better)
    #
    # return -score


# Single-objective metric (maximize negative total error)
metric_total = GenericNoisyFunctionMetric(name="total_error_sum", noise_sd=0.0, f=f_total, lower_is_better=False)

optimization_config = OptimizationConfig(objective=Objective(metric=metric_total))  # maximize)


def build_experiment():
    experiment = Experiment(
        name="single_obj_exp",
        search_space=search_space,
        optimization_config=optimization_config,
        runner=SyntheticRunner(),
    )
    return experiment


training_start_time = time.time()
experiment = build_experiment()


def initialize_experiment(experiment):
    sobol = Models.SOBOL(search_space=experiment.search_space)
    experiment.new_batch_trial(sobol.gen(n_samples)).run()
    return experiment.fetch_data()


data = initialize_experiment(experiment)


# def calibrate_scalers_from_sobol(method="zscore"):
#     global SCALES, OFFSETS
#     sobol_components = numpy.array(all_objectives[:num_sobol_samples], dtype=float)  # negative errors
#     pos = -sobol_components  # positive errors
#
#     if method == "zscore":
#         OFFSETS = pos.mean(axis=0)
#         SCALES = pos.std(axis=0)
#         SCALES[SCALES < 1e-9] = 1.0  # fallback to avoid zeros
#     elif method == "minmax":
#         lo = numpy.percentile(pos, 5, axis=0)
#         hi = numpy.percentile(pos, 95, axis=0)
#         OFFSETS = lo
#         SCALES = hi - lo
#         SCALES[SCALES < 1e-9] = 1.0
#     else:
#         raise ValueError("method must be 'zscore' or 'minmax'")
#
#
# calibrate_scalers_from_sobol(method="minmax")
# print(f"\n{Fore.RED}Normalization OFFSETS: {OFFSETS}{Style.RESET_ALL}")
# print(f"\n{Fore.RED}Normalization SCALES : {SCALES}{Style.RESET_ALL}")


training_end_time = time.time()
training_duration = training_end_time - training_start_time
print(f"\n{Fore.GREEN}Training duration is {training_duration:.2f} seconds")
logger.info(f"Training duration is {training_duration:.2f} seconds")
optimization_start_time = time.time()

for i in range(N_BATCH):
    if i == 0:
        print(f"\n\n{Fore.RED}Sobol Samples End Here{Style.RESET_ALL}")
        logger.info(f"Sobol Samples End Here{Style.RESET_ALL}")
        print(f"{Fore.RED}Optimization Loop Starts Here{Style.RESET_ALL}\n\n")
        logger.info(f"Optimization Loop Starts Here{Style.RESET_ALL}")

    model = Models.BOTORCH_MODULAR(experiment=experiment, data=data)
    generator_run = model.gen(BATCH_SIZE)
    trial = experiment.new_batch_trial(generator_run=generator_run)
    trial.run()
    data = Data.from_multiple_data([data, trial.fetch_data()])

optimization_end_time = time.time()
optimization_duration = optimization_end_time - optimization_start_time
print(f"\n{Fore.GREEN}Optimization duration is {optimization_duration:.2f} seconds")
logger.info(f"Optimization duration is {optimization_duration:.2f} seconds")
# ----------------------------------------------------------------------------------------------------------------------
# Save inputs and objectives as per new method
df = exp_to_df(experiment).sort_values(by=["trial_index"])
best_row = df.loc[df["total_error_sum"].idxmax()]
best_arm = experiment.arms_by_name[best_row["arm_name"]]
best_params = best_arm.parameters
print("Best parameters:", best_params)
print("Best objective:", best_row["total_error_sum"])

df.to_csv("D:/Ritun/Currently Developing/FleetPy_sto_tt_can_no_show_CALIBRATION_ver_sto_pudo_final/"
          "calibration_other_results/single_obj/"
          "optimization_result_new_method.csv", index=False)
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# Save the results to a CSV file
final_all_inputs = numpy.array(all_inputs)
final_all_objectives = numpy.array(all_objectives)
# final_all_other_metrics = numpy.array(all_other_metrics)

df_inputs = pd.DataFrame(all_inputs,
                         columns=["no_show_wait_time", "max_rel_error_black_box_PUDO_dur",
                                  "tt_mul_scale", "tt_mul_shape"])
df_objectives = pd.DataFrame(all_objectives, columns=["op_cancellation_rate_error", "wait_time_morning_peak_error",
                                                      "wait_time_evening_peak_error", "rides_per_veh_rev_hrs_error"])

df_total = pd.concat([df_inputs, df_objectives], axis=1)
df_total.to_csv("D:/Ritun/Currently Developing/FleetPy_sto_tt_can_no_show_CALIBRATION_ver_sto_pudo_final/"
                "calibration_other_results/single_obj/"
                "result_old.csv", index=False)
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

