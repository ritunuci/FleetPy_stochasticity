from colorama import Fore, Style
import pickle
import socket
import time
import csv
import os
import subprocess
import glob
import math
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

# ----------------------------------------------------------------------------------------------------------------------
fleetpy_input_file_path = "D:/Ritun/Currently Developing/FleetPy_sto_tt_can_no_show_CALIBRATION_ver_sto_pudo_final/studies/" \
                          "example_study/scenarios/example_depot.csv"
fleetpy_demand_folder_path_main = "D:/Ritun/Currently Developing/FleetPy_sto_tt_can_no_show_CALIBRATION_ver_sto_pudo_final/data/" \
                                  "demand/example_demand/matched/example_network"
fleetpy_output_folder_path = "D:/Ritun/Currently Developing/FleetPy_sto_tt_can_no_show_CALIBRATION_ver_sto_pudo_final/studies/" \
                             "example_study/results/example_depot_time_pool_irsonly_sc_1"
cali_output_folder_path = "D:/Ritun/Currently Developing/FleetPy_sto_tt_can_no_show_CALIBRATION_ver_sto_pudo_final/"
# fleetpy_demand_folder_path = "D:/Ritun/Currently Developing/FleetPy_sto_tt_can_no_show_CALIBRATION_ver/data/" \
#                              "demand/example_demand/matched/example_network"
# std_eval_file_path = "D:/Ritun/Currently Developing/FleetPy_sto_tt_can_no_show_CALIBRATION_ver/
# studies/example_study/" \
#                      "results/example_depot_time_pool_irsonly_sc_1/standard_eval.csv"
days = ["monday", "tuesday", "wednesday", "thursday", "friday"]



def rms_error(list1, list2):
    if len(list1) != len(list2):
        raise ValueError("Lists must have the same length")
    squared_diffs = [(a - b) ** 2 for a, b in zip(list1, list2)]
    mean_squared_diff = sum(squared_diffs) / len(squared_diffs)
    return math.sqrt(mean_squared_diff)


def run_program():
    subprocess.run(["C:/Users/ritun_uci/.conda/envs/fleetpy_backup/python", "network_preprocessing.py"])
    print(f"{Fore.GREEN} Network Processed {Style.RESET_ALL}")
    time.sleep(1)

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
            file_path = os.path.join(fleetpy_demand_folder_path + file_name)
            with open(fleetpy_input_file_path, mode='r', newline='') as csvfile:
                reader = csv.reader(csvfile)
                rows = list(reader)
            rows[1][2] = str(file_name)
            with open(fleetpy_input_file_path, mode='w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerows(rows)
            print(f"{Fore.GREEN}Running Fleetpy for: {str(file_name)} file of {day} {Style.RESET_ALL}")
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


def update_csv_with_train_x(values):
    a, b, c, d = values
    no_show_wait_time = float(a)
    max_rel_error_black_box_PUDO_dur = float(b)
    op_tt_buffer_mrng_peak = float(c)
    op_tt_buffer_evening_peak = float(d)
    op_tt_buffers = f"{op_tt_buffer_mrng_peak};{op_tt_buffer_evening_peak}"

    # a1 = float(a)           # transit fare ($)
    # b1 = float(b)           # Micro distance based fare ($/mile)
    # c1 = float(c)           # Micro start fare ($)
    # d1 = int(d)             # Fleet size
    # e1 = float(e)           # Peak fare factor
    # h1 = float(h)           # Micro to fixed factor
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


# def collect_output():
#     output_csv_path = "D:/Ritun/Currently Developing/FleetPy_sto_tt_can_no_show_CALIBRATION_ver/studies/example_study/" \
#                       "results/example_depot_time_pool_irsonly_sc_1/aggregate.csv"
#
#     # output_csv_path = "D:/Ritun/Siwei_Micro_Transit/Bayesian_Optimization/lemon_grove/output_folder" \
#     #                   "/downtown_sd_evaluation_zonal_partition_False.csv"
#
#     with open(output_csv_path, newline='') as csvfile:
#         reader = csv.reader(csvfile)
#         rows = list(reader)
#
#         x1 = rows[3][30]  # "tt_sub" ($) in .csv file
#         y1 = rows[3][50]  # "tt_mob_lgsm_inc_with_micro" in .csv file
#         a1 = [float(val) for idx, val in enumerate(rows[3]) if idx not in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 30, 50, 72]]
#         value_11 = -float(x1)
#         value_21 = float(y1)
#
#     return [value_11, value_21], a1


def collect_output():
    # global obj_error_day_rms_wait_time_morning_peak, obj_error_op_cancellation_rate,
    # obj_error_wait_time_evening_peak,obj_error_rides_per_veh_rev_hr
    real_data_dir = "D:/Ritun/Currently Developing/FleetPy_sto_tt_can_no_show_CALIBRATION_ver_sto_pudo_final/" \
                    "output_for_calibration/real_data"
    sim_data_dir = "D:/Ritun/Currently Developing/FleetPy_sto_tt_can_no_show_CALIBRATION_ver_sto_pudo_final/" \
                   "output_for_calibration/sim_data"

    day_rms_error_op_cancellation_rate_list = []
    day_rms_error_day_rms_wait_time_morning_peak_list = []
    day_rms_error_wait_time_evening_peak_list = []
    day_rms_error_rides_per_veh_rev_hr_list = []

    for day in days:
        df_op_cancellation_rate_list_real = pd.read_csv(os.path.join(real_data_dir, f"{day}_real.csv"))
        df_op_cancellation_rate_list_sim = pd.read_csv(os.path.join(sim_data_dir, f"{day}_simulated.csv"))

        day_rms_error_op_cancellation_rate_list.append(rms_error(
            df_op_cancellation_rate_list_real["op_cancellation_rate"],
            df_op_cancellation_rate_list_sim["op_cancellation_rate"]))

        day_rms_error_day_rms_wait_time_morning_peak_list.append(
            rms_error(df_op_cancellation_rate_list_real["wait_time_morning_peak"],
                      df_op_cancellation_rate_list_sim["wait_time_morning_peak"]))

        day_rms_error_wait_time_evening_peak_list.append(
            rms_error(df_op_cancellation_rate_list_real["wait_time_evening_peak"],
                      df_op_cancellation_rate_list_sim["wait_time_evening_peak"]))

        day_rms_error_rides_per_veh_rev_hr_list.append(
            rms_error(df_op_cancellation_rate_list_real["rides_per_veh_rev_hr"],
                      df_op_cancellation_rate_list_sim["rides_per_veh_rev_hr"]))

    obj_error_op_cancellation_rate = -float(numpy.mean(day_rms_error_op_cancellation_rate_list))
    obj_error_day_rms_wait_time_morning_peak = -float(numpy.mean(day_rms_error_day_rms_wait_time_morning_peak_list))
    obj_error_wait_time_evening_peak = -float(numpy.mean(day_rms_error_wait_time_evening_peak_list))
    obj_error_rides_per_veh_rev_hr = -float(numpy.mean(day_rms_error_rides_per_veh_rev_hr_list))

    return [obj_error_op_cancellation_rate, obj_error_day_rms_wait_time_morning_peak, obj_error_wait_time_evening_peak,
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
N_BATCH = 1  # Number of iterations in optimization loop   # used 30 before
BATCH_SIZE = 1  # Number of candidate points to be generated   # used 2 before
n_samples = 1  # Initial sobol samples for training             # used 15 before
# dimension = 3   # Number of input parameters
ref_point = [-20, -800, -800, -3]  # TODO: If -50000 is used then in 'metric_a': 'lower_is_better=False' to be used
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
param_max_rel_error_black_box_PUDO_dur = RangeParameter(name="max_rel_error_black_box_PUDO_dur", lower=5.0, upper=80.0,
                                          parameter_type=ParameterType.FLOAT)
param_op_tt_buffer_mrng_peak = RangeParameter(name="op_tt_buffer_mrng_peak", lower=30.0, upper=60.0,
                                              parameter_type=ParameterType.FLOAT)
param_op_tt_buffer_evening_peak = RangeParameter(name="op_tt_buffer_evening_peak", lower=30.0, upper=60.0,
                                                 parameter_type=ParameterType.FLOAT)

# param_max_rel_error_sto_PUDO = RangeParameter(name="Operator max. rel error", lower=1.0, upper=5.0, parameter_type=ParameterType.FLOAT)
# param_transit_fare = RangeParameter(name="Transit Fare ($)", lower=1.0, upper=5.0, parameter_type=ParameterType.FLOAT)
# param_micro_dist_fare = RangeParameter(name="Micro distance based fare ($/mile)", lower=0.0, upper=5.0,
#                                        parameter_type=ParameterType.FLOAT)
# param_micro_start_fare = RangeParameter(name="Micro start fare ($)", lower=0.0, upper=3.0,
#                                         parameter_type=ParameterType.FLOAT)
# param_fleet_size = RangeParameter(name="Fleet size", lower=2.0, upper=7.0, parameter_type=ParameterType.INT)
# param_peak_fare_factor = RangeParameter(name="Peak fare factor", lower=1.0, upper=4.0,
#                                         parameter_type=ParameterType.FLOAT)
# param_micro_fixed_factor = RangeParameter(name="Micro to fixed factor", lower=0.0, upper=1.0,
#                                           parameter_type=ParameterType.FLOAT)

# Define the constraint
# constraint = ParameterConstraint(
#     constraint_dict={
#         "Micro distance based fare ($/mile)": -1.0,
#         "Micro start fare ($)": -1.0
#     },
#     bound=-0.5
# )

# constraint = SumConstraint(parameters=[param_micro_dist_fare, param_micro_start_fare], is_upper_bound=False, bound=0.5)

# search_space = SearchSpace(
#     parameters=[param_transit_fare, param_micro_dist_fare, param_micro_start_fare, param_fleet_size,
#                 param_peak_fare_factor, param_micro_fixed_factor], parameter_constraints=[constraint]
#
# )

search_space = SearchSpace(
    parameters=[param_no_show_wait_time, param_max_rel_error_black_box_PUDO_dur, param_op_tt_buffer_mrng_peak,
                param_op_tt_buffer_evening_peak]
)

# ----------------------------------------------------------------------------------------------------------------------
# search_space = SearchSpace(
#     parameters=[
#         ChoiceParameter(name="Transit Fare ($)", values=[i * 0.10 for i in range(10, 51)],
#                         parameter_type=ParameterType.FLOAT, is_ordered=True, sort_values=True),
#         ChoiceParameter(name="Micro distance based fare ($/mile)", values=[i * 0.10 for i in range(51)],
#                         parameter_type=ParameterType.FLOAT, is_ordered=True, sort_values=True),
#         ChoiceParameter(name="Micro start fare ($)", values=[i * 0.10 for i in range(5, 31)],
#                         parameter_type=ParameterType.FLOAT, is_ordered=True, sort_values=True),
#         RangeParameter(name="Fleet size", lower=2.0, upper=7.0, parameter_type=ParameterType.INT),
#         RangeParameter(name="Peak fare factor", lower=1.0, upper=4.0, parameter_type=ParameterType.FLOAT),
#         RangeParameter(name="Micro to fixed factor", lower=0.0, upper=1.0, parameter_type=ParameterType.FLOAT),
#     ],
#
# )
# ----------------------------------------------------------------------------------------------------------------------
# Initialize lists to collect results
all_inputs = []
all_objectives = []
all_other_metrics = []

final_non_dominated_inputs = []
final_non_dominated_objectives = []
final_non_dominated_metrics = []

num_sobol_samples = n_samples
sobol_counter_f1 = 1
sobol_counter_f2 = 1
sobol_counter_f3 = 1
sobol_counter_f4 = 1
objective_cache = {}


def evaluate_objectives(x_sorted):
    start_time = time.time()
    input_data = torch.tensor(x_sorted).unsqueeze(0)
    print(f"{Fore.RED}send_instruction is being called for input {input_data} {Style.RESET_ALL}")
    _last_output, _last_other_metrics = send_instruction(input_data)
    end_time = time.time()
    duration = end_time - start_time
    log_path = "D:/Ritun/Currently Developing/FleetPy_sto_tt_can_no_show_CALIBRATION_ver_sto_pudo_final/calibration_other_results" \
               "/indiv_call_timings_eval.csv"
    file_exists = os.path.isfile(log_path)
    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["evaluation_time_(sec)"])
        writer.writerow([duration])

    return _last_output, _last_other_metrics


param_names = ["no_show_wait_time", "max_rel_error_black_box_PUDO_dur", "op_tt_buffer_mrng_peak", "op_tt_buffer_evening_peak", ]


def f1(x) -> float:
    x_sorted = [x[p_name] for p_name in param_names]
    input_tuple = tuple(x_sorted)
    global sobol_counter_f1
    if input_tuple not in objective_cache:
        if sobol_counter_f1 <= num_sobol_samples:
            print(f"{Fore.GREEN}f1 evaluating for sobol input: {input_tuple}{Style.RESET_ALL}")
        else:
            print(f"{Fore.GREEN}f1 evaluating for candidate input: {input_tuple}{Style.RESET_ALL}")

        # output, other_metrics = evaluate_objectives(x_sorted)
        # objective_cache[input_tuple] = (output, other_metrics)
        output, _ = evaluate_objectives(x_sorted)
        objective_cache[input_tuple] = output

        all_inputs.append(x_sorted)
        all_objectives.append(output)
        # all_other_metrics.append(other_metrics)
        print(f"{Fore.GREEN}f1 objective output:{objective_cache[input_tuple][0]}{Style.RESET_ALL}\n")
    else:
        print(f"{Fore.GREEN}f1 obj. retrieving for input: {input_tuple}{Style.RESET_ALL}")
        print(f"{Fore.GREEN}f1 objective output from cache:{objective_cache[input_tuple][0]}{Style.RESET_ALL}\n")

        all_inputs.append(x_sorted)
        all_objectives.append(objective_cache[input_tuple])
        # all_other_metrics.append(objective_cache[input_tuple][1])

    sobol_counter_f1 += 1
    return float(objective_cache[input_tuple][0])


def f2(x) -> float:
    x_sorted = [x[p_name] for p_name in param_names]
    input_tuple = tuple(x_sorted)
    global sobol_counter_f2
    if sobol_counter_f2 <= num_sobol_samples:
        print(f"{Fore.BLUE}f2 evaluating for sobol input: {input_tuple}{Style.RESET_ALL}")
    else:
        print(f"{Fore.BLUE}f2 evaluating for candidate input: {input_tuple}{Style.RESET_ALL}")
    sobol_counter_f2 += 1
    print(f"{Fore.BLUE}f2 objective output: {objective_cache[input_tuple][1]}{Style.RESET_ALL}\n")
    return float(objective_cache[input_tuple][1])


def f3(x) -> float:
    x_sorted = [x[p_name] for p_name in param_names]
    input_tuple = tuple(x_sorted)
    global sobol_counter_f3
    if sobol_counter_f3 <= num_sobol_samples:
        print(f"{Fore.BLUE}f3 evaluating for sobol input: {input_tuple}{Style.RESET_ALL}")
    else:
        print(f"{Fore.BLUE}f3 evaluating for candidate input: {input_tuple}{Style.RESET_ALL}")
    sobol_counter_f3 += 1
    print(f"{Fore.BLUE}f3 objective output: {objective_cache[input_tuple][2]}{Style.RESET_ALL}\n")
    return float(objective_cache[input_tuple][2])


def f4(x) -> float:
    x_sorted = [x[p_name] for p_name in param_names]
    input_tuple = tuple(x_sorted)
    global sobol_counter_f4
    if sobol_counter_f4 <= num_sobol_samples:
        print(f"{Fore.BLUE}f4 evaluating for sobol input: {input_tuple}{Style.RESET_ALL}")
    else:
        print(f"{Fore.BLUE}f4 evaluating for candidate input: {input_tuple}{Style.RESET_ALL}")
    sobol_counter_f4 += 1
    print(f"{Fore.BLUE}f4 objective output: {objective_cache[input_tuple][3]}{Style.RESET_ALL}\n")
    return float(objective_cache[input_tuple][3])


metric_a = GenericNoisyFunctionMetric(name="op_cancellation_rate_error", noise_sd=0.0, f=f1, lower_is_better=False)
metric_b = GenericNoisyFunctionMetric(name="wait_time_morning_peak_error", noise_sd=0.0, f=f2, lower_is_better=False)
metric_c = GenericNoisyFunctionMetric(name="wait_time_evening_peak_error", noise_sd=0.0, f=f3, lower_is_better=False)
metric_d = GenericNoisyFunctionMetric(name="rides_per_veh_rev_hrs_error", noise_sd=0.0, f=f4, lower_is_better=False)

# MultiObjective setup
mo = MultiObjective(
    objectives=[Objective(metric=metric_a), Objective(metric=metric_b), Objective(metric=metric_c),
                Objective(metric=metric_d)],
)

objective_thresholds = [
    ObjectiveThreshold(metric=metric_a, bound=ref_point[0], relative=False),
    ObjectiveThreshold(metric=metric_b, bound=ref_point[1], relative=False),
    ObjectiveThreshold(metric=metric_c, bound=ref_point[2], relative=False),
    ObjectiveThreshold(metric=metric_d, bound=ref_point[3], relative=False),
]

optimization_config = MultiObjectiveOptimizationConfig(
    objective=mo,
    objective_thresholds=objective_thresholds,
)


# Build and initialize experiment
def build_experiment():
    experiment = Experiment(
        name="multi_obj_exp",
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

# Define the surrogate model outside the loop
# surrogate = Surrogate(
#     botorch_model_class=SaasFullyBayesianSingleTaskGP,
#     mll_options={
#         "num_samples": num_samples,  # Increasing this may result in better model fits
#         "warmup_steps": warmup_steps,  # Increasing this may result in better model fits
#     },
# )
training_end_time = time.time()
training_duration = training_end_time - training_start_time
print(f"\n{Fore.GREEN}Training duration is {training_duration:.2f} seconds")

optimization_start_time = time.time()
# Run optimization using qNEHVI acquisition function
hv_list = []
for i in range(N_BATCH):
    if i == 0:
        print(f"\n\n{Fore.RED}Sobol Samples End Here{Style.RESET_ALL}")
        print(f"{Fore.RED}Optimization Loop Starts Here{Style.RESET_ALL}\n\n")

    model = Models.BOTORCH_MODULAR(
        experiment=experiment,
        data=data,
    )

    # Generate new candidates
    generator_run = model.gen(BATCH_SIZE)

    # Create and run the batch trial
    trial = experiment.new_batch_trial(generator_run=generator_run)
    trial.run()

    # Fetch data and update
    data = Data.from_multiple_data([data, trial.fetch_data()])

    # Extract input data and objectives from the trial
    exp_df = exp_to_df(experiment)
    outcomes = torch.tensor(exp_df[["op_cancellation_rate_error", "wait_time_morning_peak_error",
                                    "wait_time_evening_peak_error", "rides_per_veh_rev_hrs_error"]].values, **tkwargs)

    # Collect input parameters, objective values, and other metrics
    inputs = generator_run.arms  # Input parameter sets
    objectives = exp_df[["op_cancellation_rate_error", "wait_time_morning_peak_error",
                         "wait_time_evening_peak_error", "rides_per_veh_rev_hrs_error"]].values
    # ------------------------------------------------------------------------------------------------------------------
    # Compute hypervolume
    partitioning = DominatedPartitioning(ref_point=torch.tensor(ref_point, **tkwargs), Y=outcomes)
    try:
        hv = partitioning.compute_hypervolume().item()
    except:
        hv = 0
        print("Failed to compute hypervolume")
    hv_list.append(hv)
    print(f"{Fore.MAGENTA}Iteration {i}, Hypervolume: {hv}{Style.RESET_ALL}")

optimization_end_time = time.time()
optimization_duration = optimization_end_time - optimization_start_time
print(f"\n{Fore.GREEN}Optimization duration is {optimization_duration:.2f} seconds")
# ----------------------------------------------------------------------------------------------------------------------
# Save inputs and objectives as per new method
df = exp_to_df(experiment).sort_values(by=["trial_index"])
df.to_csv("D:/Ritun/Currently Developing/FleetPy_sto_tt_can_no_show_CALIBRATION_ver_sto_pudo_final/calibration_other_results/"
          "optimization_result_new_method.csv", index=False)
# ----------------------------------------------------------------------------------------------------------------------
# Non Dominant Data Collection (Work on this part)
all_objectives_tensor = torch.tensor(all_objectives)
all_inputs_tensor = torch.tensor(all_inputs)
# all_other_metrics_tensor = torch.tensor(all_other_metrics)

sorted_indices = all_objectives_tensor[:, 0].sort()[1]

all_objectives_tensor_sorted = all_objectives_tensor[sorted_indices]
all_inputs_sorted = all_inputs_tensor[sorted_indices]
# all_other_metrics_sorted = all_other_metrics_tensor[sorted_indices]

non_dominated_mask = is_non_dominated(all_objectives_tensor_sorted)

final_non_dominated_input_tensor = all_inputs_sorted[non_dominated_mask]
final_non_dominated_objective_tensor = all_objectives_tensor_sorted[non_dominated_mask]
# final_non_dominated_other_metrics_tensor = all_other_metrics_sorted[non_dominated_mask]

df_non_dom_inputs = pd.DataFrame(final_non_dominated_input_tensor.numpy(),
                                 columns=["no_show_wait_time", "max_rel_error_black_box_PUDO_dur",
                                          "op_tt_buffer_mrng_peak", "op_tt_buffer_evening_peak"])
df_non_dom_objectives = pd.DataFrame(final_non_dominated_objective_tensor.numpy(),
                                     columns=["op_cancellation_rate_error", "wait_time_morning_peak_error",
                                              "wait_time_evening_peak_error", "rides_per_veh_rev_hrs_error"])

# df_non_dom_metrics = pd.DataFrame(final_non_dominated_other_metrics_tensor.numpy(),
#                                   columns=['car_users', 'car_mode_share (%)',
#                                            ])

# df_total_non_dom = pd.concat([df_non_dom_inputs, df_non_dom_objectives, df_non_dom_metrics], axis=1)
df_total_non_dom = pd.concat([df_non_dom_inputs, df_non_dom_objectives], axis=1)
df_total_non_dom.to_csv("D:/Ritun/Currently Developing/FleetPy_sto_tt_can_no_show_CALIBRATION_ver_sto_pudo_final/"
                        "calibration_other_results/"
                        "non_dominated_data_old_method.csv", index=False)
# ----------------------------------------------------------------------------------------------------------------------
# Work on this part
# Save the results to a CSV file
final_all_inputs = numpy.array(all_inputs)
final_all_objectives = numpy.array(all_objectives)
# final_all_other_metrics = numpy.array(all_other_metrics)

df_inputs = pd.DataFrame(all_inputs,
                         columns=["no_show_wait_time", "max_rel_error_black_box_PUDO_dur",
                                  "op_tt_buffer_mrng_peak", "op_tt_buffer_evening_peak"])
df_objectives = pd.DataFrame(all_objectives, columns=["op_cancellation_rate_error", "wait_time_morning_peak_error",
                                                      "wait_time_evening_peak_error", "rides_per_veh_rev_hrs_error"])

# df_other_metrics = pd.DataFrame(all_other_metrics, columns=['car_users', 'car_mode_share (%)',
#                                                             ])

# Concatenate all data and save to output file
# df_total = pd.concat([df_inputs, df_objectives, df_other_metrics], axis=1)
df_total = pd.concat([df_inputs, df_objectives], axis=1)
df_total.to_csv("D:/Ritun/Currently Developing/FleetPy_sto_tt_can_no_show_CALIBRATION_ver_sto_pudo_final/calibration_other_results/"
                "result_old_method.csv", index=False)
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
batch_number = torch.cat(
    [
        torch.zeros(n_samples),
        torch.arange(1, N_BATCH + 1).repeat(BATCH_SIZE, 1).t().reshape(-1),
    ]
).numpy()

non_dominated_batch_numbers = batch_number[sorted_indices][non_dominated_mask]
# HYPERVOLUME PLOTTING
# Plot the observed hypervolumes over iterations
iterations = numpy.arange(1, N_BATCH + 1)
fig3, ax3 = plt.subplots()
ax3.plot(iterations, hv_list, color='blue')
ax3.set_xlabel("Iteration")
ax3.set_ylabel("Hypervolume")
fig3.savefig('D:/Ritun/Currently Developing/FleetPy_sto_tt_can_no_show_CALIBRATION_ver_sto_pudo_final/calibration_other_results/'
             'hypervolume_plot.png', format='png', dpi=300)
plt.show()
print(f"Hypervolumes: {hv_list}")
print(f"Non-dom batch numbers: {non_dominated_batch_numbers}")
# ----------------------------------------------------------------------------------------------------------------------

