## Imports
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
from pathlib import Path
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
PROJECT_ROOT = Path(__file__).resolve().parent
calibration_study_folder_name = str("cancellation_model_calibration")
fleetpy_input_file_path = PROJECT_ROOT / "studies/example_study/scenarios/example_depot.csv"
fleetpy_demand_folder_path_main = PROJECT_ROOT / "data/demand/example_demand/matched/example_network"
fleetpy_output_folder_path = PROJECT_ROOT / "studies/example_study/results/example_depot_time_pool_irsonly_sc_1"
cali_output_folder_path = PROJECT_ROOT / "calibration_outputs" / calibration_study_folder_name
os.makedirs(cali_output_folder_path, exist_ok=True)
veh_folder = PROJECT_ROOT / "data/fleetctrl"
active_fleetsize_path = PROJECT_ROOT / "data/fleetctrl/elastic_fleet_size/example_active_fleetsize.csv"
real_data_dir = PROJECT_ROOT / "output_for_calibration/real_data"         
# days = ["monday", "tuesday", "wednesday", "thursday", "friday"]
days = ["tuesday"]
city_id = 269       # 269 for Cupertino
num_replications = 2   # per day/date

KPI_COLS = [
    "op_cancellation_rate",
    "wait_time_morning_peak",
    "wait_time_evening_peak",
    "rides_per_veh_rev_hr",
]

# Set up logging
log_file_path = cali_output_folder_path / "run_log.txt"
log_file_path.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file_path, mode="a"),
        logging.StreamHandler()  # keep console output too
    ]
)

logger = logging.getLogger(__name__)


def run_one_fleetpy_rep_and_read_kpis(file_name, day, active_fleetsize_path):
    # runs one FleetPy replication and returns raw KPI vector
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
        subprocess.run(["C:/Users/ritun_uci/.conda/envs/fleetpy_backup/python", "run_examples.py"], check=True)
    time.sleep(1)
    df = pd.read_csv(fleetpy_output_folder_path / "standard_eval.csv", index_col=0, header=0)
    kpi_vec = numpy.array([
        100 - float(df.loc["% created offers"]["MoD_0"]),       # op_cancellation_rate
        float(df.loc["waiting time for morning"]["MoD_0"]),        # wait_time_morning_peak
        float(df.loc["waiting time for evening"]["MoD_0"]),        # wait_time_evening_peak
        float(df.loc["rides per veh rev hours"]["MoD_0"]),        # rides_per_veh_rev_hr
    ], dtype=float)
    return kpi_vec


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
    mu = numpy.nanmean(Y, axis=0)
    sigma = numpy.nanstd(Y, axis=0, ddof=0)
    sigma[sigma < 1e-12] = 1.0  # avoid divide-by-zero
    return mu, sigma


def zscore_standardize(vec, mu, sigma):
    vec = numpy.asarray(vec, dtype=float)
    mu = numpy.asarray(mu, dtype=float).reshape(-1)
    sigma = numpy.asarray(sigma, dtype=float).reshape(-1)
    return (vec - mu) / sigma


def compute_energy_score_file(Z_samples, z_obs):
    """
    Z_samples: array (S, d) standardized simulated KPI vectors for day/date i
    z_obs:     array (d,) standardized observed KPI vector for day/date i
    Implements Eq (7) in the doc
    """
    Z = numpy.asarray(Z_samples, dtype=float)  # (S,d)
    z = numpy.asarray(z_obs, dtype=float)      # (d,)

    # drop any rows with NaNs
    mask = numpy.isfinite(Z).all(axis=1) & numpy.isfinite(z).all()
    if not mask.any():
        raise ValueError("No valid Z_samples for Energy Score")
    Z = Z[mask]
    S = Z.shape[0]
    if S < 2:
        raise ValueError("Energy Score needs at least S>=2 replications")

    # Term1 = (1/S) * sum ||Z_s - z||
    term1 = numpy.mean(numpy.linalg.norm(Z - z[None, :], axis=1))

    # Term2 = (1/(2*S*(S-1))) * sum_{s!=t} ||Z_s - Z_t||
    # vectorized pairwise distances
    diff = Z[:, None, :] - Z[None, :, :]           # (S,S,d)
    D = numpy.linalg.norm(diff, axis=2)               # (S,S)
    term2 = D.sum() / (2.0 * S * (S - 1))          # includes diagonal zeros, ok

    return float(term1 - term2)


def run_program_day(day):
    day_Energy_Score_list = []
    mu, sigma = compute_day_wise_standardized_mu_sigma_from_real(day)
    df_real = pd.read_csv(real_data_dir / f"{day}_real.csv")
    day_real_dict = {}
    for _, row in df_real.iterrows():
        date_key = pd.to_datetime(row["date"]).date()
        day_real_dict[date_key] = row[KPI_COLS].to_numpy(dtype=float)

    fleetpy_demand_folder_path = fleetpy_demand_folder_path_main / day
    fleetpy_input_df = pd.read_csv(fleetpy_input_file_path, index_col=0)
    fleetpy_input_df.loc["PoolingIRSOnly", "day_dir_name"] = str(day)
    fleetpy_input_df.to_csv(fleetpy_input_file_path)
    # with open(fleetpy_input_file_path, mode='r', newline='') as csvfile:
    #     reader = csv.reader(csvfile)
    #     rows = list(reader)
    # rows[1][33] = str(day)
    # with open(fleetpy_input_file_path, mode='w', newline='') as csvfile:
    #     writer = csv.writer(csvfile)
    #     writer.writerows(rows)
    time.sleep(1)

    demand_csv_files = glob.glob(str(fleetpy_demand_folder_path / "*.csv"))
    for file in demand_csv_files:
        file_Z_samples_raw = []
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
        fleetpy_input_file_df = pd.read_csv(fleetpy_input_file_path, index_col=0)
        fleetpy_input_file_df.loc["PoolingIRSOnly", "rq_file"] = str(file_name)
        fleetpy_input_file_df.loc["PoolingIRSOnly", "op_fleet_composition"] = f"default_vehtype:{fleet_size}"
        fleetpy_input_file_df.to_csv(fleetpy_input_file_path)
        # with open(fleetpy_input_file_path, mode='r', newline='') as csvfile:
        #     reader = csv.reader(csvfile)
        #     rows = list(reader)
        # rows[1][2] = str(file_name)
        # rows[1][3] = f"default_vehtype:{fleet_size}"
        # with open(fleetpy_input_file_path, mode='w', newline='') as csvfile:
        #     writer = csv.writer(csvfile)
        #     writer.writerows(rows)
        print(f"{Fore.GREEN}Running Fleetpy for: {str(file_name)} file of {day} {Style.RESET_ALL}")
        time.sleep(1)
        for rep in range(num_replications):
            print(f"{Fore.GREEN}Replication {rep+1} of {num_replications} for {str(file_name)} file of {day} "
                    f"{Style.RESET_ALL}")
            kpi_vec = run_one_fleetpy_rep_and_read_kpis(file_name, day, active_fleetsize_path)
            file_Z_samples_raw.append(kpi_vec)
        file_Z_samples_raw = numpy.vstack(file_Z_samples_raw)       # From the simulated replications
        file_z_real_raw = day_real_dict[date]
        # Standardize both simulated and observed KPI vectors
        file_Z_samples_standardized = zscore_standardize(file_Z_samples_raw, mu, sigma)
        file_z_real_standardized = zscore_standardize(file_z_real_raw, mu, sigma)
        Energy_Score_file = compute_energy_score_file(file_Z_samples_standardized, file_z_real_standardized)
        day_Energy_Score_list.append(Energy_Score_file)
        # weekly_energy_score_list.append(day_Energy_Score_list)
    return day_Energy_Score_list

        
def update_csv_with_train_x(values):
    a, b, c, d = values
    no_show_wait_time = float(a)
    max_rel_error_black_box_PUDO_dur = float(b)
    tt_mul_scale = float(c)
    tt_mul_shape = float(d)
    op_tt_buffers = f"{tt_mul_scale};{tt_mul_shape}"

    input_csv_path = fleetpy_input_file_path
    input_csv_df = pd.read_csv(input_csv_path, index_col=0)
    input_csv_df.loc["PoolingIRSOnly", "no_show_wait_time"] = no_show_wait_time
    input_csv_df.loc["PoolingIRSOnly", "max_rel_error_black_box_PUDO_duration"] = max_rel_error_black_box_PUDO_dur
    input_csv_df.loc["PoolingIRSOnly", "operator_tt_buffers"] = op_tt_buffers
    input_csv_df.to_csv(input_csv_path)
    time.sleep(1)
    # with open(input_csv_path, mode='r', newline='') as csvfile:
    #     reader = csv.reader(csvfile)
    #     rows = list(reader)

    # # Update the required values
    # rows[1][25] = no_show_wait_time
    # rows[1][22] = max_rel_error_black_box_PUDO_dur
    # rows[1][14] = op_tt_buffers

    # # Write the updated rows back to the CSV file
    # with open(input_csv_path, mode='w', newline='') as csvfile:
    #     writer = csv.writer(csvfile)
    #     writer.writerows(rows)
    # time.sleep(1)


def collect_output():
    weekly_energy_score_list = []
    for day in days:
        day_Energy_Score_list = run_program_day(day)
        day_average_Energy_Score = numpy.mean(day_Energy_Score_list)
        weekly_energy_score_list.append(day_average_Energy_Score)
    return -float(numpy.mean(weekly_energy_score_list))        #TODO: Finalize if we have to take the sum or the average


# Function for Exchanging Input and Output with the Simulation (Fleetpy)
def send_instruction(train_x):
    train_x_array = train_x.numpy()
    all_outputs_list = []
    # other_metrics = []
    for values in train_x_array:
        update_csv_with_train_x(values)
        # Process the network here
        subprocess.run(["C:/Users/ritun_uci/.conda/envs/fleetpy_backup/python", "network_preprocessing.py"])
        print(f"{Fore.GREEN} Network Processed {Style.RESET_ALL}")
        time.sleep(1)

        output = collect_output()
        all_outputs_list.append(output)

    return float(all_outputs_list[0])


# Parameters for Ax experiment
N_BATCH = 3  # Number of iterations in optimization loop   # used 30 before
n_samples = 2  # Initial sobol samples for training             # used 15 before
BATCH_SIZE = 1  # Number of candidate points to be generated   # THIS CODE DOES NOT SUPPORT BATCH_SIZE > 1
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


def evaluate_objectives(x_sorted):
    start_time = time.time()
    input_data = torch.tensor(x_sorted).unsqueeze(0)
    print(f"{Fore.RED}send_instruction is being called for input {input_data} {Style.RESET_ALL}")
    _last_output = send_instruction(input_data)
    end_time = time.time()
    duration = end_time - start_time
    log_path = cali_output_folder_path / "indiv_call_timings_eval.csv"
    file_exists = os.path.isfile(log_path)
    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["evaluation_time_(sec)"])
        writer.writerow([duration])

    return _last_output


param_names = ["no_show_wait_time", "max_rel_error_black_box_PUDO_dur", "tt_mul_scale",
               "tt_mul_shape", ]


# For single objective
def f_total(x) -> float:
    
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

        output = evaluate_objectives(x_sorted)
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

    total = float(objective_cache[input_tuple])
    return total


# Single-objective metric (maximize negative total error)
metric_total = GenericNoisyFunctionMetric(name="total_negative_ES", noise_sd=0.0, f=f_total, lower_is_better=False)

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
best_row = df.loc[df["total_negative_ES"].idxmax()]
best_arm = experiment.arms_by_name[best_row["arm_name"]]
best_params = best_arm.parameters
print("Best parameters:", best_params)
print("Best objective:", best_row["total_negative_ES"])

df.to_csv(cali_output_folder_path / "optimization_result.csv", index=False)
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# Save the results to a CSV file
final_all_inputs = numpy.array(all_inputs)
final_all_objectives = numpy.array(all_objectives)

df_inputs = pd.DataFrame(all_inputs,
                         columns=["no_show_wait_time", "max_rel_error_black_box_PUDO_dur",
                                  "tt_mul_scale", "tt_mul_shape"])
df_objectives = pd.DataFrame(all_objectives, columns=["total_energy_score"])

df_total = pd.concat([df_inputs, df_objectives], axis=1)
df_total.to_csv(cali_output_folder_path / "result_old.csv", index=False)
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

