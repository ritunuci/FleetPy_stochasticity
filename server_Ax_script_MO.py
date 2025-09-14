# ----------------------------------------------------------------------------------------------------------------------
import numpy as np
import socket
import subprocess
import pickle
import csv
import time  # To add a delay between script runs if necessary
import threading
# import pandas as pd
import os

def run_script():
    subprocess.run(["C:/ProgramData/Anaconda3/envs/fleetpy/python", "run_examples_new.py"])
    time.sleep(3)
    print("\n\n")
def update_csv_with_train_x(values):

    a, b, c, d, e, h = values

    a1 = float(a)           # transit fare ($)
    b1 = float(b)           # Micro distance based fare ($/mile)
    c1 = float(c)           # Micro start fare ($)
    d1 = int(d)             # Fleet size
    e1 = float(e)           # Peak fare factor
    h1 = float(h)           # Micro to fixed factor
    input_csv_path = "D:/Ritun/Siwei_Micro_Transit/Bayesian_Optimization/Input_parameter/input_parameter.csv"
    with open(input_csv_path, mode='r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        rows = list(reader)

    # Update the required values
    rows[1][5] = str(a1)  # "transit_fare ($)" in .csv file
    rows[1][6] = str(b1)  # "microtransit_dist_based_rate ($/mile)" in .csv file
    rows[1][7] = str(c1)  # "microtransit_start_fare ($)" in .csv file
    rows[1][8] = d1       # "Fleet_size" in .csv file

    rows[1][9] = e1  # "PkFareFactor" in .csv file
    rows[1][12] = h1  # "Micro2FixedFactor" in .csv file

    # Write the updated rows back to the CSV file
    with open(input_csv_path, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(rows)
    time.sleep(2)
def collect_output():

    output_csv_path = "D:/Ritun/Siwei_Micro_Transit/Bayesian_Optimization/lemon_grove/output_folder" \
                      "/lemon_grove_evaluation_zonal_partition_False.csv"

    with open(output_csv_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        rows = list(reader)

        # x = rows[3][36]  # "sub_per_T_rider" ($) in .csv file
        x = rows[3][30]  # "tt_sub" ($) in .csv file
        y = rows[3][50]  # "tt_mob_lgsm_inc_with_micro" in .csv file

        # a5 = [float(val) for idx, val in enumerate(rows[3]) if idx not in [0,1,2,3,4,5,6,7,8,9,36,50,72]]

        a5 = [float(val) for idx, val in enumerate(rows[3]) if idx not in [0,1,2,3,4,5,6,7,8,9,30,50,72]]
        value_1 = -float(x)
        value_2 = float(y)

    return [value_1, value_2], a5

def handle_client(conn, addr):
    print('Connected by', addr)
    with conn:
        while True:
            data = conn.recv(4096)  # Increased buffer size for larger data
            if not data:
                break
            train_x = pickle.loads(data)  # Deserialize train_x

            all_outputs = []
            other_metrics = []     #TODO

            for values in train_x:
                # Update CSV with the current value of train_x
                update_csv_with_train_x(values)

                # Run the script
                run_script()
                # time.sleep(5)

                # Collect the output
                print(f"Collecting the output...")
                output, other_metric = collect_output()    #TODO
                all_outputs.append(output)
                other_metrics.append(other_metric)         #TODO

                # Optionally, add a delay to ensure the script has time to finish
                # time.sleep(1)

            all_outputs_array = np.array(all_outputs)
            other_metrics_array = np.array(other_metrics)
            # Serialize and send back the collected output
            serialized_output = pickle.dumps((all_outputs_array, other_metrics_array))   #TODO
            conn.sendall(serialized_output)
            print(f"Results sent back to the client.")

def start_server():
    host = '127.0.0.1'  # Localhost
    port = 52097

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, port))
        s.listen()
        print("Server started, waiting for connections...")
        while True:
            conn, addr = s.accept()
            # Start a new thread to handle the client
            client_thread = threading.Thread(target=handle_client, args=(conn, addr))
            client_thread.start()

if __name__ == "__main__":
    start_server()
