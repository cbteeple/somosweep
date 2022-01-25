import os
import argparse
from pathlib import Path
import somosweep

# Invoke the example using command:
#    python run_simple_sweep.py -dp "data" -s "sweeps/simple_sweep.yaml" -setup
#    python run_simple_sweep.py -dp "data" -s "sweeps/simple_sweep.yaml" -run --force


# Example function where one experiment is run. 
def run_experiment(in_args):
    config_filename = in_args["filename"] # Filename where run config is stored
    config = somosweep.iter_utils.load_yaml(config_filename)
    index = in_args["index"] # Unique index of the run
    replace_existing = in_args.get("replace", True) # decide whether data gets replaced or left alone

    log_filename = os.path.join(os.path.dirname(config_filename), "data.txt")

    print(index)

    # If data exists, cancel the run if we are not explicitly replacing.
    if os.path.exists(log_filename) and not replace_existing:
        return

    # Generate data (fake data for example purposes)
    with open(log_filename,'w') as f:
        for i in range(10):
            f.writelines(["%d,%f\n"%(i, config['finger_arrangement'])])
    




if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Arguments to run an expert trajectory and optionally record data."
    )
    parser.add_argument(
        "-dp",
        "--data_path",
        help="Folder where data should be stored.",
        required=True)
    parser.add_argument(
        "-s",
        "--sweep_name",
        help="Name of the sweep configuration file.",
        required=True,
    )
    parser.add_argument(
        "-setup",
        "--setup_configs",
        help="Generate a batch of config files.",
        action="store_true",
    )
    parser.add_argument(
        "-run",
        "--run_simulations",
        help="Run simulations.",
        action="store_true",
    )
    parser.add_argument(
        "-f",
        "--force",
        help="Force overwrite simulations.",
        action="store_true",
    )
    
    arg = parser.parse_args()

    sweep_file = arg.sweep_name
    sweep_name = os.path.basename(sweep_file)
    sweep_name = Path(sweep_name).with_suffix("")
    data_path = os.path.join(arg.data_path,sweep_name)
    

    #Generate config files
    if arg.setup_configs:
        run_gen = somosweep.RunGenerator(data_path)
        run_gen.from_file(sweep_file)

    # Run the simulation batch
    if arg.run_simulations:
        batchsim = somosweep.BatchSimulation()
        batchsim.load_run_list(data_path, recalculate=arg.force)
        batchsim.run_from_function(
            run_function=run_experiment, parallel=True, num_processes=4
        )