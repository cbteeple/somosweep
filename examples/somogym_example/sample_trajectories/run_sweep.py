# Invoke the example using command (first navigate to "sample_trajectories"):
#
#  * Clean the data  (if nessecary):
#      python run_sweep.py -env "SoMoGymExampleEnv" -t "SoMoGymExampleEnv-test" -s "sweeps/arrangement_sweep.yaml" -clean
#
#  * Generate config files and run todo list:
#      python run_sweep.py -env "SoMoGymExampleEnv" -t "SoMoGymExampleEnv-test" -s "sweeps/arrangement_sweep.yaml" -setup
#
#  * Run the sweep (skipping runs that have already generated data):
#     python run_sweep.py -env "SoMoGymExampleEnv" -t "SoMoGymExampleEnv-test" -s "sweeps/arrangement_sweep.yaml" -run
#
#  * Run the sweep (skipping runs that have already generated data) with the visualizer:
#     python run_sweep.py -env "SoMoGymExampleEnv" -t "SoMoGymExampleEnv-test" -s "sweeps/arrangement_sweep.yaml" -run -v
#
#  * Run the sweep (force overwrite of existing data):
#     python run_sweep.py -env "SoMoGymExampleEnv" -t "SoMoGymExampleEnv-test" -s "sweeps/arrangement_sweep.yaml" -run --force
#


from math import ceil
import os
import sys
import argparse
import importlib
import shutil
from pathlib import Path
import copy
import gym
import somosweep
import pybullet as p
import sorotraj
import csv
import numpy as np

path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, path)


# Define the run function
def run_expert_trajectory(
    environment_name,
    run_config,
    run_render=False,
    debug=False,
    record_video=True,
):

    # Import the environment
    try:
        importlib.import_module('environments.'+environment_name)
    except:
        print(f"CRITICAL ERROR: Invalid environment '{environment_name}' selected.")
        sys.exit(1)

    curr_id = run_config["env_id"].split("-")[0] + "-v%d"%(run_config['run_index'])

    env = gym.make(
        run_config["env_id"],
        run_config=run_config,
        run_ID=run_config['save']['run_name'],
        render=run_render,
        debug=debug,
    )
    
    # todo: only seed when required, make seed selectable; this should use the seed from the run_config
    env.seed(0)

    expert_rel_path = run_config['expert_rel_path']
    expert_data_dir = expert_rel_path / "data"
    expert_traj_def_file = expert_rel_path / run_config['trajectory']
    
    vid_path = Path(run_config['log_path']) / "vid.mp4"

    obs = env.reset(run_render=run_render)

    traj_build = sorotraj.TrajBuilder()
    traj_build.load_traj_def(str(expert_traj_def_file))
    trajectory = traj_build.get_trajectory()
    interp = sorotraj.Interpolator(trajectory)
    action_len = env.action_space.shape[0]

    traj_actuation_fn, final_time = interp.get_traj_function(
            num_reps=1,  # change num repos here
            speed_factor=1.0,
            invert_direction=True,  # todo: this is hacky - change expert traj entries and set invert_direction=False
        )

    if "max_episode_steps" not in run_config:
        run_config["max_episode_steps"] = ceil(float(final_time)/(run_config["action_time"]))
    num_steps = run_config["max_episode_steps"]


    data_labels = env.get_observation_labels()

    data=np.array([[None]*len(data_labels)]*num_steps)

    # Start saving a video
    if run_render and record_video:
        if vid_path.exists():
            os.remove(vid_path)
        vid_filename = os.path.abspath(vid_path)
        logIDvideo = p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, vid_filename)

    try:
        # Run the actual simulation
        for i in range(num_steps):
            applied_action = (
                np.array(traj_actuation_fn(env.step_count * env.action_time))
                / env.torque_multiplier
            )
            restricted_action = np.minimum(
                np.maximum(applied_action, np.array([-1.0] * action_len)),
                np.array([1.0] * action_len),
            )
            _obs, _rewards, _dones, info = env.step(restricted_action)

            data[i,:] = _obs

            if run_render:
                env.render()

    except KeyboardInterrupt:
        pass

    # Shut everything down and return the data
    env.close()
    if run_render and record_video:
        p.stopStateLogging(logIDvideo)

    return data, data_labels


class RunExperiment():
    def __init__(self, render, debug, env_name, expert_name, expert_rel_path):
        self.render = render
        self.debug = debug
        self.env_name = env_name
        self.expert_name = expert_name
        self.expert_rel_path = expert_rel_path

    
    # Define what happens when the class is called.
    def __call__(self, sweep_args):
        config_filename = sweep_args["filename"] # Filename where run config is stored
        config = somosweep.iter_utils.load_yaml(config_filename)
        index = sweep_args["index"] # Unique index of the run
        replace_existing = sweep_args.get("replace", True) # decide whether data gets replaced or left alone

        log_filename = os.path.join(os.path.dirname(config_filename), "data.npy")
        log_labels_filename = os.path.join(os.path.dirname(config_filename), "data_labels.txt")

        print(index)

        # If data exists, cancel the run if we are not explicitly replacing.
        if os.path.exists(log_filename) and not replace_existing:
            return


        # Add impotant info to the run config
        config['expert_name'] = self.expert_name
        config['expert_rel_path'] = self.expert_rel_path
        config['log_path'] = os.path.dirname(config_filename)
        config['tmp_path'] = sweep_args["tmp_path"]
        config['run_index'] = index

        # Run the simulations
        data, data_labels = run_expert_trajectory(
            environment_name = self.env_name,
            run_config       = config,
            run_render       = self.render,
            debug            = self.debug,
            record_video      = False,
        )

        # Save data
        np.save(log_filename,data)

        # Save data labels
        with open(log_labels_filename,'w') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerow(data_labels)

        



def main():
    parser = argparse.ArgumentParser(
        description="Arguments to run an expert trajectory and optionally record data."
    )
    parser.add_argument(
        "-env",
        "--env_name",
        help="Environment name.",
        required=True)
    parser.add_argument(
        "-t",
        "--traj_name",
        help="Trajectory name.",
        required=True,
    )
    parser.add_argument(
        "-s",
        "--sweep_name",
        help="Name of the sweep configuration file.",
        required=True,
    )
    parser.add_argument(
        "-dp",
        "--data_path",
        help="Folder where data should be stored. If excluded, data is stored inside the expert folder",
        required=False,
        default=None)
    parser.add_argument(
        "-v",
        "--render",
        help="Render the environment.",
        action="store_true",
    )
    parser.add_argument(
        "-d", "--debug", help="Display SoMo-RL Debugging Dashboard", action="store_true"
    )
    parser.add_argument(
        "-dl",
        "--debug_list",
        nargs="+",
        help="List of debugger components to show in panel (space separated). Choose from reward_components, observations, actions, applied_torques",
        required=False,
        default=[],
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
        "-clean",
        "--clean",
        help="Delete data",
        action="store_true",
    )
    parser.add_argument(
        "-f",
        "--force",
        help="Force overwrite simulations.",
        action="store_true",
    )
    
    # Parse args
    arg = parser.parse_args()

    # Condense debugging flags
    debug = arg.debug
    if len(arg.debug_list) > 0:
        debug = copy.deepcopy(arg.debug_list)

    # Get the location of the expert trajectory
    expert_dir_abs_path = Path(os.path.dirname(__file__))
    expert_abs_path = Path(expert_dir_abs_path) / arg.traj_name
    expert_rel_path = Path(os.path.relpath(expert_abs_path))

    # Get the data storage location
    data_folder = arg.data_path
    if data_folder is None:
        data_folder = expert_rel_path / "data"
    
    # Get correct data path based on combo of sweep name and data path
    sweep_file = arg.sweep_name
    sweep_name = os.path.basename(sweep_file)
    sweep_name = Path(sweep_name).with_suffix("")
    data_path = os.path.join(data_folder,sweep_name)
    
    
    #Generate config files
    if arg.clean:
        delete = input("Are you sure you want to delete %s? (y/[N]): "%(data_path))
        ## Try to remove tree; if failed show an error using try...except on screen
        if delete == 'y':
            try:
                shutil.rmtree(data_path)
            except OSError as e:
                print ("Error: %s - %s." % (e.filename, e.strerror))

    if arg.setup_configs:
        run_gen = somosweep.RunGenerator(data_path)
        run_gen.from_file(os.path.join(expert_rel_path, sweep_file))

    # Run the simulation batch
    if arg.run_simulations:
        experiment_runner = RunExperiment(
            render = arg.render,
            debug = debug,
            env_name = arg.env_name,
            expert_name = arg.traj_name,
            expert_rel_path = expert_rel_path)
        batchsim = somosweep.BatchSimulation()
        batchsim.load_run_list(data_path, recalculate=arg.force)
        batchsim.run_from_function(
            run_function=experiment_runner,
            parallel=True,
            num_processes=None, # set the number of processes to use, otherwise use 1 per cpu core
        )


if __name__ == "__main__":
    main()