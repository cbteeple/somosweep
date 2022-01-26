import os
import sys
import shutil
import yaml
import copy
from functools import reduce  # forward compatibility for Python 3
import operator
import pybullet as p
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
from natsort import natsorted
from somo.logger import LogReader


def add_tmp(tmp="_tmp"):
    """
    Add a temporary directory

    Parameters
    ----------
    tmp : str
        Name of tmp directory. Default is ``"_tmp"``, used as a relative path.
    """
    delete_tmp(tmp)
    if not os.path.exists(tmp):
        os.makedirs(tmp)


def delete_tmp(tmp="_tmp"):
    """
    Clean up a temporary folder

    Parameters
    ----------
    tmp : str
        Name of tmp directory. Default is ``"_tmp"``, used as a relative path.
    """
    if os.path.exists(tmp):
        shutil.rmtree(tmp)


def save_yaml(yaml_dict, filename):
    """
    Save any python object to a yaml file 

    Parameters
    ----------
    yaml_dict : dict, list
        Object to save
    filename : str
        Filename to save
    """
    try:
        with open(filename, "w") as f:
            yaml.dump(yaml_dict, f, default_flow_style=None)
    except:
        raise(ValueError("Unable to save yaml: '%s'"%(filename)))


def load_yaml(filename):
    """
    Load any yaml file into a python object 

    Parameters
    ----------
    filename : str
        Filename to save

    Returns
    -------
    object : object
        Yaml object loaded from the file
    """
    try:
        with open(filename, "r") as f:
            yaml_dict = yaml.safe_load(f)
        return yaml_dict
    except:
        raise(ValueError("Unable to open yaml: '%s'"%(filename)))


def get_files_recursively(start_directory, filter_extension=None):
    """
    Recursively get all files with a specific extension

    Parameters
    ----------
    start_directory : str
        Directory to start in
    filter_extension : str
        File extension to use as filter (return only files with
        this extension)

    Yields
    -------
    files : iterable
        Iterable object with all filenames meeting the criteria. when iterated upon
        (ie. for loop), returns a tuple: ``(root_path, filename, combined_path)`` 
    """
    for root, dirs, files in os.walk(start_directory):
        for file in files:
            if filter_extension is None or file.lower().endswith(filter_extension):
                yield (root, file, os.path.abspath(os.path.join(root, file)))


def scrape_folder(directory, filter_extensions=[], file_blacklist=[]):
    """
    Recursively get all files with a specific extension

    Parameters
    ----------
    start_directory : str
        Directory to start in
    filter_extensions : list
        List of file extensions to use as filters (return only files with
        these extension)
    file_blacklist : list
        List of filenames to explicitly exclude

    Returns
    -------
    files : list
        List of files meeting the criteria
    """
    directory_usr = os.path.expanduser(directory)
    val_list = []
    if os.path.isdir(directory_usr):
        dir_list = os.listdir(directory_usr)
        dir_list = natsorted(dir_list)
        for f in dir_list:
            ext = os.path.splitext(f)[-1].lower()
            ext_accept = not filter_extensions or ext in filter_extensions
            file_accept = os.path.basename(f) not in file_blacklist and os.path.isfile(
                os.path.join(directory_usr, f)
            )
            if ext_accept and file_accept:
                val_list.append(os.path.join(directory, f))
    return val_list


def get_folders(directory, blacklist=[]):
    """
    Get all files in a folder (not recursive)

    Parameters
    ----------
    directory : str
        Directory to start in
    file_blacklist : list
        List of filenames to explicitly exclude

    Returns
    -------
    files : list
        List of files meeting the criteria
    """
    directory_usr = os.path.expanduser(directory)
    val_list = []
    if os.path.isdir(directory_usr):
        dir_list = os.listdir(directory_usr)
        dir_list = natsorted(dir_list)
        for f in dir_list:
            folder_accept = os.path.isdir(os.path.join(directory_usr, f))
            blacklist_accept = os.path.basename(f) not in blacklist
            if folder_accept and blacklist_accept:
                val_list.append(os.path.join(directory, f))
    return val_list


# Parse data from a file
def read_parse_data(filename, verbose=False):
    """
    Read data from a pybullet log file.

    Parameters
    ----------
    filename : str
        Filename to read
    verbose : bool
        Show all the inner working of the log parser.

    Returns
    -------
    reader : LogReader
        Log reader object
    """
    reader = LogReader()
    reader.read(filename, verbose)
    return reader


# Get the folder of the current simulation group
def get_group_folder(config, data_path):
    data_folder = data_path

    # data_folder = config['save']['folder']
    group_name = config["save"]["group_name"]

    save_folder = os.path.expanduser(os.path.join(data_folder, group_name))

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    return save_folder


# Generate the filename and folder name to save data
def generate_save_location(config, data_path, filename="data.dat"):
    """
    Generate correct sub-folder structure for runs to be saved in

    Parameters
    ----------
    config : dict
        Run config to use
    data_path : str
        Path where data should be stored
    filename : str, optional
        Filename used to store data

    Returns
    -------
    log_filename : str
        Filename to use for logging data
    out_folder : str
        Sub-folder generated
    """
    run_name = config["save"].get("run_name", None)
    if run_name is None:
        run_name = datetime.now().strftime("%Y%m%d_%H%M%S")

    save_folder = data_path #get_group_folder(config, data_path)
    out_folder = os.path.join(save_folder, run_name)

    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    log_filename = os.path.join(out_folder, filename)

    return (log_filename, out_folder)


def get_from_dict(dataDict, mapList):
    """
    Get a specific element of a dictionary by specifying a "path"

    Parameters
    ----------
    dataDict : dict
        Dictionary to access
    mapList : list
        List of path segments (ie. ``path/to/item`` becomes ``[path, to, item]``)

    Returns
    -------
    value : Any
        Value accessed by the mapList
    """
    if len(mapList) == 0:
        return dataDict
    mapList_use = copy.deepcopy(mapList)
    last_var_idx = None
    if "[" in mapList_use[-1]:
        var_list = mapList_use[-1].split("[")
        last_var_name = var_list[0]
        last_var_idx = int(var_list[1].split("]")[0])

        mapList_use[-1] = last_var_name

    val = reduce(operator.getitem, mapList_use, dataDict)
    if last_var_idx is None:
        return val
    else:
        return val[last_var_idx]


def set_in_dict(dataDict, mapList, value):
    """
    Get a specific element of a dictionary by specifying a "path"

    Parameters
    ----------
    dataDict : dict
        Dictionary to access
    mapList : list
        List of path segments (ie. ``path/to/item`` becomes ``[path, to, item]``)
    value : Any
        Value to insert
    """
    #print(mapList)
    mapList_use = copy.deepcopy(mapList)
    last_var_idx = None
    if "[" in mapList_use[-1]:
        var_list = mapList_use[-1].split("[")
        last_var_name = var_list[0]
        last_var_idx = int(var_list[1].split("]")[0])

        mapList_use[-1] = last_var_name

    pendultimate_val = get_from_dict(dataDict, mapList_use[:-1])
    if last_var_idx is None:
        pendultimate_val[mapList_use[-1]] = value
    else:
        pendultimate_val[mapList_use[-1]][last_var_idx] = value


def parse_variable_name(var_str):
    """
    Get a path list from a "path" for use with ``get_from_dict()``

    ``path/to/item`` is converted to ``[path, to, item]``

    Parameters
    ----------
    var_str : str
        Path to the dict item (i.e. ``path/to/item``)
    
    Returns
    -------
    map_list : list
        List of path (i.e. ``[path, to, item]``)
    """
    if isinstance(var_str, str):
        keys = var_str.split("/")

        if isinstance(keys, str):
            return [keys]
        else:
            return keys

    else:
        return None


# Generate and save the actuation data
def log_actuation(log_filename, cyc_filename, actuation_fn=None, cycle_fn=None):
    # If there are no functions, do nothing
    if actuation_fn is None and cycle_fn is None:
        return False

    reader = read_parse_data(log_filename)
    df = reader.make_dataframe(["timeStamp", "objectId"])
    times = df["timeStamp"].values.squeeze()

    # Build the actuation data object
    cycles_out = dict()
    cycles_out["timeStamp"] = df["timeStamp"].values

    # Build cycle data
    if isinstance(cycle_fn, dict):
        for key in cycle_fn:
            cycles_out["cycles_" + key] = cycle_fn[key](times.tolist())
    elif cycle_fn is not None:
        cycles_out["cycles"] = cycle_fn(times.tolist())

    # Build actuation data
    if isinstance(actuation_fn, dict):
        for key in actuation_fn:
            curr_act = actuation_fn[key]

            if isinstance(curr_act, list):
                for idx, curr_chan in enumerate(curr_act):
                    cycles_out["actuation_" + key + "%d" % (idx)] = curr_chan(
                        times.tolist()
                    )
            else:
                cycles_out["actuation_" + key] = actuation_fn[key](times.tolist())
    elif actuation_fn is not None:
        cycles_out["actuation"] = actuation_fn(times.tolist())

    # Save the actuation data to a file
    df_cyc = pd.DataFrame(cycles_out)
    df_cyc.to_pickle(cyc_filename)


def auto_inc_file(in_path, fullpath=False, index=0):
    """
    Check filenames and automatically increment them to
    avoid overwriting.

    Parameters
    ----------
    in_path : str
        Path to check
    fullpath : bool
        Is this a fullpath?
    index : int
        Numerical suffix to start with
    
    Returns
    -------
    fixed_path : str
        The path with auto-incremented suffix
    """
    path = os.path.abspath(in_path)

    in_dirname = os.path.dirname(in_path)

    # if not os.path.exists(path):
    #    return path

    root, ext = os.path.splitext(os.path.expanduser(path))
    dir = os.path.dirname(root)
    fname = os.path.basename(root)
    # candidate = fname+ext
    candidate = "{}_{}{}".format(fname, str(index).zfill(5), ext)
    ls = set(os.listdir(dir))
    while candidate in ls:
        candidate = "{}_{}{}".format(fname, str(index).zfill(5), ext)
        index += 1

    if fullpath:
        out = os.path.join(dir, candidate)
    else:
        out = os.path.join(in_dirname, candidate)
    return out



# Plot the logged data (object position and orientation)
def graph_data(df=None, filename=None, cyc_filename=None, show=False, cyclic_key=None):
    # Get the raw data
    if df is None:
        fields = [
            "timeStamp",
            "objectId",
            "posX",
            "posY",
            "posZ",
            "oriX",
            "oriY",
            "oriZ",
            "oriW",
        ]

        reader = read_parse_data(filename)
        objectpose_df = reader.make_dataframe(fields)

        # Get euler angles from the quaternions
        euler = []
        for quaternion in zip(
            objectpose_df["oriX"],
            objectpose_df["oriY"],
            objectpose_df["oriZ"],
            objectpose_df["oriW"],
        ):
            # print(quaternion)
            euler.append(p.getEulerFromQuaternion(quaternion))

        euler = np.array(euler)
        euler = np.unwrap(euler, axis=0)
        euler = np.rad2deg(euler)

        objectpose_df["eulerX"] = euler[:, 0]
        objectpose_df["eulerY"] = euler[:, 1]
        objectpose_df["eulerZ"] = -euler[:, 2]

        objectpose_df = objectpose_df - objectpose_df.iloc[0].values.squeeze()

    else:
        reader = LogReader()
        objectpose_df = df

    # Make a new plot for each tracked object
    print("Plotting raw data...")

    for (o_id, df_curr) in objectpose_df.groupby("objectId"):

        # Set up the plot nicely
        rcParams["lines.linewidth"] = 2
        rcParams["font.size"] = 14
        rcParams["font.family"] = "Arial"
        sns.set_style("ticks")
        sns.set_palette([(1, 0, 0), (0, 1, 0), (0, 0, 1)])
        fig = plt.figure()

        # Plot object position
        plt.subplot(2, 1, 1)
        df_pos = df_curr[["timeStamp", "posX", "posY", "posZ"]]
        reader.plot(data=df_pos, x="timeStamp", y="vals", hue="cols")
        plt.xlabel("")
        plt.ylabel("Position (m?)")
        leg = plt.legend(loc="upper right")
        texts = leg.get_texts()
        texts[0].set_text("Axes")
        labels = ["X", "Y", "Z"]
        for idx, text in enumerate(texts[1:]):
            text.set_text(labels[idx])

        # Plot object orientation
        plt.subplot(2, 1, 2)
        df_ori = df_curr[["timeStamp", "eulerX", "eulerY", "eulerZ"]]
        reader.plot(data=df_ori, x="timeStamp", y="vals", hue="cols")
        plt.xlabel("Time (sec)")
        plt.ylabel("Orientation (deg)")
        ax = plt.gca()
        ax.get_legend().remove()

        # Pretty-up the plot
        plt.tight_layout()

        # Save the plot to a file and show it
        graph_filename = filename
        graph_filename = graph_filename.replace(".dat", "")
        plt.savefig(graph_filename + "_%d" % (int(o_id)) + ".png", dpi=450)

        if show:
            plt.show()

        plt.close()

    print("\t" + "Done!")

    if cyclic_key is not None:
        graph_cyclic(objectpose_df, cyc_filename, cyclic_key)


def graph_cyclic(df, cyc_filename, cycle_key, show=False):
    # Add in the cycle information
    df_cyc = pd.read_pickle(cyc_filename)

    df["cycles"] = df_cyc["cycles_" + cycle_key].values

    if 0 not in df["cycles"].unique():
        print("There is no cyclic part in this file")
    else:
        df = df[df["cycles"] >= 0]
        df = df.sort_values(["cycles", "timeStamp"])
        times_all = []
        for cycle_curr in df["cycles"].unique():
            times = df[df.cycles == cycle_curr].values
            df[df.cycles == cycle_curr] = times - times[0, :]
            # times_all.extend(times-times[0])

        df["timeStamp"] = np.round(df["timeStamp"].values, 2)

    # Make a new plot for each tracked object
    print("Plotting cyclic data...")

    for (o_id, df_curr) in df.groupby("objectId"):

        # Set up the plot nicely
        rcParams["lines.linewidth"] = 2
        rcParams["font.size"] = 14
        rcParams["font.family"] = "Arial"
        sns.set_style("ticks")
        sns.set_palette([(1, 0, 0), (0, 1, 0), (0, 0, 1)])
        fig = plt.figure()
        labels = ["X", "Y", "Z"]
        # Plot object position
        plt.subplot(2, 1, 1)
        df_pos = df_curr[["timeStamp", "posX", "posY", "posZ"]]
        group = df_pos.groupby("timeStamp")
        means = group.mean().values
        std = group.std().values
        times = group.groups.keys()
        # reader.plot(data=df_pos, x="timeStamp",  y="vals", hue='cols')
        for i in range(means.shape[1]):
            plt.plot(times, means[:, i], label=labels[i])
            plt.fill_between(
                times, means[:, i] - std[:, i], means[:, i] + std[:, i], alpha=0.3
            )

        plt.xlabel("")
        plt.ylabel("Position (m?)")
        leg = plt.legend(loc="upper right", frameon=False)

        # Plot object orientation

        plt.subplot(2, 1, 2)
        df_ori = df_curr[["timeStamp", "eulerX", "eulerY", "eulerZ"]]
        group = df_ori.groupby("timeStamp")
        means = group.mean().values
        means[:, 2] = -means[:, 2]
        std = group.std().values
        times = group.groups.keys()
        # reader.plot(data=df_pos, x="timeStamp",  y="vals", hue='cols')
        for i in range(means.shape[1]):
            plt.plot(times, means[:, i], label=labels[i])
            plt.fill_between(
                times, means[:, i] - std[:, i], means[:, i] + std[:, i], alpha=0.3
            )

        plt.xlabel("Time (sec)")
        plt.ylabel("Orientation (deg)")

        # Pretty-up the plot
        plt.tight_layout()

        # Save the plot to a file and show it
        graph_filename = cyc_filename
        graph_filename = graph_filename.replace(".act", "")
        plt.savefig(graph_filename + "_cyc_%d" % (int(o_id)) + ".png", dpi=450)

        if show:
            plt.show()

        plt.close()

    print("\t" + "Done!")