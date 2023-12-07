#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# author : Santosh Kesiraju
# e-mail : kcraj2[AT]gmail[DOT]com
# Date created : 03 Dec 2015
# Last modified : 07 Mar 2021

"""
Simple utility functions
"""

import os
import sys
import csv
import json
import logging
from datetime import datetime
from collections import defaultdict


def save_to_json(d, json_file):
    """Save the dictionary to json file"""

    with open(json_file, 'w', encoding='utf-8') as fpw:
        json.dump(d, fpw, indent=2, sort_keys=True, ensure_ascii=False)


def load_json(json_file):
    """Load json file into a dictionary"""
    data = {}
    if os.path.exists(json_file):
        with open(json_file, "r") as fpr:
            data = json.load(fpr)

    else:
        print("- ERROR: File not found.", json_file, file=sys.stderr)
        sys.exit()

    return data


def create_log_file(log_dir, phase, verbose=False):
    """Create a log file and return the logger.

    Args:
    -----
        log_dir (str): Path to dir where log file shall be saved.
        phase (str): Prefix name of the log file
        verbose (bool): Adds streamHandler so that log prints on stdout as well

    """

    now_str = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    log_file = os.path.join(log_dir, f"{phase}_{now_str}.log")
    logging.basicConfig(
        format="%(asctime)s %(message)s",
        datefmt="%d-%m-%Y %H:%M:%S",
        filename=log_file,
        filemode="w",
        level=logging.INFO,
    )
    print("- Log file:", log_file)
    logger = logging.getLogger()
    if verbose:
        logger.addHandler(logging.StreamHandler())
    return logger


def get_ivectors_from_h5(
    ivecs_h5_file,
    iter_num=-1,
    max_iters=None,
    dim_load="half",
    config=None,
    em_dim=None,
):
    """Load ivectors from h5 file and return them in numpy array.

    Args:
    ----
        ivecs_h5_file (str): Path to ivectors h5 file
        iter_num (int): Which iteration of ivectors should be loaded. -1 implies final
        max_iters (int): If not given, will try to determine automatically from file path (test_T2000_e500.h5)
        dim_load (str): Should load full or half dimension. Useful when the second half represents log-std.dev
        config (dict): config file (If None, will try to find the path to config file)
        em_dim (int): Embedding dimension (required if config is None and cannot be determined)

    Returns:
    --------
        numpy ndarray: n_samples x em_dim or n_samples x (em_dim*2) if dim_load is "full"

    """

    import h5py
    import numpy as np

    # == Determine max_iters from file name
    if not max_iters:
        try:
            max_iters = int(
                os.path.splitext(os.path.basename(ivecs_h5_file))[0].split("_")[-1][1:]
            )
        except ValueError:
            pass

    # == Determine em_dim
    cfg_file = ""
    if not config:
        cfg_file = os.path.join(os.path.dirname(ivecs_h5_file), "../config.json")
        if os.path.exists(cfg_file):
            with open(cfg_file, "r") as fpr:
                config = json.load(fpr)
        else:
            cfg_file = os.path.join(os.path.dirname(ivecs_h5_file), "../../config.json")
            if os.path.exists(cfg_file):
                with open(cfg_file, "r") as fpr:
                    config = json.load(fpr)

    if not em_dim and config:
        em_dim = config["hyper"]["K"]

    if not max_iters and config:
        max_iters = config["xtr_iters"]

    else:
        if em_dim is None:
            print(
                "config or dim should be given, since there is no config file at",
                cfg_file,
                file=sys.stderr,
            )
            sys.exit()

        if max_iters is None:
            print(
                "Cannot automatically determine `max_iters` from the file name",
                ivecs_h5_file,
                "Expected format *_T[INT]_e[INT].h5  (test_T5000_e1000.h5)",
                file=sys.stderr,
            )
            print(
                "Cannot get `max_iters (xtr_iters)` from config file", file=sys.stderr
            )
            sys.exit()

    try:
        ivecs_h5f = h5py.File(ivecs_h5_file, "r")
        ivecs_h5 = ivecs_h5f.get("ivecs")

        if iter_num == -1:
            ivecs = ivecs_h5.get(str(max_iters))[()]
        else:
            ivecs = ivecs_h5.get(str(iter_num))[()]

        # Transpose if the columns represent n_samples
        if ivecs.shape[0] in (2 * em_dim, em_dim):
            ivecs = ivecs.T

        if dim_load == "half":
            ivecs = ivecs[:, :em_dim]

    finally:
        ivecs_h5f.close()

    return ivecs


def convert_data_to_svm(data, labels):
    """Convert data matrix and labels to LIBSVM format

    Parameters:
    -----------
    data: numpy ndarray, rows = feats, cols = dim \n
    labels: numpy array, equal to no of rows in data \n

    Returns:
    --------
    data_lst: list where every element is a string
    """

    nof, dim = data.shape

    data_lst = []
    for i in range(nof):
        vec_str = str(labels[i]) + " "
        for j in range(dim):
            # index should start from 1
            vec_str += str(j + 1) + ":" + str(data[i, j]) + " "
        data_lst.append(vec_str.strip())
    return data_lst


def chunkify(lst, n_chunks):
    """ chunkify a list to n chunks """
    return [lst[i::n_chunks] for i in range(n_chunks)]


def read_simple_flist(fname, pre="", sfx=""):
    """read simple file line by line into list.
    Append the prefix and suffix for every line if given

    Args:
    fname (str): full path to file
    pre (str): string for prefix
    sfx (str): string for suffix

    Returns:
    list
    """

    flist = []
    with open(fname, "r", encoding="utf-8") as fpr:
        flist = [line.strip() for line in fpr if line.strip()]

    if flist[-1].strip() == "":
        flist = flist[:-1]

    if pre != "" or sfx != "":
        flist_new = [pre + f + sfx for f in flist]
        flist = flist_new

    return flist


def read_time_stamps(fname):
    """Read simple file that has time stamps (2 cols)
    line by line into list.

    Args:
    fname (str): full path to file

    Returns:
    list with tuples
    """

    tlist = []
    with open(fname, "r", encoding="utf-8") as fpr:
        for line in fpr:
            line = line.strip()
            vals = line.split(" ")
            stime = float(vals[0])
            etime = float(vals[1])
            tlist.append((stime, etime))

    return tlist


def load_key_file(kname):
    """Loads the key file in to dictionaries

    Parameters:
    -----------
    kname (str): full path to the key file

    Returns:
    --------
    collections.defaultdict
    """

    cols = defaultdict(list)
    with open(kname, "r") as fpr:
        reader = csv.DictReader(fpr)
        # read a row as {column1: value1, column2: value2,...}
        for row in reader:
            for (k, val) in row.items():  # go over each column name and value
                # append the value into the appropriate list
                # based on column name k
                cols[k].append(val)
    return cols


def write_simple_flist(some_list, out_fname):
    """Write the elements in the list line by line
    in the given out file

    Parameters:
    -----------
    some_list (list): list of elements
    out_fname (str): output file name

    """

    with open(out_fname, "w", encoding="utf-8") as fpw:
        fpw.write("\n".join(some_list))
