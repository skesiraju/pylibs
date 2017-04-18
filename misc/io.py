#!/usr/bin/env python
# -*- coding: utf-8 -*-

# author : Santosh
# e-mail : kcraj2[AT]gmail[DOT]com
# Date created : 03 Dec 2015
# Last modified : 17 Feb 2016

"""
Simple utility functions
"""

import csv
import codecs
from collections import defaultdict


def convert_data_to_svm(data, labels):
    """ Convert data matrix and labels to LIBSVM format

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
            vec_str += str(j+1) + ":" + str(data[i, j]) + " "
        data_lst.append(vec_str.strip())
    return data_lst


def chunkify(lst, n_chunks):
    """ chunkify a list to n chunks """
    return [lst[i::n_chunks] for i in range(n_chunks)]


def read_simple_flist(fname, pre="", sfx=""):
    """ read simple file line by line into list.
    Append the prefix and suffix for every line if given

    Args:
    fname (str): full path to file
    pre (str): string for prefix
    sfx (str): string for suffix

    Returns:
    list
    """

    flist = []
    with codecs.open(fname, 'r', 'utf-8') as fpr:
        flist = fpr.read().split("\n")

    if flist[-1].strip() == "":
        flist = flist[:-1]

    if pre != "" or sfx != "":
        flist_new = [pre + f + sfx for f in flist]
        flist = flist_new

    return flist


def read_time_stamps(fname):
    """ Read simple file that has time stamps (2 cols)
    line by line into list.

    Args:
    fname (str): full path to file

    Returns:
    list with tuples
    """

    tlist = []
    with codecs.open(fname, 'r', 'utf-8') as fpr:
        for line in fpr:
            line = line.strip()
            vals = line.split(" ")
            stime = float(vals[0])
            etime = float(vals[1])
            tlist.append((stime, etime))

    return tlist


def load_key_file(kname):
    """ Loads the key file in to dictionaries

    Parameters:
    -----------
    kname (str): full path to the key file

    Returns:
    --------
    collections.defaultdict
    """

    cols = defaultdict(list)
    with open(kname, 'r') as fpr:
        reader = csv.DictReader(fpr)
        # read a row as {column1: value1, column2: value2,...}
        for row in reader:
            for (k, val) in row.items():  # go over each column name and value
                # append the value into the appropriate list
                # based on column name k
                cols[k].append(val)
    return cols


def write_simple_flist(some_list, out_fname):
    """ Write the elements in the list line by line
    in the given out file

    Parameters:
    -----------
    some_list (list): list of elements
    out_fname (str): output file name

    """

    with codecs.open(out_fname, "w", "utf-8") as fpw:
        fpw.write("\n".join(some_list))
