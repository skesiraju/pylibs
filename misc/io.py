#!/usr/bin/env python
# -*- coding: utf-8 -*-

# author : Santosh
# e-mail : kcraj2[AT]gmail[DOT]com
# Date created : 03 Dec 2015
# Last modified : 17 Feb 2016

"""
Simple utility functions
"""

from __future__ import print_function
from os import linesep
import csv
from collections import defaultdict


def chunkify(lst, n):
    """ chunkify a list to n chunks """
    return [lst[i::n] for i in xrange(n)]


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
    with open(fname, 'r') as fpr:
        flist = fpr.read().split(linesep)

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
    with open(fname, 'r') as fpr:
        for line in fpr:
            line = line.strip()
            vals = line.split(" ")
            st = float(vals[0])
            et = float(vals[1])
            tlist.append((st, et))

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
            for (k, v) in row.items():  # go over each column name and value
                # append the value into the appropriate list
                # based on column name k
                cols[k].append(v)
    return cols
