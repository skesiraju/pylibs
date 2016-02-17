#!/usr/bin/env python
# -*- coding: utf-8 -*-

# author: Santosh
# email; kcraj2[AT]gmail[DOT]com

# Date created: Nov 9, 2015
# Last modified: Jan 17, 2016

"""
A bunch of pre-processing routines for text based
information retrieval (IR) tasks
"""


from __future__ import print_function
import codecs
import os
import re
import sys
import string
import numpy as np
import scipy.sparse
from math import log


def replace_data(data, replace):
    """ replace the certain tokens in the data according to
    the tuples in the list replace. Input data is string """

    for pair in replace:
        patt = re.compile(pair[0])
        data = re.sub(patt, pair[1], data)
    data = re.sub("\s\s+", " ", data)  # remove multiple spaces
    return data


def remove_punc(data):
    """ Remove punctuation easily. Input data is string """
    clean_data = data.translate(string.maketrans("", ""), string.punctuation)
    clean_data = re.sub("\s\s+", " ", clean_data)  # remove multiple spaces
    return clean_data


def clean_data(data, clean):
    """ clean the data by removing the symbols in clean list.
    Input data is string """

    patt = re.compile("|".join(clean))
    clean_data = re.sub(patt, '', data)
    clean_data = re.sub("\s\s+", " ", clean_data)  # remove multiple spaces
    return clean_data.strip()


def get_top_ngrams_per_class(DbyW, vocab, labels, top, prior_weight=False):
    """ Get top (ML sense) n-grams (feats) per class(topic)

    Parameters:
    -----------
    DbyW (scipy.sparse): Doc by Word matrix with freq. \n
    vocab (list): list of vocabulary. \n
    labels (numpy.ndarray): class labels (may start from 0 or 1). \n
    top (int): top ngrams per class (topic). \n
    prior_weight (bool): If True, uniform prior weights are chosen and
    will added to the ML estimated ngrams per class. \n

    Returns:
    -------
    list : list of ngrams
    """

    D, W = DbyW.shape

    uniq_lab = sorted(np.unique(labels))
    L = len(uniq_lab)
    lab_ixs = np.zeros(labels.shape)

    print('D, W, L:', D, W, L)

    for i in xrange(len(labels)):
        lab_ixs[i] = uniq_lab.index(labels[i])

    DbyL = scipy.sparse.csr_matrix((np.ones(D), (np.arange(D), lab_ixs)),
                                   shape=(D, L))

    WbyL = DbyW.T.dot(DbyL)
    print('DbyL:', DbyL.shape, 'WbyL:', WbyL.shape)

    if prior_weight:
        # non-uniform priors
        # priors = DbyL.sum(axis=0).T / float(D)
        # print('non-uni priors:', priors.shape, priors.T)

        # uniform priors
        priors = np.ones((L, 1), dtype=np.float32) / float(L)
        # print('priors:', priors.shape, priors.T)

        WbyL = (WbyL.T + (priors * L)).T

    P_WL = np.asarray(WbyL / WbyL.sum(axis=1))
    print('P_WL:', P_WL.shape)

    all_ixs = []
    for j in xrange(P_WL.shape[1]):
        w_ixs = np.argsort(P_WL[:, j])[-top:]
        all_ixs.append(w_ixs)

    uniq_ixs = np.unique(np.concatenate(all_ixs, axis=0))

    class_vocab = []
    for uix in uniq_ixs:
        class_vocab.append(vocab[uix])

    return sorted(class_vocab)


def update_postings(post_d, tf, d_ix):
    """ Given postings, term freq and doc ix (not ID or path), update the
    postings.

    Parameters:
    -----------
    post_d (dict): postings dict (dict with a dict), {word: {doc_ix:freq}} \n
    tf (dict): term freq dictionary \n
    fid (int): file id as integer (index)

    Returns:
    --------
    post_d (dict): Updates the give post_d and returns it
    """

    for tok, freq in tf.iteritems():
        tmp_d = {}
        if tok in post_d:
            tmp_d = post_d[tok]

            if d_ix in tmp_d:
                print('Doc index:', d_ix, 'already in the postings.')
                print('This cannot happen.')
                sys.exit()

        tmp_d[d_ix] = freq
        post_d[tok] = tmp_d
    return post_d


def convert_postings_to_dbyw(post_d, ndocs, idf=False):
    """ Convert the postings to doc by word sparse matrix. Row indices are
    doc indices from postings and column indices are word indices from
    vocabulary.

    Parameters:
    -----------
    post_d (dict): postings dict (dict with a dict), {word: {doc_ix:freq}} \n
    ndocs (int): number of documents \n
    idf (bool): inverse doc freq, default=False


    Returns:
    --------
    scipy.sparse.coo_matrix: Doc-by-word matrix \n
    vocab (list): list of vocabulary in same order as cols in Doc-by-word
    matrix.
    """

    vocab = post_d.keys()

    D = ndocs
    W = len(vocab)

    rows = []
    cols = []
    vals = []
    for tok, doc_d in post_d.iteritems():
        w_ix = vocab.index(tok)
        doc_ixs = doc_d.keys()

        if idf:
            cnts = [doc_d[d] * log(float(D)/len(doc_ixs)) for d in doc_ixs]
        else:
            cnts = [doc_d[d] for d in doc_ixs]

        cols += [w_ix] * len(doc_ixs)
        rows += doc_ixs
        vals += cnts

    DbyW = scipy.sparse.coo_matrix((vals, (rows, cols)), shape=(D, W))

    return DbyW, vocab


def convert_postings_to_tf(post_d):
    """ Given a postings dict, return words and their total freq.

    Parameters:
    -----------
    post_d (dict with a dict): {word: {doc_id: freq}}

    Returns:
    --------
    dict: {word: freq}
    """

    tf = {}
    for k, doc_d in post_d.iteritems():
        tf[k] = sum(doc_d.values())
    return tf


def get_term_freq(content, n=1, vocab=None, replace=None,
                  clean=None, to_lower=False, remove_punc=False):
    """ Get the term frequency from a given list of strings or file.
    It also does basic pruning.

    Parameters:
    -----------
    content (list or str): list of strings of path to file \n
    n=1 (int): n in n-gram, default is unigram \n
    vocab (list): vocabulary list \n
    replace (list of tuples): list of tuples, where the first element is
    replaced by the second one \n
    clean (list): list of symbols (with escape seq) to be removed from
    the text \n
    to_lower (bool): if True, tokens are converted to lower case \n
    remove_punc (bool): Removes punctuation if True

    Returns:
    --------
    d (dictionary): terms with corresponding freq
    """

    data = []

    # in case the content is a string with file name
    if type(content) is str:
        if os.path.exists(content) is False:
            print("Cannot find", content)
            sys.exit()
        else:
            with codecs.open(content, 'r') as fpr:
                data = fpr.read().split(os.linesep)
    elif type(content) is list:
        data = content
    else:
        print('Input should be a list of strings or path to a file.')
        sys.exit()

    # sort the vocab; takes time; but look up is faster
    if vocab is not None:
        vocab = set(sorted(vocab))

    tf_d = {}

    for line in data:
        line = line.strip()

        if to_lower:
            line = line.lower()
        if clean is not None:
            line = clean_data(line, clean)
        if replace is not None:
            line = replace_data(line, replace)
        if remove_punc:
            line = remove_punc(line)

        tokens = line.split(" ")

        for i in xrange(len(tokens) - n+1):
            tok = " ".join(tokens[i: i+n]).strip()
            if(vocab is not None):
                if(tok not in vocab):
                    continue

            if(tok in tf_d):
                tf_d[tok] += 1
            else:
                tf_d[tok] = 1
    return tf_d


class TextVector:

    def __init__(self, n=1, vocab=None, clean=None, replace=None,
                 to_lower=False, remove_punc=False, idf=False):
        """

        Parameters:
        -----------
        n=1 (int): n in n-gram, default is unigram \n
        vocab (list): vocabulary list \n
        replace (list of tuples): list of tuples, where the first element is
        replaced by the second one \n
        clean (list): list of symbols (with escape seq) to be removed from
        the text \n
        to_lower (bool): if True, tokens are converted to lower case \n
        remove_punc (bool): Removes punctuation if True \n
        idf (bool): Apply inverse doc freq weighting ?
        """

        self.n = n
        self.vocab = vocab
        self.clean = clean
        self.replace = replace
        self.to_lower = to_lower
        self.remove_punc = remove_punc
        self.idf = idf

    def get_postings(self, flist):
        """ Get the postings (dict), i.e., tf w.r.t every file.
        This function assumes that all the files are stripped of punctuation
        and other junk symbols.

        Parameters:
        -----------
        flist (list): list of file names with full path
        n=1 (int): n in n-gram, default is unigram

        Returns:
        --------
        d (dict): (postings) terms occurring in docs with corresponding freq
        """

        if type(flist) is not list:
            print('Input should be list of files with full path.')
            print('Exiting..')
            sys.exit()

        # print('NOTE: Doc IDs are indices in the file list.')

        post_d = {}

        for doc_id in xrange(len(flist)):

            fname = flist[doc_id]

            if os.path.exists(fname) is False:
                print(fname, 'does not exist. Skipping..')
                continue

            with codecs.open(fname, 'r') as fpr:
                data = fpr.read()

                if self.to_lower:
                    data = data.lower()
                if self.clean is not None:
                    data = clean_data(data, self.clean)
                if self.replace is not None:
                    data = replace_data(data, self.replace)
                if self.remove_punc:
                    data = remove_punc(data)

                lines = data.split(os.linesep)

            for line in lines:
                line = line.strip()
                tokens = line.split(" ")

                for i in range(len(tokens) - self.n+1):
                    tok = " ".join(tokens[i: i+self.n]).strip()

                    tmp_d = {}
                    if tok in post_d:
                        tmp_d = post_d[tok]
                        if doc_id in tmp_d:
                            tmp_d[doc_id] += 1
                        else:
                            tmp_d[doc_id] = 1
                    else:
                        tmp_d[doc_id] = 1

                    post_d[tok] = tmp_d

        return post_d

    def fit_postings(self, flist):
        """ Fit the postings using the existing vocabulary

        Parameters:
        -----------
        flist (list): list of file names with full path

        Returns:
        --------
        d (dict): (postings) terms occurring in docs with corresponding freq
        """

        if self.vocab is None:
            print('Vocabulary is empty. First generate the vocabulary.')
            print('Exiting..')
            sys.exit()

        # init postings dictionary with empty entries for each vocab
        post_d = {}
        for v in self.vocab:
            post_d[v] = {}

        for doc_id in xrange(len(flist)):

            fname = flist[doc_id]

            if os.path.exists(fname) is False:
                print(fname, 'does not exist. Skipping..')
                continue

            with codecs.open(fname, 'r') as fpr:
                data = fpr.read()

                if self.to_lower:
                    data = data.lower()
                if self.clean is not None:
                    data = clean_data(data, self.clean)
                if self.replace is not None:
                    data = replace_data(data, self.replace)
                if self.remove_punc:
                    data = remove_punc(data)

                lines = data.split(os.linesep)

            for line in lines:
                line = line.strip()
                tokens = line.split(" ")

                for i in range(len(tokens) - self.n+1):
                    tok = " ".join(tokens[i: i+self.n]).strip()

                    tmp_d = {}
                    if tok in post_d:
                        tmp_d = post_d[tok]
                        if doc_id in tmp_d:
                            tmp_d[doc_id] += 1
                        else:
                            tmp_d[doc_id] = 1

                        post_d[tok] = tmp_d

        return post_d

    def get_doc_by_word_matrix(self, flist):
        """ Get document-by-word matrix (sparse) where every element is
        the tf. This function assumes that all the files are stripped of
        punctuation and other junk symbols.

        Parameters:
        -----------
        flist (list): list of file names with full path \n

        Returns:
        --------
        scipy.sparse.coo_matrix: (rows = doc IDs and cols = words) \n
        list of vocabulary: list and the sequence reflects the cols
        in DxW matrix.
        """

        if type(flist) is not list:
            print('Input should be list of files.')
            print('Exiting..')
            sys.exit()

        # print('NOTE: Doc IDs are indices in the file list.')

        post_d = self.get_postings(flist)

        self.vocab = post_d.keys()

        D = len(flist)

        DbyW, self.vocab = convert_postings_to_dbyw(post_d, D, self.idf)

        return DbyW, self.vocab

    def fit_doc_by_word_matrix(self, flist):
        """ Get the doc by word matrix for the files in flist using the given
        vocabulary

        Paramters:
        ----------
        flist (list): list with file names with full path \n
        vocab (list): list of vocabulary \n

        Returns:
        --------
        scipy.sparse.coo_matrix: (rows = doc IDs and cols = words) \n
        """

        if type(flist) is not list:
            print('Input should be list of files.')
            print('Exiting..')
            sys.exit()

        # print('NOTE: Doc IDs are indices in the file list.')

        post_d = self.fit_postings(flist)

        D = len(flist)

        DbyW, _ = convert_postings_to_dbyw(post_d, D, self.idf)

        return DbyW