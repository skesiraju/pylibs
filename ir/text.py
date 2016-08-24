#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# author: Santosh
# email; kcraj2[AT]gmail[DOT]com

# Date created: Nov 9, 2015
# Last modified: March 07, 2016

"""
A bunch of pre-processing routines for text based
information retrieval (IR) tasks
"""


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
    translator = str.maketrans({key: None for key in string.punctuation})
    clean_data = data.translate(translator)
    clean_data = re.sub("\s\s+", " ", clean_data)  # remove multiple spaces
    return clean_data


def clean_data(data, clean):
    """ clean the data by removing the symbols in clean list.
    Input data is string """

    patt = re.compile("|".join(clean))
    clean_data = re.sub(patt, '', data)
    clean_data = re.sub("\s\s+", " ", clean_data)  # remove multiple spaces
    return clean_data.strip()


def get_soft_postings(in_fpaths, vocab=None):
    """ Get postings with freq replaced by soft counts

    Parameters:
    -----------
    in_fpaths: list with full paths

    Returns:
    --------
    dictionary: {trigram: {doc_id: freq; ..}; ..}
    """

    spost = {}

    if vocab is not None:
        for v in vocab:
            spost[v] = {}
        print('Postings init with empty values for', len(vocab),
              'keys. Post len:', len(spost))

    for doc_id in range(len(in_fpaths)):
        fpath = in_fpaths[doc_id]
        with open(fpath, 'r') as fpr:
            for line in fpr:
                line = line.strip()
                if line == "":
                    continue

                vals = line.split(";")
                for val in vals:
                    if val.strip() == '':
                        continue

                    parts = val.split("=")
                    if len(parts) < 2:
                        print('val:', val, 'line:', line)
                        sys.exit()

                    tok = parts[0]  # phoneme trigram
                    freq = float(parts[1])

                    tmp_d = {}
                    try:
                        tmp_d = spost[tok]
                        try:
                            tmp_d[doc_id] += freq
                        except KeyError:
                            tmp_d[doc_id] = freq
                    except KeyError:
                        tmp_d[doc_id] = freq

                    spost[tok] = tmp_d

    return spost


def get_top_ngrams_per_class(DbyW, vocab, labels, top, prior_weight=True):
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

    # print('D, W, L:', D, W, L)

    for i in range(len(labels)):
        lab_ixs[i] = uniq_lab.index(labels[i])

    DbyL = scipy.sparse.csr_matrix((np.ones(D), (np.arange(D), lab_ixs)),
                                   shape=(D, L))

    WbyL = DbyW.T.dot(DbyL)
    # print('DbyL:', DbyL.shape, 'WbyL:', WbyL.shape)

    if prior_weight:
        # non-uniform priors
        # priors = DbyL.sum(axis=0).T / float(D)
        # print('non-uni priors:', priors.shape, priors.T)

        # uniform priors
        priors = np.ones((L, 1), dtype=np.float32) / float(L)
        # print('priors:', priors.shape, priors.T)

        WbyL = (WbyL.T + (priors * L)).T

    else:
        print("ERROR: prior_weight=False, not implemented.")
        sys.exit()

    P_WL = np.asarray(WbyL / WbyL.sum(axis=1))
    # print('P_WL:', P_WL.shape)

    all_ixs = []
    for j in range(P_WL.shape[1]):
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

    for tok, freq in tf.items():
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


def convert_postings_to_dbyw(post_d, ndocs, vocab=None):
    """ Convert the postings to doc by word sparse matrix. Row indices are
    doc indices from postings and column indices are word indices from
    vocabulary. In the postings, doc ID are doc indices (int) not alpha-num.

    Parameters:
    -----------
    post_d (dict): postings dict (dict with a dict), {word: {doc_ix:freq}} \n
    ndocs (int): number of documents \n\
    vocab (list): If given, the same sequence is followed for cols in DbyW,
    else vocab will be sorted keys from postings \n

    Returns:
    --------
    scipy.sparse.coo_matrix: Doc-by-word matrix \n
    vocab (list): list of vocabulary in same order as cols in Doc-by-word
    matrix.
    """

    if vocab is None:
        vocab = sorted(post_d.keys())
        # print('Vocab is obtained from postings, using sorted sequence.')
    # else:
        #  print('Vocab given, using the same sequence to generate DbyW.')

    if len(post_d) != len(vocab):
        print("No. of tokens in postings do not match with the given vocab",
              "size. But that's okay, I will consider only the ones that",
              "are present in the vocab.")

    vocab = vocab
    D = ndocs
    W = len(vocab)

    rows = []
    cols = []
    vals = []

    for tok, doc_d in post_d.items():

        try:
            
            w_ix = vocab.index(tok)
            doc_ixs = doc_d.keys()

            cnts = [doc_d[d] for d in doc_ixs]

            cols += [w_ix] * len(doc_ixs)
            rows += doc_ixs
            vals += cnts

        except ValueError:
            pass

    DbyW = scipy.sparse.coo_matrix((vals, (rows, cols)), shape=(D, W),
                                   dtype=int)    

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
    for k, doc_d in post_d.items():
        tf[k] = sum(doc_d.values())
    return tf


def get_term_freq(content, n=1, vocab=None, replace=None,
                  clean=None, to_lower=False, remove_punc=False,
                  starts_with_id=False):
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
    starts_with_id (bool): Does each line starts with some setence ID?
    default=False.

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
    # print(data)

    for line in data:
        line = line.strip()

        if line == "":
            continue

        if starts_with_id:
            sid_len = len(line.split()[0])
            line = line[sid_len:].strip()

        if to_lower:
            line = line.lower()
        if clean is not None:
            line = clean_data(line, clean)
        if replace is not None:
            line = replace_data(line, replace)
        if remove_punc:
            line = remove_punc(line)

        tokens = line.split(" ")

        for i in range(len(tokens) - n+1):
            tok = " ".join(tokens[i: i+n]).strip()
            if(vocab is not None):
                if(tok not in vocab):
                    continue

            if tok == "":
                continue

            try:
                tf_d[tok] += 1
            except KeyError:
                tf_d[tok] = 1
    return tf_d


def trim_postings(post_d, min_freq):
    """ Trim postings based on min freq """

    new_post = {}
    
    for tok, doc_d in post_d.items():
        new_doc_d = {}
        for doc_id, freq in doc_d.items():
            if freq >= min_freq:
                new_doc_d[doc_id] = freq

        if len(new_doc_d) > 0:
            new_post[tok] = new_doc_d
        
    return new_post
                
    

class TextVector:

    def __init__(self, n=1, vocab=None, clean=None, replace=None,
                 to_lower=False, remove_punc=False, min_freq=1,
                 starts_with_id=False):
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
        min_freq (int): Ignores tokens that occur less than min_freq \n
        starts_with_id (bool): Does each line starts with some setence ID?
        default=False.
        """

        self.n = n
        self.vocab = vocab
        self.clean = clean
        self.replace = replace
        self.to_lower = to_lower
        self.remove_punc = remove_punc
        self.min_freq = min_freq
        self.starts_with_id = starts_with_id

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

        for doc_id in range(len(flist)):

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
                if line == "":
                    continue

                tokens = line.split()

                if self.starts_with_id:
                    tokens = tokens[1:]

                for i in range(len(tokens) - self.n+1):
                    tok = " ".join(tokens[i: i+self.n]).strip()

                    if len(tok) == 0:
                        continue

                    tmp_d = {}
                    try:
                        tmp_d = post_d[tok]
                        try:
                            tmp_d[doc_id] += 1
                        except KeyError:
                            tmp_d[doc_id] = 1
                    except KeyError:
                        tmp_d[doc_id] = 1

                    post_d[tok] = tmp_d

        if self.min_freq > 1:
            post_d = trim_postings(post_d, self.min_freq)
        
                    
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

        # init postings dictionary with empty entries for each token in vocab
        post_d = {}
        for v in self.vocab:
            post_d[v] = {}

        for doc_id in range(len(flist)):

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
                tokens = line.split()

                if self.starts_with_id:
                    tokens = tokens[1:]

                for i in range(len(tokens) - self.n+1):
                    tok = " ".join(tokens[i: i+self.n]).strip()

                    tmp_d = {}
                    try:
                        tmp_d = post_d[tok]
                        try:
                            tmp_d[doc_id] += 1
                        except KeyError:
                            tmp_d[doc_id] = 1

                        post_d[tok] = tmp_d

                    except KeyError:
                        pass

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

        if self.vocab is not None:
            print("Vocab is already given, use fit_ method.")
            sys.exit()

        post_d = self.get_postings(flist)

        self.vocab = sorted(post_d.keys())

        D = len(flist)

        DbyW, _ = convert_postings_to_dbyw(post_d, D, self.vocab)

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
        # print('Vocab length:', len(self.vocab))

        post_d = self.fit_postings(flist)

        D = len(flist)

        DbyW, _ = convert_postings_to_dbyw(post_d, D, self.vocab)

        return DbyW
