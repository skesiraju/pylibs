#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# textvector.py
# @Author : Santosh Kesiraju (kcraj2@gmail.com)
# @Date   : April 30, 2020

"""
Bag-of-words statisitcs accumulator. Generates doc-by-word sparse matrices.
"""

import sys
import string
import codecs
import argparse
from tqdm import tqdm
from collections import OrderedDict
import numpy as np
import scipy.sparse as sparse


def split_as_ngrams(words, n=3):
    """ Split a sentence into char n-grams respecting the space.

    Args:
    -----
        words (list): List of words from a sentence
        n (int): n-gram length

    Returns:
    --------
        list: list of ngrams
    """

    ngrams = []
    ngrams_extend = ngrams.extend

    for w in words:
        if len(w) >= n:
            ngrams_extend([w[i:i+n] for i, _ in enumerate(w[:1-n])])

    return ngrams


class BoW:
    """ Bag-of-words statistics accumulator """

    def __init__(self, sp_char=" ", lower=True, remove_punc=True,
                 analyzer='word', ngram=1, dtype=np.uint16):
        """ Initialize """

        self.sp_char = sp_char
        self.lower = lower
        self.remove_punc = remove_punc
        self.analyzer = analyzer
        self.ngram = ngram
        self.dtype = dtype

        self.vocab_d = OrderedDict()

        self.punc = str.maketrans('', '', string.punctuation)

    def tokenize_and_get_words(self, doc, is_tokenized):
        """ Tokenize the document and return list of words.

        Args:
        -----
            doc: string or list
            is_tokenized (bool):

        Returns:
        -------
            list of words
        """

        words = []

        if is_tokenized:
            words = doc

        else:
            if self.remove_punc:
                doc = doc.translate(self.punc).strip()
            if self.lower:
                doc = doc.lower().strip()

            words = doc.split(self.sp_char)

        return words


    def fit_transform(self, docs, is_tokenized=False, callable_fn=None):
        """ Incrementally construct doc-by-word sparse matrix.

            Args:
            -----
                docs (`iterable` or `list`): List of docs, where every
                    doc is a `string` or a `list` of tokens. If path to files,
                    use next arg aslo.
                is_tokenized (bool): is every document already tokenized and
                    is in the form of a `list`.
                callable_fn (`function`): If `docs` is not list,
                    instead if it is a list of file paths for every doc
                    then pass a 'function' that returns a doc in the
                    above format and also return doc_id in interger
                    or str format.

            Returns:
            --------
                dbyw (`scipy.sparse.csr_matrix`): doc-by-word count
                    matrix in sparse format.
        """

        indptr = [0]
        indices = []
        data = []

        docs_d = OrderedDict()

        tqdm_it = tqdm(enumerate(docs))
        for did, doc in tqdm_it:
            if callable_fn:
                doc, did = callable_fn(doc)

            docs_d[str(did)] = len(docs_d)

            wcount = OrderedDict()

            words = self.tokenize_and_get_words(doc, isinstance(doc,list))

            if self.analyzer == 'char':
                ngrams = split_as_ngrams(words, self.ngram)
                words = ngrams

            for w in words:
                if w:
                    try:
                        wix = self.vocab_d[w]
                    except KeyError:
                        self.vocab_d[w] = len(self.vocab_d)
                        wix = len(self.vocab_d) - 1

                    try:
                        wcount[wix] += 1
                    except KeyError:
                        wcount[wix] = 1

            tqdm_it.set_description(f"# words: {len(self.vocab_d)} # docs: {did}")

            indptr.append(indptr[-1] + len(wcount))
            data.extend(wcount.values())
            indices.extend(wcount.keys())

        dbyw = sparse.csr_matrix((data, indices, indptr),
                                 shape=(len(docs_d), len(self.vocab_d)),
                                 dtype=self.dtype)

        return dbyw

    def transform(self, docs, is_tokenized=False, callable_fn=None):
        """ Incrementally construct doc-by-word sparse matrix using the
            existing vocabulary.

            Args:
            -----
                docs (`iterable` or `list`): List of docs, where every
                    doc is a `string` or a `list` of tokens. If path to files,
                    use next arg aslo.
                is_tokenized (bool): is every document already tokenized?
                callable_fn (`function`): If `docs` is not list,
                    instead if it is a list of file paths for every doc
                    then pass a 'function' that returns a doc in the
                    above format and also return doc_id in interger
                    or str format.

            Returns:
            --------
                dbyw (`scipy.sparse.csr_matrix`): doc-by-word count
                    matrix in sparse format.
        """

        print("Vocabulary:", len(self.vocab_d))
        indptr = [0]
        indices = []
        data = []

        docs_d = OrderedDict()

        tqdm_it = tqdm(enumerate(docs))
        for did, doc in tqdm_it:
            if callable_fn:
                doc, did = callable_fn(doc)

            docs_d[str(did)] = len(docs_d)

            wcount = OrderedDict()
            words = self.tokenize_and_get_words(doc, isinstance(doc, list))

            if self.analyzer == 'char':
                ngrams = split_as_ngrams(words, self.ngram)
                words = ngrams

            for w in words:
                try:
                    wix = self.vocab_d[w]
                except KeyError:
                    continue

                try:
                    wcount[wix] += 1
                except KeyError:
                    wcount[wix] = 1

            tqdm_it.set_description(f"# words: {len(self.vocab_d)} # docs: {did}")

            indptr.append(indptr[-1] + len(wcount))
            data.extend(wcount.values())
            indices.extend(wcount.keys())

        dbyw = sparse.csr_matrix((data, indices, indptr),
                                 shape=(len(docs_d), len(self.vocab_d)),
                                 dtype=self.dtype)

        return dbyw

    def apply_min_wfreq(self, dbyw, min_wfreq):
        """ Apply minimum word frequency constraint, update the vocabulary,
        and return updated counts matrix.

        Args:
        ----
            dbyw (`scipy.sparse`): Doc-by-word matrix
            min_wfreq (`int`): Minimum word freq. to prune the vocab.

        Returns:
        -------
            dbyw (`scipy.sparse`): Doc-by-word matrix
        """

        if dbyw.shape[1] != len(self.vocab_d):
            print("Word count in given matrix and the vocabulary do not match.")
            print("DbyW:", dbyw.shape, "Vocab:", len(self.vocab_d))
            sys.exit()

        marginal_wcount = dbyw.sum(axis=0).T

        ixs = np.where(marginal_wcount >= min_wfreq)[0]
        if ixs.any():
            new_dbyw = dbyw[:, ixs]
            new_vocab = []
            set_ixs = set(list(ixs))
            for i, w in enumerate(self.vocab_d):
                if i in set_ixs:
                    new_vocab.append(w)
            self.set_vocab(new_vocab)

        else:
            new_dbyw = dbyw
            # new_vocab = vocab_npy

        return new_dbyw #, # list(new_vocab)

    def apply_min_dfreq(self, dbyw, min_dfreq):
        """ Apply minimum document frequency constraint, update the vocabulary,
        and return updated counts matrix.

        Args:
        ----
            dbyw (`scipy.sparse`): Doc-by-word matrix
            min_dfreq (`int`): Minimum doc. freq. to prune the vocab.

        Returns:
        -------
            dbyw (`scipy.sparse`): Doc-by-word matrix
        """

        if dbyw.shape[1] != len(self.vocab_d):
            print("Word count in given matrix and the vocabulary do not match.")
            print("DbyW:", dbyw.shape, "Vocab:", len(self.vocab_d))
            sys.exit()

        marginal_dcount = dbyw.sum(axis=1).T

        ixs = np.where(marginal_dcount >= min_dfreq)[0]
        if ixs.any():
            new_dbyw = dbyw[:, ixs]
        else:
            new_dbyw = dbyw

        return new_dbyw

    def reset_vocab(self):
        """ Empty vocabulary """
        self.vocab_d = OrderedDict()

    def set_vocab(self, vocab):
        """ Set vocabulary with the given list

        Args:
        -----
            vocab (list): List of words
        """
        vocab = set(vocab)
        self.vocab_d = OrderedDict()
        for i, word in enumerate(vocab):
            self.vocab_d[word] = i

    def append_vocab(self, vocab):
        """ Append vocabulary with the given list

        Args:
        -----
            vocab (list): List of words
        """
        new_vocab = set(vocab) - set(self.vocab_d.keys())
        for word in new_vocab:
            self.vocab_d[word] = len(self.vocab_d)


def test():
    """ test method """

    lines = []
    with codecs.open(ARGS.input_file, 'r', 'utf-8') as fpr:
        lines = [line.strip() for line in fpr if line.strip()]
    print("Number of lines (docs):", len(lines))

    bow = BoW(sp_char=ARGS.sep, lower=ARGS.tolower, remove_punc=ARGS.nopunc)
    stats = bow.fit_transform(lines)
    print("Stats:", stats.shape)
    print("Vocabulary size:", len(bow.vocad_d))


if __name__ == "__main__":

    PARSER = argparse.ArgumentParser(description=__doc__)
    PARSER.add_argument("input_file", help="path to input file")
    PARSER.add_argument("sep", default=" ", help="token sepearator")
    PARSER.add_argument("--tolower", action="store_true",
                        help="lowercase the text")
    PARSER.add_argument("--nopunc", action="store_true",
                        help="remove punctuation")
    ARGS = PARSER.parse_args()

    test()
