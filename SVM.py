import time
from itertools import product
import numpy as np
from scipy.sparse import csr_matrix
import csv
import pandas as pd


class SVM:
    """ Base class of modeling SVM logic on the data"""

    # shared among all instances of the class'
    amino_mapping = {'TTT': 'Phe', 'TTC': 'Phe', 'TTA': 'Leu', 'TTG': 'Leu', 'CTT': 'Leu', 'CTC': 'Leu',
                     'CTA': 'Leu', 'CTG': 'Leu', 'ATT': 'Ile', 'ATC': 'Ile', 'ATA': 'Ile', 'ATG': 'Met',
                     'GTT': 'Val', 'GTC': 'Val', 'GTA': 'Val', 'GTG': 'Val', 'TCT': 'Ser', 'TCC': 'Ser',
                     'TCA': 'Ser', 'TCG': 'Ser', 'CCT': 'Pro', 'CCC': 'Pro', 'CCA': 'Pro', 'CCG': 'Pro',
                     'ACT': 'Thr', 'ACC': 'Thr', 'ACA': 'Thr', 'ACG': 'Thr', 'GCT': 'Ala', 'GCC': 'Ala',
                     'GCA': 'Ala', 'GCG': 'Ala', 'TAT': 'Tyr', 'TAC': 'Tyr', 'TAA': 'stop', 'TAG': 'stop',
                     'CAT': 'His', 'CAC': 'His', 'CAA': 'Gin', 'CAG': 'Gin', 'AAT': 'Asn', 'AAC': 'Asn',
                     'AAA': 'Lys', 'AAG': 'Lys', 'GAT': 'Asp', 'GAC': 'Asp', 'GAA': 'Glu', 'GAG': 'Glu',
                     'TGT': 'Cys', 'TGC': 'Cys', 'TGA': 'stop', 'TGG': 'Trp', 'CGT': 'Arg', 'CGC': 'Arg',
                     'CGA': 'Arg', 'CGG': 'Arg', 'AGT': 'Ser', 'AGC': 'Ser', 'AGA': 'Arg', 'AGG': 'Arg',
                     'GGT': 'Gly', 'GGC': 'Gly', 'GGA': 'Gly', 'GGG': 'Gly'}

    stop_keys = ['TGA', 'TAA', 'TAG']

    start_keys = ['ATG']

    word_tag_dict = {'A': ['1', '5'], 'C': ['2', '6'], 'G': ['3', '7'], 'T': ['4', '8'], '#':['#']}

    def __init__(self, training_file):

        self.tags_dict = {'1': [0, 'A+'], '2': [0, 'C+'], '3': [0, 'G+'], '4': [0, 'T+'], '5': [0, 'A-'], '6': [0, 'C-'],
                          '7': [0, 'G-'], '8': [0, 'T-']}

        self.words_dict = {'A': 0, 'T': 0, 'C': 0, 'G': 0}

        self.training_file = training_file
        self.data = pd.read_excel(self.training_file)

        # the dictionary that will hold all indexes for all the instances of the features
        self.features_vector = {}

        # mainly for debugging and statistics
        self.features_vector_mapping = {}

        # final vector for Viterbi and GA
        self.history_tag_feature_vector = {}

        # build the type of features
        self.build_features_from_train()

        # build the features_vector
        self.build_features_vector()

    def build_features_from_train(self):
        # In this function we are counting amount of instances from
        # each feature for statistics and feature importance

        start_time = time.time()
        print('starting building features from train')

        return

    def
