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

    amino_mapping = {'Phe': ['TTT', 'TTC'], 'Leu': ['TTA', 'TTG', 'CTT', 'CTC', 'CTA', 'CTG'], 'Met': ['ATG'],
                     'Ile': ['ATT', 'ATC', 'ATA'], 'Val': ['GTT', 'GTC', 'GTA', 'GTG'] , 'TCT': 'Ser', 'TCC': 'Ser',
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

    def __init__(self, chrome_list):

        self.tags_dict = {'1': [0, 'A+'], '2': [0, 'C+'], '3': [0, 'G+'], '4': [0, 'T+'], '5': [0, 'A-'], '6': [0, 'C-'],
                          '7': [0, 'G-'], '8': [0, 'T-']}

        self.words_dict = {'A': 0, 'T': 0, 'C': 0, 'G': 0}

        self.chrome_list = chrome_list
        # self.data = pd.read_excel(self.training_file)

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
        sequence_index = 0
        print '{}: Start build transition and emission matrices'.format(time.asctime(time.localtime(time.time())))
        for chrome in self.chrome_list:
            training_file = 'C:\\gitprojects\\ML_PROJECT\\labels150\\chr' + chrome + '_data.csv'
            with open(training_file) as training:
                for sequence in training:
                    sequence_index += 1
                    word_tag_list = sequence.split(',')
                    # Calculate feature 1: number of occurrences per base
                    number_of_A = word_tag_list.count('1')
                    number_of_C = word_tag_list.count('2')
                    number_of_G = word_tag_list.count('3')
                    number_of_T = word_tag_list.count('4')
                    # Calculate feature 2: count amino + stop + start codons:
                    for index, amino in enumerate(self.amino_mapping.values())


        start_time = time.time()
        print('starting building features from train')

        return


