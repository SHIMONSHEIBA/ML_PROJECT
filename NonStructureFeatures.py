import time
from itertools import product
import numpy as np
import csv
import itertools
import pandas as pd


class NonStructureFeatures:
    """ Base class of modeling SVM logic on the data"""

    # shared among all instances of the class'
    # amino_mapping = {'Phe': ['TTT', 'TTC'], 'Leu': ['TTA', 'TTG', 'CTT', 'CTC', 'CTA', 'CTG'], 'Met': ['ATG'],
    #                  'Ile': ['ATT', 'ATC', 'ATA'], 'Val': ['GTT', 'GTC', 'GTA', 'GTG'],
    #                  'Ser': ['TCT', 'TCC', 'TCA', 'TCG', 'AGT', 'AGC'], 'Pro': ['CCT', 'CCC', 'CCA', 'CCG'],
    #                  'Thr': ['ACT', 'ACC', 'ACA', 'ACG'], 'Ala': ['GCT', 'GCC', 'GCA', 'GCG'],
    #                  'Tyr': ['TAT', 'TAC'], 'stop': ['TAA', 'TAG', 'TGA'], 'His': ['CAT', 'CAC'], 'Gin': ['CAA', 'CAG'],
    #                  'Asn': ['AAT', 'AAC'], 'Lys': ['AAA', 'AAG'], 'Asp': ['GAT', 'GAC'], 'Glu': ['GAA', 'GAG'],
    #                  'Cys': ['TGT', 'TGC'], 'Trp': ['TGG'], 'Arg': ['CGT', 'CGC', 'CGA', 'CGG', 'AGA', 'AGG'],
    #                  'Gly': ['GGT', 'GGC', 'GGA', 'GGG']}

    amino_mapping = {'Phe': ['444', '442'], 'Leu': ['441', '443', '244', '242', '241', '243'], 'Met': ['143'],
                     'Ile': ['144', '142', '141'], 'Val': ['344', '342', '341', '343'],
                     'Ser': ['424', '422', '421', '423', '134', '132'], 'Pro': ['224', '222', '221', '223'],
                     'Thr': ['124', '122', '121', '123'], 'Ala': ['324', '322', '321', '323'],
                     'Tyr': ['414', '412'], 'stop': ['411', '413', '431'], 'His': ['214', '212'], 'Gin': ['211', '213'],
                     'Asn': ['114', '112'], 'Lys': ['111', '113'], 'Asp': ['314', '312'], 'Glu': ['311', '313'],
                     'Cys': ['434', '432'], 'Trp': ['433'], 'Arg': ['234', '232', '231', '233', '131', '133'],
                     'Gly': ['334', '332', '331', '333']}

    three_words = list(itertools.chain.from_iterable(amino_mapping.values()))
    three_words.sort()

    def __init__(self, chrome_train_list, chrome_test_list):

        # chore to use as training data
        self.chrome_train_list = chrome_train_list
        self.chrome_test_list = chrome_test_list

        # build the type of features
        self.all_train_samples_features = self.build_features_from_train(self.chrome_train_list)
        self.all_test_samples_features = self.build_features_from_train(self.chrome_test_list)

    def build_features_from_train(self, chrome_list):
        # In this function we are counting amount of instances from
        # each feature for statistics and feature importance
        sequence_index = 0
        print('{}: Starting building features from train'.format(time.asctime(time.localtime(time.time()))))
        for chrome in chrome_list:
            training_file = 'C:\\gitprojects\\ML_PROJECT\\data150\\chr' + chrome + '_data.csv'
            with open(training_file) as training:
                for sequence in training:
                    sequence_index += 1
                    vector_index = []
                    word_tag_list = sequence.split(',')
                    # Calculate feature 1: number of occurrences per base
                    vector_index.append(word_tag_list.count('1'))
                    vector_index.append(word_tag_list.count('2'))
                    vector_index.append(word_tag_list.count('3'))
                    vector_index.append(word_tag_list.count('4'))
                    # Calculate feature 2: count amino + stop + start codons:
                    for amino_index, amino in enumerate(self.amino_mapping.keys()):
                        count_amino = 0
                        for three_index, three_word in enumerate(self.amino_mapping.get(amino)):
                            count_amino += word_tag_list.count(three_word)
                        vector_index.append(count_amino)
                    # Calculate feature 3: count each three_words:
                    for three_word in self.three_words:
                        vector_index.append(word_tag_list.count(three_word))

                    featuresDF = pd.Series(vector_index)

                    if sequence_index == 1:
                        all_samples_features = featuresDF
                    else:
                        all_samples_features = pd.concat([featuresDF, all_samples_features], axis=1)

        return all_samples_features


