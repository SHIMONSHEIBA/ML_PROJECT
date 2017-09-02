import time
from itertools import product
import itertools
import csv
import numpy as np
import logging


class HMM(object):
    ''' Simple Hidden Markov Model implementation.  User provides
        transition, emission and initial probabilities in dictionaries
        mapping 2-character codes onto floating-point probabilities
        for those table entries.  States and emissions are represented
        with single characters.  Emission symbols comes from a finite.  '''

    def __init__(self, chrome_list, lambda1, lambda2, lambda3, is_smooth):
        ''' Initialize the HMM given transition, emission and initial
            probability tables. '''
        # Declare class variables:
        # transition matrix for 2-order HMM model: the probability to see a tag after 2 tags (e.g: P(3|12))
        self.transition_mat = {}
        self.emission_mat = {}  # emission matrix: the probability to see base with tag (e.g: P(A|1))
        # self.initial_mat = {}  # initial matrix:
        # number of instances of a word and a tag together
        self.word_tag_dict_count =\
            {'A_1': 0, 'A_2': 0, 'A_3': 0, 'A_4': 0, 'A_5': 0, 'A_6': 0, 'A_7': 0, 'A_8': 0,
             'C_1': 0, 'C_2': 0, 'C_3': 0, 'C_4': 0, 'C_5': 0, 'C_6': 0, 'C_7': 0, 'C_8': 0,
             'G_1': 0, 'G_2': 0, 'G_3': 0, 'G_4': 0, 'G_5': 0, 'G_6': 0, 'G_7': 0, 'G_8': 0,
             'T_1': 0, 'T_2': 0, 'T_3': 0, 'T_4': 0, 'T_5': 0, 'T_6': 0, 'T_7': 0, 'T_8': 0}

        self.word_tag_dict = {'A': ['1', '5'], 'C': ['2', '6'], 'G': ['3', '7'], 'T': ['4', '8'], '#': ['#']}
        self.states = list(itertools.chain.from_iterable(self.word_tag_dict.values()))
        num_states = len(self.states)
        num_symbols = len(self.word_tag_dict.keys())

        # transition matrix for 1-order HMM model: the probability to see a tag after 2 tags (e.g: P(3|12))
        self.transition_mat_array = np.zeros(shape=(num_states, num_states, num_states))
        self.emission_mat_array = np.zeros(shape=(num_states, num_symbols))  # emission matrix: the probability to see base with tag (e.g: P(A|1))

        self.three_tags_dict = self.declare_dict(order=3)  # number of instances of three tags together
        self.two_tags_dict = self.declare_dict(order=2)  # number of instances of two tags together
        # number of instances of each tag
        self.one_tags_dict = {'1': [0, 'A+', 1], '2': [0, 'C+', 2], '3': [0, 'G+', 3], '4': [0, 'T+', 4],
                              '5': [0, 'A-', 5], '6': [0, 'C-', 6], '7': [0, 'G-', 7], '8': [0, 'T-', 8],
                              '#': [0, '#', 0]}
        self.word_index = {'#': 0, 'A': 1, 'C': 2, 'G': 3, 'T': 4}

        self.chrome_list = chrome_list
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3

        # Create the relevant dictionaries to calculate probabilities:
        self.create_dictionaries()
        # print dictionary for checking logic
        self.write_dictionary('three', self.three_tags_dict)
        self.write_dictionary('two', self.two_tags_dict)
        self.write_dictionary('one', self.one_tags_dict)
        # Calculate matrix:
        if is_smooth:
            self.create_transition_matrix()
            # print self.transition_mat_array
            print np.sum(self.transition_mat_array, axis=2)
            logging.info('{}: using smoothing: {}'.format(time.asctime(time.localtime(time.time())), is_smooth))
            logging.info('{}: sum of rows in transition matrix is: \n {}'.
                         format(time.asctime(time.localtime(time.time())),
                                np.sum(self.transition_mat_array, axis=2)))
        else:
            self.create_transition_matrix_no_smooth()
            # print self.transition_mat_array
            print np.sum(self.transition_mat_array, axis=2)
            logging.info('{}: using smoothing: {}'.format(time.asctime(time.localtime(time.time())), is_smooth))
            logging.info('{}: sum of rows in transition matrix is: \n {}'.
                         format(time.asctime(time.localtime(time.time())),
                                np.sum(self.transition_mat_array, axis=2)))
        self.create_emission_matrix()
        print self.emission_mat_array
        with open('transition_mat.csv', 'wb') as csv_file:
            writer = csv.writer(csv_file)
            for key, value in self.transition_mat.items():
                writer.writerow([key, value])

    def create_dictionaries(self):
        sequence_index = 0
        print '{}: Start build transition and emission matrices'.format(time.asctime(time.localtime(time.time())))
        for chrome in self.chrome_list:
            print '{}: Start train on chrome: {}'.format((time.asctime(time.localtime(time.time()))), chrome)
            training_file = 'C:\\gitprojects\\ML_PROJECT\\labels150\\chr' + chrome + '_label.csv'
            with open(training_file, 'r') as training:
                for sequence in training:
                    sequence_index += 1
                    word_tag_list = sequence.split(',')
                    # handel , in the end of the sequence:
                    # if '\n' in word_tag_list[len(word_tag_list) - 1]:
                    #     word_tag_list[len(word_tag_list) - 1] = word_tag_list[len(word_tag_list) - 1].replace('\n', '')
                    while ' ' in word_tag_list:
                        word_tag_list.remove(' ')
                    while '' in word_tag_list:
                        word_tag_list.remove('')
                    while '\n' in word_tag_list:
                        word_tag_list.remove('\n')
                    # define two first word_tags for some features
                    first_tag = '#'
                    second_tag = '#'
                    previous_tag = '0'
                    for word_in_seq_index, word_tag in enumerate(word_tag_list):
                        if word_tag not in ('A_1', 'A_5', 'C_2', 'C_6', 'G_3', 'G_7', 'T_4', 'T_8',
                                            'A_1\n', 'A_5\n', 'C_2\n', 'C_6\n', 'G_3\n', 'G_7\n', 'T_4\n', 'T_8\n'):
                            print('Error, word_tag is:{}').format(word_tag)
                        word_tag_tuple = word_tag.split('_')
                        if len(word_tag_tuple) == 1:
                            reut = 1
                        # if word_in_seq_index == len(word_tag_list)-1 and '\n' not in word_tag_tuple[1]:
                        #     print('Error: n not in last tuple')
                        if '\n' in word_tag_tuple[1]:  # end of sequence
                            current_tag = word_tag_tuple[1][:1]
                            word_tag = word_tag_tuple[0] + '_' + current_tag
                            two_tags_end = current_tag + '_#'
                            self.two_tags_dict[two_tags_end] += 1
                            three_tags_end = current_tag + '_#' + '_#'
                            self.three_tags_dict[three_tags_end] += 1
                            three_tags_end_with_two_tags = previous_tag + '_' + current_tag + '_#'
                            self.three_tags_dict[three_tags_end_with_two_tags] += 1
                        elif word_in_seq_index == len(word_tag_list)-1 and '\n' not in word_tag_tuple[1]:
                            # print('Error: n not in last tuple')
                            current_tag = word_tag_tuple[1]
                            word_tag = word_tag_tuple[0] + '_' + word_tag_tuple[1]
                            two_tags_end = current_tag + '_#'
                            self.two_tags_dict[two_tags_end] += 1
                            three_tags_end = current_tag + '_#' + '_#'
                            self.three_tags_dict[three_tags_end] += 1
                            three_tags_end_with_two_tags = previous_tag + '_' + current_tag + '_#'
                            self.three_tags_dict[three_tags_end_with_two_tags] += 1

                        else:
                            current_tag = word_tag_tuple[1]

                        # count number of instances for each tag in train set
                        self.one_tags_dict[current_tag][0] += 1

                        # count number of instances of a word and a tag together
                        self.word_tag_dict_count[word_tag] += 1

                        # count number of instances of two tags together
                        two_tags = second_tag + '_' + current_tag
                        self.two_tags_dict[two_tags] += 1

                        # count number of instances of three tags together
                        three_tags = first_tag + '_' + second_tag + '_' + current_tag
                        self.three_tags_dict[three_tags] += 1

                        # update tags
                        first_tag = second_tag
                        second_tag = current_tag
                        previous_tag = current_tag

            self.two_tags_dict['#_#'] = sequence_index
            self.one_tags_dict['#'][0] = sequence_index

        print '{}: Finish build transition and emission matrices'.format(time.asctime(time.localtime(time.time())))

        return

    def write_dictionary(self, order, dict):
        dict_name = order + '_tags_dict'
        with open(dict_name + '.csv', 'wb') as csv_file:
            writer = csv.writer(csv_file)
            for key, value in dict.items():
                writer.writerow([key, value])

    def create_transition_matrix_no_smooth(self):
        for three_tags, count_all_three_tags in self.three_tags_dict.items():
            tags = three_tags.split('_')
            # calculate all counts to calculate the probability
            count_first_two_tags = self.two_tags_dict[tags[0] + '_' + tags[1]]
            # calculate probability P(word3|word1,word2)
            if count_first_two_tags != 0:
                first_composed = float(count_all_three_tags) / float(count_first_two_tags)
            else:
                first_composed = 0.0
            self.transition_mat[tags[2] + '|' + tags[0] + ',' + tags[1]] = first_composed
            self.transition_mat_array[self.one_tags_dict[tags[0]][2], self.one_tags_dict[tags[1]][2],
                                      self.one_tags_dict[tags[2]][2]] = first_composed

    def create_transition_matrix(self):
        count_all_tags_instances = sum(self.three_tags_dict.values())
        for three_tags, count_all_three_tags in self.three_tags_dict.items():
            tags = three_tags.split('_')
            # calculate all counts to calculate the probability
            count_first_two_tags = self.two_tags_dict[tags[0] + '_' + tags[1]]
            count_last_two_tags = self.two_tags_dict[tags[1] + '_' + tags[2]]
            count_second_tag = self.one_tags_dict[tags[1]][0]
            count_third_tag = self.one_tags_dict[tags[2]][0]
            # calculate probability P(word3|word1,word2)
            if count_first_two_tags != 0:
                first_composed = float(count_all_three_tags) / float(count_first_two_tags)
            else:
                first_composed = 0.0
            if count_second_tag != 0:
                second_composed = float(count_last_two_tags) / float(count_second_tag)
            else:
                second_composed = 0.0
            third_composed = float(count_third_tag) / float(count_all_tags_instances)
            self.transition_mat[tags[2] + '|' + tags[0] + ',' + tags[1]] = \
                (self.lambda1 * first_composed) + (self.lambda2 * second_composed) + (self.lambda3 * third_composed)
            self.transition_mat_array[self.one_tags_dict[tags[0]][2], self.one_tags_dict[tags[1]][2],
                                      self.one_tags_dict[tags[2]][2]] = (self.lambda1 * first_composed) +\
                                                                        (self.lambda2 * second_composed) +\
                                                                        (self.lambda3 * third_composed)

    def create_emission_matrix(self):
        for word_tag, count_word_tag in self.word_tag_dict_count.items():
            word_tag_list = word_tag.split('_')
            count_tag = self.one_tags_dict[word_tag_list[1]][0]
            self.emission_mat[word_tag_list[0] + '|' + word_tag_list[1]] = float(count_word_tag) / float(count_tag)
            if float(count_word_tag) / float(count_tag) not in (0.0, 1.0):
                print('Error: emission probability not 0 or 1')
            self.emission_mat_array[self.one_tags_dict[word_tag_list[1]][2], self.word_index[word_tag_list[0]]] =\
                float(count_word_tag) / float(count_tag)
        self.emission_mat_array[0, 0] = 1.0


    def declare_dict(self, order):
        permutations_list = product('12345678', repeat=order)
        dictionary = {}
        # create key for all permutation of size order
        for permutation in permutations_list:
            if order == 2:
                key = permutation[0] + '_' + permutation[1]
            else:
                key = permutation[0] + '_' + permutation[1] + '_' + permutation[2]
            dictionary[key] = 0
        # create key for first word in two_tags_dict (i.e: #_word)
        tag_list = list(itertools.chain.from_iterable(self.word_tag_dict.values()))
        if order == 2:
            for word in tag_list:
                dictionary['#_' + word] = 0
                dictionary[word + '_#'] = 0
            dictionary['#_#'] = 0
        # create key for first word in three_tags_dict (i.e: #_#_word)
        else:
            for word in tag_list:
                dictionary['#_#_' + word] = 0
                dictionary[word + '_#_#'] = 0
            # create key for second word in three_tags_dict (i.e: #_word1_word2)
            permutations_list = product('12345678', repeat=order-1)
            for permutation in permutations_list:
                key1 = '#_' + permutation[0] + '_' + permutation[1]
                key2 = permutation[0] + '_' + permutation[1] + '_#'
                dictionary[key1] = 0
                dictionary[key2] = 0

        return dictionary
