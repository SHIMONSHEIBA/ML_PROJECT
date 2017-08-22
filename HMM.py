import time
from itertools import product
import itertools


class HMM(object):
    ''' Simple Hidden Markov Model implementation.  User provides
        transition, emission and initial probabilities in dictionaries
        mapping 2-character codes onto floating-point probabilities
        for those table entries.  States and emissions are represented
        with single characters.  Emission symbols comes from a finite.  '''

    def __init__(self, train_file, lambda1, lambda2, lambda3):
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

        self.three_tags_dict = self.declare_dict(order=3)  # number of instances of three tags together
        self.two_tags_dict = self.declare_dict(order=2)  # number of instances of two tags together
        # number of instances of each tag
        self.one_tags_dict = {'1': [0, 'A+'], '2': [0, 'C+'], '3': [0, 'G+'], '4': [0, 'T+'], '5': [0, 'A-'],
                              '6': [0, 'C-'], '7': [0, 'G-'], '8': [0, 'T-'], '#': [0, '#']}

        self.training_file = train_file
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3

        # Create the relevant dictionaries to calculate probabilities:
        self.create_dictionaries()
        # Calculate matrix:
        self.create_transition_matrix()
        self.create_emission_matrix()

    def create_dictionaries(self):
        print '{}: Start build transition and emission matrices'.format(time.asctime(time.localtime(time.time())))
        with open(self.training_file) as training:
            sequence_index = 0

            for sequence in training:
                word_tag_list = sequence.split(',')
                # define two first word_tags for some features
                first_tag = '#'
                second_tag = '#'
                for word_in_seq_index, word_tag in enumerate(word_tag_list):
                    word_tag_tuple = word_tag.split('_')
                    if '\n' in word_tag_tuple[1]:
                        word_tag_tuple[1] = word_tag_tuple[1][:1]
                        word_tag = word_tag_tuple[0] + '_' + word_tag_tuple[1]

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

                sequence_index += 1

            self.two_tags_dict['#_#'] = sequence_index
            self.one_tags_dict['#'][0] = sequence_index

        print '{}: Finish build transition and emission matrices'.format(time.asctime(time.localtime(time.time())))

        return

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

    def create_emission_matrix(self):
        for word_tag, count_word_tag in self.word_tag_dict_count.items():
            word_tag_list = word_tag.split('_')
            count_tag = self.one_tags_dict[word_tag_list[1]][0]
            self.emission_mat[word_tag_list[0] + '|' + word_tag_list[1]] = 1.0 * count_word_tag / count_tag


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
            dictionary['#_#'] = 0
        # create key for first word in three_tags_dict (i.e: #_#_word)
        else:
            for word in tag_list:
                dictionary['#_#_' + word] = 0
            # create key for second word in three_tags_dict (i.e: #_word1_word2)
            permutations_list = product('12345678', repeat=order-1)
            for permutation in permutations_list:
                key = '#_' + permutation[0] + '_' + permutation[1]
                dictionary[key] = 0

        return dictionary
