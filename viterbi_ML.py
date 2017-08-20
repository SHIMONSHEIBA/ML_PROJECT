import numpy as np
import time
from time import time
import math


class viterbi(object):
    """ Viterbi algorithm for 2-order HMM model"""
    def __init__(self, model, model_type, data_file, w=0):
        # model will be HMM or MEMM object, model_type in ['hmm','memm']
        self.model_type = model_type
        self.transition_mat = model.self.transition_mat
        self.emission_mat = model.emission_mat
        self.states = model.word_tag_dict.values()
        self.weight = w
        self.training_file = data_file
        self.word_tag_dict = model.word_tag_dict
        if model_type == 'memm':
            self.history_tag_feature_vector = model.history_tag_feature_vector
        else:
            self.history_tag_feature_vector = {}

    def viterbi_all_data(self):
        predict_dict = {}

        with open(self.training_file) as training:
            sequence_index = 1
            print '{}: Start viterbi on sequence index {}'.\
                format((time.asctime(time.localtime(time.time()))), sequence_index)
            for sequence in training:
                viterbi_result = self.viterbi_sequence(sequence)
                seq_word_tag_predict = []
                word_tag_list = sequence.split(',')
                for idx_tag, tag in viterbi_result.iteritems():
                    word = word_tag_list[idx_tag].split('_')
                    prediction = str(word + '_' + str(tag))
                    seq_word_tag_predict.append(prediction)
                predict_dict[sequence_index] = seq_word_tag_predict
                print '{}: prediction for sequence index {} is: {}'.format((time.asctime(time.localtime(time.time()))),
                                                                           sequence_index, seq_word_tag_predict)
                sequence_index += 1
            print '{}: prediction for all sequences{}'.format((time.asctime(time.localtime(time.time()))),
                                                              predict_dict)
        return predict_dict

    def viterbi_sequence(self, sequence):
        seq_word_tag_predict = {}

        n = len(sequence)
        num_states = len(self.states)
        word_tag_list = sequence.split(',')

        # create pi and bp numpy
        pi = np.zeros(shape=(n+1, num_states, num_states))
        bp = np.zeros(shape=(n+1, num_states, num_states), dtype='int32')

        # initialization
        pi[0, '#', '#'] = 1

        # algorithm:
        for k in range(1, n+1):
            if k == 1:  # the word in position 1
                x_k_3, x_k_2, x_k_1 = '#', '#', '#'  # words in k-3, k-2 and in k-1
            elif k == 2:  # the word in position 2
                x_k_1 = word_tag_list[k - 2].split('_')[0]  # word in k-1
                x_k_3, x_k_2 = '#', '#'  # word in k-2
            elif k == 3:  # the word in position 3
                x_k_1 = word_tag_list[k - 2].split('_')[0]  # word in k-1
                x_k_2 = word_tag_list[k - 3].split('_')[0]  # word in k-2
                x_k_3 = '#'
            else:
                x_k_1 = word_tag_list[k - 2].split('_')[0]  # word in k-1
                x_k_2 = word_tag_list[k - 3].split('_')[0]  # word in k-2
                x_k_3 = word_tag_list[k - 4].split('_')[0]  # word in k-3
            if k in range(1, n-2):
                x_k_p_3 = word_tag_list[k + 2].split('_')[0]  # word k+3
                x_k_p_2 = word_tag_list[k + 1].split('_')[0]  # word k+2
                x_k_p_1 = word_tag_list[k].split('_')[0]      # word k+1
            elif k == n-2:  # word in position n-2, no word in k+3
                x_k_p_3 = '#'  # word k+3
                x_k_p_2 = word_tag_list[k + 1].split('_')[0]  # word k+2
                x_k_p_1 = word_tag_list[k].split('_')[0]      # word k+1
            elif k == n-1:  # word in position n-1, no word in k+3 and k+2
                x_k_p_3, x_k_p_2 = '#', '#'  # word k+3 and k+2 and k+1
                x_k_p_1 = word_tag_list[k].split('_')[0]      # word k+1
            else:  # word in position n, no word in k+3 and k+2
                x_k_p_3, x_k_p_2, x_k_p_1 = '#', '#', '#'  # word k+3 and k+2 and k+1

            x_k = word_tag_list[k-1].split('_')[0]
            for u in self.possible_tags(x_k_1):
                for v in self.possible_tags(x_k):
                    calc_max_pi = 0
                    calc_argmax_pi = 0
                    for w in self.possible_tags(x_k_2):
                        w_u_pi = pi[k - 1, w, u]
                        if self.model_type == 'hmm':  # for HMM calc q*e
                            qe = self.calc_qe(v, u, w, x_k)
                            calc_pi = w_u_pi * qe

                        elif self.model_type == 'memm':  # for MEMM calc q
                            q = self.calc_q(v, u, w, x_k_3, x_k_2, x_k_1, x_k_p_3, x_k_p_2, x_k_p_1, x_k)
                            calc_pi = w_u_pi * q

                        if calc_pi > calc_max_pi:
                            calc_max_pi = calc_pi
                            calc_argmax_pi = int(w)

                    pi[k, int(u)-1, int(v)-1] = calc_max_pi  # store the max(pi)
                    bp[k, int(u)-1, int(v)-1] = calc_argmax_pi  # store the argmax(pi)

        u = np.unravel_index(pi[n].argmax(), pi[n].shape)[0]  # argmax for u in n-1
        v = np.unravel_index(pi[n].argmax(), pi[n].shape)[1]  # argmax for v in n

        seq_word_tag_predict[n - 1] = int(v)
        seq_word_tag_predict[n - 2] = int(u)

        for k in range(n-2, 0, -1):
            seq_word_tag_predict[k - 1] = bp[k+2, seq_word_tag_predict[k], seq_word_tag_predict[k+1]]

        return seq_word_tag_predict

    def possible_tags(self, word):
        if word == '#':
            return ('#')
        else:
            # get all relevant tags for word
            return self.word_tag_dict.get(word)

    def calc_qe(self, v, u, w, x_k):  # calculate q*e for HMM model
        q = self.transition_mat[v + '|' + w + ',' + u]
        e = self.emission_mat[x_k + '|' + v]
        return q * e

    def calc_q(self, v, u, w, x_k_3, x_k_2, x_k_1, x_k_p_3, x_k_p_2, x_k_p_1, x_k):  # calculate q for MEMM model

        sum_denominator = 0
        tag_exp_dict = {}

        for tag in self.word_tag_dict.get(x_k):  # all possible tags for the word x_k
            # history + tag feature vector
            current_history_tag_feature_vector = self.history_tag_feature_vector[(w, u, x_k_3, x_k_2, x_k_1,
                                                                                  x_k_p_3, x_k_p_2, x_k_p_1, x_k, tag)]
            # calculate e^(weight*f(history, tag))
            numerators = math.exp(current_history_tag_feature_vector.dot(self.weight))
            sum_denominator += numerators  # sum for the denominator
            tag_exp_dict[tag] = numerators  # save in order to get tag_exp_dict[v]

        return tag_exp_dict[v] / float(sum_denominator)
