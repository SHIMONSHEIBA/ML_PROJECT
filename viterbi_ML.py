import numpy as np
import time
import math
import itertools


class viterbi(object):
    """ Viterbi algorithm for 2-order HMM model"""
    def __init__(self, model, model_type, data_file, is_log, use_stop_prob, w=0):
        # model will be HMM or MEMM object, model_type in ['hmm','memm']
        self.model_type = model_type
        if model_type == 'hmm':
            self.transition_mat = model.transition_mat
            self.emission_mat = model.emission_mat
        else:
            self.transition_mat = {}
            self.emission_mat = {}
        self.states = list(itertools.chain.from_iterable(model.word_tag_dict.values()))
        self.weight = w
        self.training_file = data_file
        self.word_tag_dict = model.word_tag_dict
        self.use_stop_prob = use_stop_prob
        if model_type == 'memm':
            self.history_tag_feature_vector = model.history_tag_feature_vector
        else:
            self.history_tag_feature_vector = {}
        if is_log:
            self.transition_mat = {key: math.log10(value) for key, value in self.transition_mat.items()}

    def viterbi_all_data(self):
        predict_dict = {}

        with open(self.training_file) as training:
            sequence_index = 0
            for sequence in training:
                print('{}: Start viterbi on sequence index {}: \n {}'. \
                    format(time.asctime(time.localtime(time.time())), sequence_index, sequence))
                viterbi_result = self.viterbi_sequence(sequence)
                seq_word_tag_predict = []
                word_tag_list = sequence.split(',')
                if '\n' in word_tag_list:
                    word_tag_list.remove('\n')
                word_tag_list[len(word_tag_list) - 1] = word_tag_list[len(word_tag_list) - 1][:1]
                for idx_tag, tag in viterbi_result.iteritems():
                    if tag == 0 or tag == '0' or tag == -1 or tag == '-1':
                        print('Error: tag is: {}'.format(tag))
                    word = word_tag_list[idx_tag].split('_')[0]
                    prediction = str(word + '_' + str(tag))
                    seq_word_tag_predict.append(prediction)
                predict_dict[sequence_index] = seq_word_tag_predict
                # print '{}: prediction for sequence index {} is: {}'.format((time.asctime(time.localtime(time.time()))),
                #                                                            sequence_index, seq_word_tag_predict)
                sequence_index += 1
            # print '{}: prediction for all sequences{}'.format((time.asctime(time.localtime(time.time()))),
            #                                                   predict_dict)
        return predict_dict

    def viterbi_sequence(self, sequence):
        seq_word_tag_predict = {}

        n = len(sequence.split(','))
        num_states = len(self.states)
        word_tag_list = sequence.split(',')

        # create pi and bp numpy
        pi = np.ones(shape=(n+1, num_states, num_states), dtype=float) * float("-inf")
        bp = np.ones(shape=(n+1, num_states, num_states), dtype='int32') * -1

        # initialization: # will be 0 in the numpy
        pi[0, 0, 0] = 1.0

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
            if k == n:
                reut = 1
            x_k = word_tag_list[k-1].split('_')[0]
            for u in self.possible_tags(x_k_1):
                for v in self.possible_tags(x_k):
                    calc_max_pi = float("-inf")
                    calc_argmax_pi = -1
                    for w in self.possible_tags(x_k_2):
                        w_u_pi = pi[k - 1, int(w), int(u)]
                        if self.model_type == 'hmm':  # for HMM calc q*e
                            qe = self.calc_qe(v, u, w, x_k)
                            calc_pi = w_u_pi * qe

                        elif self.model_type == 'memm':  # for MEMM calc q
                            if v == '0':
                                memm_v = '#'
                            else:
                                memm_v = v
                            if u == '0':
                                memm_u = '#'
                            else:
                                memm_u = u
                            if w == '0':
                                memm_w = '#'
                            else:
                                memm_w = w
                            print('memm_v:{}, memm_u:{}, memm_w:{}, x_k_3:{}, x_k_2:{}, x_k_1:{}, x_k_p_3:{},'
                                  'x_k_p_2:{}, x_k_p_1:{}, x_k:{}'.format(memm_v, memm_u, memm_w, x_k_3, x_k_2, x_k_1,
                                                                          x_k_p_3, x_k_p_2, x_k_p_1, x_k))
                            if x_k_p_3 == '' or x_k_p_2 == '' or x_k_p_1 == '':
                                reut = 1
                            q = self.calc_q(memm_v, memm_u, memm_w, x_k_3, x_k_2, x_k_1, x_k_p_3, x_k_p_2, x_k_p_1, x_k)
                            calc_pi = w_u_pi * q

                        else:
                            print('Error: model_type is not in [memm, hmm]')

                        if calc_pi > calc_max_pi:
                            calc_max_pi = calc_pi
                            calc_argmax_pi = int(w)

                    if calc_argmax_pi == 0:
                        reut = 1
                    # print int(u), int(v)
                    pi[k, int(u), int(v)] = calc_max_pi  # store the max(pi)
                    bp[k, int(u), int(v)] = calc_argmax_pi  # store the argmax(pi)

        # print pi[n]
        # print bp[n]
        if self.model_type == 'hmm' and self.use_stop_prob:
            stop_p_array = np.ones(shape=(num_states, num_states), dtype=float) * float("-inf")
            x_n_1 = word_tag_list[n - 2].split('_')[0]
            x_n = word_tag_list[n - 1].split('_')[0]
            for u in self.possible_tags(x_n_1):
                for v in self.possible_tags(x_n):
                    u_v_pi = pi[n, int(u), int(v)]
                    transition_stop = self.transition_mat['#' + '|' + u + ',' + v]
                    stop_p = u_v_pi * transition_stop
                    stop_p_array[int(u), int(v)] = stop_p

            u = np.unravel_index(stop_p_array.argmax(), stop_p_array.shape)[0]  # argmax for u in n-1
            v = np.unravel_index(stop_p_array.argmax(), stop_p_array.shape)[1]  # argmax for v in n

            if v == -1 or u == -1:
                print('Error: v or u value is -1')

            seq_word_tag_predict[n - 1] = v
            seq_word_tag_predict[n - 2] = u

            for k in range(n - 2, 0, -1):
                seq_word_tag_predict[k - 1] = bp[k + 2, seq_word_tag_predict[k], seq_word_tag_predict[k + 1]]

            return seq_word_tag_predict

        elif self.model_type == 'memm' or self.use_stop_prob is False:
            u = np.unravel_index(pi[n].argmax(), pi[n].shape)[0]  # argmax for u in n-1
            v = np.unravel_index(pi[n].argmax(), pi[n].shape)[1]  # argmax for v in n

            if v == -1 or u == -1:
                print('Error: v or u value is -1')

            seq_word_tag_predict[n - 1] = v
            seq_word_tag_predict[n - 2] = u

            for k in range(n-2, 0, -1):
                seq_word_tag_predict[k - 1] = bp[k+2, seq_word_tag_predict[k], seq_word_tag_predict[k+1]]

            return seq_word_tag_predict

        else:
            print('Error: model_type is not in [memm, hmm]')

    def possible_tags(self, word):
        if word == '#':
            return ['0']
        else:
            # get all relevant tags for word
            return self.word_tag_dict.get(word)

    def calc_qe(self, v, u, w, x_k):  # calculate q*e for HMM model
        tags_for_matrix = [v, u, w]
        for tag_index, tag in enumerate(tags_for_matrix):
            if tag == 0:
                tags_for_matrix[tag_index] = '#'

        q = self.transition_mat[tags_for_matrix[0] + '|' + tags_for_matrix[2] + ',' + tags_for_matrix[1]]
        e = self.emission_mat[x_k + '|' + tags_for_matrix[0]]
        return q * e

    def calc_q(self, v, u, w, x_k_3, x_k_2, x_k_1, x_k_p_3, x_k_p_2, x_k_p_1, x_k):  # calculate q for MEMM model

        sum_denominator = 0
        tag_exp_dict = {}

        for tag in self.word_tag_dict.get(x_k):  # all possible tags for the word x_k
            # history + tag feature vector
            current_history_tag_feature_vector = self.history_tag_feature_vector[(w, u, x_k_3, x_k_2, x_k_1,
                                                                                  x_k_p_1, x_k_p_2, x_k_p_3, x_k), tag]
            # calculate e^(weight*f(history, tag))
            numerators = math.exp(current_history_tag_feature_vector.dot(self.weight))
            sum_denominator += numerators  # sum for the denominator
            tag_exp_dict[tag] = numerators  # save in order to get tag_exp_dict[v]

        return tag_exp_dict[v] / float(sum_denominator)
