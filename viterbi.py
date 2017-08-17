import numpy as np
import math
from MEMM import MEMM
from gradient import Gradient

class Viterbi(object):

    def __init__(self, memm_obj, type, v):

        # MEMM Object members
        self.memm_obj = memm_obj
        self.train_f_v = memm_obj.train_f_v
        self.train_f_v_prime = memm_obj.train_f_v_prime
        self.tag_dict = memm_obj.tag_dict
        self.word_dictionary_count = self.memm_obj.word_dictionary_count
        self.common_tag_list = self.memm_obj.common_tag_list

        self.tag_ordered_dict = {}
        self.common_tag_dict = {}
        self.v = v

        self.sentences_storage = {}
        self.f_v = {}
        self.f_v_prime = {}

        if type == 'training':
            self.sentences_storage = memm_obj.sentences_storage
            self.f_v = memm_obj.train_f_v
            self.f_v_prime = memm_obj.train_f_v_prime

        elif type == 'test':
            self.sentences_storage = memm_obj.test_sentences_storage
            self.f_v = memm_obj.test_f_v
            self.f_v_prime = memm_obj.test_f_v_prime

        elif type == 'competition':
            self.sentences_storage = memm_obj.comp_sentences_storage

    def run_viterbi_algorithm_on_sentence(self, sentence, sentence_index):

        output_tag_dict = {}

        # pi(k,u,v) - max prob of tag sequence ending in tags u,v at position k

        n = len(sentence)
        T = len(self.tag_dict)

        index = 0
        for tag in self.tag_dict.keys():
            self.tag_ordered_dict[tag] = index
            index += 1

        for tag in self.common_tag_list:
            self.common_tag_dict[tag] = self.tag_ordered_dict[tag]

        self.flip_tag_ordered_dict = {v: k for k, v in self.tag_ordered_dict.iteritems()}

        pi = np.zeros(shape=(n+1, T, T))
        bp = np.ones(shape=(n+1, T, T), dtype='int32') * -1

        # initialization
        pi[0, self.tag_ordered_dict['*'], self.tag_ordered_dict['*']] = 1

        # algorithm:

        for k in xrange(1, n+1):
            #print 'viterbi k iteration: ' + str(k)
            for u in self.get_Sk(k-1, sentence_index):              # TODO consistancy
                for v in self.get_Sk(k, sentence_index):            # TODO relevant tags only
                    current_max_pi = 0
                    current_argmax_pi = -1          # index of best tag
                    for t in self.get_Sk(k-2, sentence_index):

                        prob = pi[k-1, self.tag_ordered_dict[t], self.tag_ordered_dict[u]]
                        q = self.get_q(v=v, t=t, u=u, w=sentence_index, k=k)

                        current_value = prob * q

                        # TODO: max based on numpy.max
                        if current_value > current_max_pi:
                            current_max_pi = current_value
                            current_argmax_pi = self.tag_ordered_dict[t]

                    pi[k, self.tag_ordered_dict[u], self.tag_ordered_dict[v]] = current_max_pi
                    bp[k, self.tag_ordered_dict[u], self.tag_ordered_dict[v]] = current_argmax_pi

            #print 'DEBUG: end of viterbi iteration #' + str(k) + ' of total len(sentence)=' + str(n)

        # TODO: finish algorithm - final run for tn tn-1)

        u = np.unravel_index(pi[n].argmax(), pi[n].shape)[0]
        v = np.unravel_index(pi[n].argmax(), pi[n].shape)[1]

        output_tag_dict[n-1] = self.flip_tag_ordered_dict[v]
        output_tag_dict[n-2] = self.flip_tag_ordered_dict[u]

        for k in xrange(n-2, 0, -1):
            #print 'xrange: ' + str(k)
            output_tag_dict[k-1] = self.flip_tag_ordered_dict[bp[k+2, self.tag_ordered_dict[output_tag_dict[k]], self.tag_ordered_dict[output_tag_dict[k+1]]]]

        return output_tag_dict

    def get_q(self, v, t, u, w, k):

        tag_exp_dict = {}
        sum_dict_denominator = 0

        cur_w = self.sentences_storage[w][k-1]         # k-1 our fix left shift
        # create relevant list
        tag_w_list = []
        if cur_w in self.word_dictionary_count:
            for existing_tag, cnt in self.word_dictionary_count[cur_w].iteritems():
                if existing_tag != 'cnt':
                    tag_w_list.append(existing_tag)
        else:
            tag_w_list = self.common_tag_list

        for tag_prime in tag_w_list:

            if ((t, u, w, k-1), tag_prime) in self.f_v_prime:       # w: idx of cur sentences, k-1: index of specific word in sentence
                f_v_current = self.f_v_prime[(t, u, w, k-1), tag_prime]

            else:
                print 'bug! not contain in f_v_prime'
                f_v_current = self.memm_obj.generate_feature_vector(m_2_t=t, m_1_t=u, s_i=w, w_i=k, w_t=tag_prime, relevant_storage=self.sentences_storage)

            cur_res = math.exp(f_v_current.dot(self.v))
            sum_dict_denominator += cur_res
            tag_exp_dict[tag_prime] = cur_res           # TODO reduce we need to save only one tag
        if v in tag_exp_dict:
            return tag_exp_dict[v] / float(sum_dict_denominator)
        else:
            return 0

    def get_Sk(self, k, w):
        if k in (-1, 0):
            return ('*')
        else:
            # get all relevant tags
            cur_w = self.sentences_storage[w][k - 1]  # k-1 our fix left shift
            tag_w_dict = {}
            if cur_w in self.word_dictionary_count:
                for existing_tag, cnt in self.word_dictionary_count[cur_w].iteritems():
                    if existing_tag != 'cnt':
                        tag_w_dict[existing_tag] = self.tag_ordered_dict[existing_tag]
            else:
                tag_w_dict = self.common_tag_dict
            return tag_w_dict

    def run_viterbi_on_all_storage(self):

        predicted_values = {}
        for idx, cur_sentences in self.sentences_storage.iteritems():
            #print idx
            #print cur_sentences
            output_tag_dict = self.run_viterbi_algorithm_on_sentence(cur_sentences, idx)
            #print 'output dict'
            #print output_tag_dict
            inner_list = []
            for idx_tag, tag in output_tag_dict.iteritems():
                #print 'check'
                #print self.sentences_storage
                #print self.sentences_storage[idx]
                #print self.sentences_storage[idx][idx_tag]
                str_pre = str(self.sentences_storage[idx][idx_tag]) + '_' + str(tag)
                inner_list.append(str_pre)
            predicted_values[idx] = inner_list
            print 'predict list: ' + str(inner_list)
        print 'finish'
        print predicted_values
        return predicted_values

def main():

    evaluate_train_data = True
    evaluate_test_data = True
    feature_list = ['104']

    train_file = 'C:\\gitprojects\\nlp\\hw1\\data\\small_train.wtag'
    test_file = 'C:\\gitprojects\\nlp\\hw1\\data\\small_test_1.wtag'
    comp_file = 'C:\\gitprojects\\nlp\\hw1\\data\\comp.wtag'

    memm_obj = MEMM(evaluate_train_data, evaluate_test_data, feature_list, train_file, test_file, comp_file)

    gradient_obj = Gradient(memm_obj=memm_obj, PARAM_LAMBDA=0.0001)

    # v = gradient_obj.gradientDescent()

    v = np.ones_like(np.arange(memm_obj.feature_vec_type_counter['All']))

    viterbi_obj = Viterbi(memm_obj, type = 'test', v=v)

    sentence_1 = memm_obj.test_sentences_storage[0]         # load only 1 sentences

    # TODO wrap all document

    viterbi_obj.run_viterbi_algorithm_on_sentence(sentence=sentence_1, sentence_index=0)

    # TODO evaluate data

if __name__ == "__main__":
    main()