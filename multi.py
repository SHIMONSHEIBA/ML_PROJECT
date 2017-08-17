import numpy as np
import math
import threading
import time
from MEMM import MEMM
from gradient import Gradient
from viterbi import Viterbi


exitFlag = 0

class Multi (threading.Thread):
    def __init__(self, threadID, test_file, test_f_v_prime_output_dict):
        threading.Thread.__init__(self)
        self.threadID = threadID

        self.test_file = test_file
        self.test_f_v_prime_output_dict = test_f_v_prime_output_dict

    def run(self):
        evaluate_train_data = True
        evaluate_test_data = True
        feature_list = ['104']

        train_file = 'C:\\gitprojects\\nlp\\hw1\\data\\small_train.wtag'
        test_file = 'C:\\gitprojects\\nlp\\hw1\\data\\small_test.wtag'
        comp_file = 'C:\\gitprojects\\nlp\\hw1\\data\\comp.wtag'

        print "Starting " + self.threadID

        memm_obj = MEMM(evaluate_train_data, evaluate_test_data,
                        feature_list, train_file, test_file=self.test_file, comp_file=comp_file)

        print "Exiting " + self.threadID



def main():

    test_file_1 = 'C:\\gitprojects\\nlp\\hw1\\data\\small_test_1.wtag'
    test_file_2 = 'C:\\gitprojects\\nlp\\hw1\\data\\small_test_1.wtag'
    test_file_3 = 'C:\\gitprojects\\nlp\\hw1\\data\\small_test_1.wtag'
    test_file_4 = 'C:\\gitprojects\\nlp\\hw1\\data\\small_test_1.wtag'

    test_f_v_prime_1 = {}
    test_f_v_prime_2 = {}
    test_f_v_prime_3 = {}
    test_f_v_prime_4 = {}

    # Create new threads
    thread1 = Multi('1', test_file_1, test_f_v_prime_1)
    thread2 = Multi('2', test_file_2, test_f_v_prime_2)
    thread3 = Multi('3', test_file_3, test_f_v_prime_3)
    thread4 = Multi('4', test_file_4, test_f_v_prime_4)


    # Start new Threads
    thread1.start()
    thread2.start()
    thread3.start()
    thread4.start()



    # memm_obj = MEMM(evaluate_train_data, evaluate_test_data, feature_list, train_file, test_file, comp_file)



    print "Exiting Main Thread"

    # multi_obj = Multi(memm_obj)
    #
    # gradient_obj = Gradient(memm_obj=memm_obj, PARAM_LAMBDA=0.0001)
    #
    # # v = gradient_obj.gradientDescent()
    #
    # v = np.ones_like(np.arange(memm_obj.feature_vec_type_counter['All']))
    #
    # viterbi_obj = Viterbi(memm_obj, type = 'test', v=v)
    #
    # sentence_1 = memm_obj.test_sentences_storage[10]
    #
    # viterbi_obj.run_viterbi_algorithm_on_sentence(sentence=sentence_1, sentence_index=10)

if __name__ == "__main__":
    main()