from HMM import HMM
from viterbi_ML import viterbi
from Print_and_save_results import print_save_results
import time
from datetime import datetime
import logging
from MEMM_try import MEMM
from gradient_try import Gradient
from collections import Counter


LOG_FILENAME = datetime.now().strftime('C:\\gitprojects\\ML_PROJECT\\logs\\LogFileMajority150_%d_%m_%Y_%H_%M.log')
logging.basicConfig(filename=LOG_FILENAME, level=logging.INFO)


def main():
    logging.info('{}: Train list is (short chromes): {}, test list is (long chromes): {}'
                 .format(time.asctime(time.localtime(time.time())), chrome_train_list, chrome_test_list))
    print('{}: Start creating HMM'.format(time.asctime(time.localtime(time.time()))))
    logging.info('{}: Start creating HMM'.format(time.asctime(time.localtime(time.time()))))
    lambda1 = 0.8
    lambda2 = 0.1
    lambda3 = 0.1
    # Train HMM on training data
    hmm = HMM(chrome_train_list, lambda1, lambda2, lambda3, is_smooth=False)
    # Train MEMM on training data
    memm = MEMM(chrome_train_list)
    gradient_class = Gradient(memm=memm, lamda=1)
    gradient_result = gradient_class.gradient_descent()
    weights = gradient_result.x

    use_stop_prob = False
    logging.info('{}: use stop probability is: {}'.format(time.asctime(time.localtime(time.time())), use_stop_prob))

    for chrome in chrome_test_list:
        test_file = 'C:\\gitprojects\\ML_PROJECT\\labels150\\chr' + chrome + '_label.csv'
        print '{}: Start viterbi HMM for chrome: {}'.format((time.asctime(time.localtime(time.time()))), chrome)
        viterbi_obj = viterbi(hmm, 'hmm', data_file=test_file, is_log=False, use_stop_prob=use_stop_prob,
                              phase_number=1, use_majority_vote=False)
        # need to return a dictionary that each seq in chrome_test_list have the first base prediction
        # in the format: {seq_index:base_tag}
        hmm_viterbi_result = viterbi_obj.viterbi_all_data(chrome)
        print '{}: Start viterbi MEMM for chrome: {}'.format((time.asctime(time.localtime(time.time()))), chrome)
        viterbi_obj = viterbi(memm, 'memm', data_file=test_file, is_log=False, use_stop_prob=use_stop_prob,
                              phase_number=1, use_majority_vote=False, w=weights)
        # need to return a dictionary that each seq in chrome_test_list have the first base prediction
        # in the format: {seq_index:base_tag}
        memm_viterbi_result = viterbi_obj.viterbi_all_data(chrome)
        print '{}: Start train non-structure classifier for chrome: {}'.format((time.asctime(time.localtime(time.time()))), chrome)
        # Train non-structure classifier
        # need to return a dictionary that each seq in chrome_test_list have the first base prediction
        # in the format: {seq_index:base_tag}
        # svm_results - SVM(chrome_train_list, chrome_test_list)
        chrome_len = len(memm_viterbi_result.keys())
        most_common_tags_first_base = range(chrome_len)
        for sequence_index in range(chrome_len):
            compare_list = []
            # add each model prediction for the first base:
            compare_list.append(hmm_viterbi_result[sequence_index])
            compare_list.append(memm_viterbi_result[sequence_index])
            # compare_list.append(svm_results[sequence_index])
            count = Counter(compare_list)
            #TODO: change most_common_tags_first_base to be dict
            most_common_tags_first_base[sequence_index] = count.most_common()[0][0]
        print '{}: Start viterbi HMM for chrome: {} in phase 2'.format((time.asctime(time.localtime(time.time()))),
                                                                       chrome)
        viterbi_obj_phase2 = viterbi(hmm, 'hmm', data_file=test_file, is_log=False, use_stop_prob=use_stop_prob,
                                     phase_number=2, use_majority_vote=False,
                                     prediction_for_phase2=most_common_tags_first_base)
        phase2_viterbi_result = viterbi_obj_phase2.viterbi_all_data(chrome)

        print('start evaluation')
        write_file_name = datetime.now().strftime('C:\\gitprojects\\ML_PROJECT\\file_results\\chr' + chrome +
                                                  '_resultMajority_%d_%m_%Y_%H_%M.csv')
        confusion_file_name = datetime.now().strftime('C:\\gitprojects\\ML_PROJECT\\confusion_files\\chr' + chrome +
                                                      '_CMMajority_%d_%m_%Y_%H_%M.xls')
        seq_confusion_file_name = datetime.now().strftime('C:\\gitprojects\\ML_PROJECT\\confusion_files\\chr' + chrome +
                                                          '_sqeCMMajority_%d_%m_%Y_%H_%M.xls')
        seq_labels_file_name = 'C:\\gitprojects\\ML_PROJECT\\sample_labels150\\chr' + chrome + '_sample_label.xlsx'
        logging.info('{}: Related results files are: \n {} \n {} \n {}'.
                     format(time.asctime(time.localtime(time.time())), write_file_name, confusion_file_name,
                            seq_confusion_file_name))
        evaluate_obj = print_save_results(hmm, 'hmm', test_file, phase2_viterbi_result, write_file_name,
                                          confusion_file_name, seq_labels_file_name, seq_confusion_file_name)
        word_results_dictionary, seq_results_dictionary = evaluate_obj.run()

        print(word_results_dictionary)
        print(seq_results_dictionary)
        logging.info('{}: Evaluation results for chrome number: {}, after freeze the first base are: \n {} \n {} \n'.
                     format(time.asctime(time.localtime(time.time())), chrome, word_results_dictionary,
                            seq_results_dictionary))

if __name__ == "__main__":
    all_chromes = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17']
    for test_chrome in range(1, 18):
        chrome_train_list = [x for x in all_chromes if x != str(test_chrome)]
        print chrome_train_list
        chrome_test_list = [str(test_chrome)]
        print chrome_test_list
        main()
