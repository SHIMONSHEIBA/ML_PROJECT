from MEMM_try import MEMM
from viterbi_ML import viterbi
import time
from Print_and_save_results import print_save_results
import numpy as np
from gradient_try import Gradient
import logging
from datetime import datetime

LOG_FILENAME = datetime.now().strftime\
    ('C:\\gitprojects\\ML_PROJECT\\logs\\LogFileMEMM_%d_%m_%Y_%H_%M.log')
logging.basicConfig(filename=LOG_FILENAME, level=logging.INFO)

def main():
    chrome_train_list = ['1']#, '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12']
    chrome_test_list = ['13']#, '14', '15', '16', '17']
    logging.info('{}: Train list is: {}, test list is: {}'
                 .format(time.asctime(time.localtime(time.time())), chrome_train_list, chrome_test_list))
    print('{}: Start creating MEMM'.format(time.asctime(time.localtime(time.time()))))
    logging.info('{}: Start creating HMM'.format(time.asctime(time.localtime(time.time()))))
    memm = MEMM(chrome_train_list)

    gradient_class = Gradient(memm=memm, lamda=1)
    gradient_result = gradient_class.gradient_descent()
    weights = gradient_result.x
    #np.savetxt(gradient_file, weights, delimiter=",")

    for chrome in chrome_test_list:

        test_file = 'C:\\gitprojects\\ML_PROJECT\\labels150\\chr' + chrome + '_label.csv'
        print('{}: Start viterbi for chrome: {}'.format((time.asctime(time.localtime(time.time()))), chrome))
        viterbi_obj = viterbi(memm, 'memm', data_file=test_file, is_log=False, use_stop_prob=False, w=weights )
        viterbi_result = viterbi_obj.viterbi_all_data()

        print('start evaluation')
        write_file_name = datetime.now().strftime\
            ('C:\\gitprojects\\\ML_PROJECT\\file_results\\chr' + chrome + '_resultMEMM_%d_%m_%Y_%H_%M.csv')
        confusion_file_name = datetime.now().strftime\
            ('C:\\gitprojects\\ML_PROJECT\\confusion_files\\chr' + chrome + '_CMMEMM_%d_%m_%Y_%H_%M.xls')
        seq_confusion_file_name = datetime.now().strftime\
            ('C:\\gitprojects\\ML_PROJECT\\confusion_files\\chr' + chrome + '_sqeCMMEMM_%d_%m_%Y_%H_%M.xls')
        # seq_labels_file_name = 'C:/gitprojects/ML project/samples_small_data/chr1_sample_label.xlsx'
        seq_labels_file_name = 'C:\\gitprojects\\ML_PROJECT\\sample_labels150\\chr' + chrome + '_sample_label.xlsx'
        evaluate_obj = print_save_results(memm, 'memm', test_file, viterbi_result, write_file_name,
                                          confusion_file_name, seq_labels_file_name, seq_confusion_file_name)
        word_results_dictionary, seq_results_dictionary = evaluate_obj.run()

        print(word_results_dictionary)
        print(seq_results_dictionary)
        logging.info('{}: Evaluation results for chrome number: {} are: \n {} \n {} \n'.
                     format(time.asctime(time.localtime(time.time())), chrome, word_results_dictionary,
                            seq_results_dictionary))


if __name__ == "__main__":
    main()
