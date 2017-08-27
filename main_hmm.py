from HMM import HMM
from viterbi_ML import viterbi
import time
from Print_and_save_results import print_save_results
from hmm_learn import HMM_learn
import time
from datetime import datetime
import logging

LOG_FILENAME = datetime.now().strftime('C:\\gitprojects\\ML_PROJECT\\logs\\LogFileTry150_%d_%m_%Y_%H_%M.log')
logging.basicConfig(filename=LOG_FILENAME, level=logging.INFO)

def main():
    chrome_train_list = ['16', '12', '7', '15', '4', '8', '5', '11', '14', '2', '13', '10']
    chrome_test_list = ['17', '1', '6', '3', '9']
    logging.info('{}: Train list is (short chromes): {}, test list is (long chromes): {}'
                 .format(time.asctime(time.localtime(time.time())), chrome_train_list, chrome_test_list))
    print('{}: Start creating HMM'.format(time.asctime(time.localtime(time.time()))))
    logging.info('{}: Start creating HMM'.format(time.asctime(time.localtime(time.time()))))
    lambda1 = 0.8
    lambda2 = 0.1
    lambda3 = 0.1
    # hmm_learn = HMM_learn(train_file, lambda1, lambda2, lambda3, is_smooth=False, train_data_file=train_data_file)
    hmm = HMM(chrome_train_list, lambda1, lambda2, lambda3, is_smooth=False)

    use_stop_prob = False
    logging.info('{}: use stop probability is: {}'.format(time.asctime(time.localtime(time.time())), use_stop_prob))

    for chrome in chrome_test_list:
        test_file = 'C:\\gitprojects\\ML_PROJECT\\labels150\\chr' + chrome + '_label.csv'
        print '{}: Start viterbi for chrome: {}'.format((time.asctime(time.localtime(time.time()))), chrome)
        viterbi_obj = viterbi(hmm, 'hmm', data_file=test_file, is_log=False, use_stop_prob=use_stop_prob,
                              use_majority_vote=True)
        viterbi_result = viterbi_obj.viterbi_all_data()

        print('start evaluation')
        write_file_name = datetime.now().strftime('C:\\gitprojects\\ML_PROJECT\\file_results\\chr' + chrome +
                                                  '_resultTry150_%d_%m_%Y_%H_%M.csv')
        confusion_file_name = datetime.now().strftime('C:\\gitprojects\\ML_PROJECT\\confusion_files\\chr' + chrome +
                                                      '_CMTry150_%d_%m_%Y_%H_%M.xls')
        seq_confusion_file_name = datetime.now().strftime('C:\\gitprojects\\ML_PROJECT\\confusion_files\\chr' + chrome +
                                                          '_sqeCMTry150_%d_%m_%Y_%H_%M.xls')
        seq_labels_file_name = 'C:\\gitprojects\\ML_PROJECT\\sample_labels150\\chr' + chrome + '_sample_label.xlsx'
        logging.info('{}: Related results files are: \n {} \n {} \n {}'.format(time.asctime(time.localtime(time.time())),
                     write_file_name, confusion_file_name, seq_confusion_file_name))
        evaluate_obj = print_save_results(hmm, 'hmm', test_file, viterbi_result, write_file_name,
                                          confusion_file_name, seq_labels_file_name, seq_confusion_file_name)
        word_results_dictionary, seq_results_dictionary = evaluate_obj.run()

        print(word_results_dictionary)
        print(seq_results_dictionary)
        logging.info('{}: Evaluation results for chrome number: {} are: \n {} \n {} \n'.
                     format(time.asctime(time.localtime(time.time())), chrome, word_results_dictionary,
                            seq_results_dictionary))

if __name__ == "__main__":
    main()
