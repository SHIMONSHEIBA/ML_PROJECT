from HMM import HMM
from viterbi_ML import viterbi
from Print_and_save_results import print_save_results
import time
from datetime import datetime
import logging

directory = 'C:\\Users\\Meir\\PycharmProjects\\ML_PROJECT\\'
LOG_FILENAME = datetime.now().strftime(directory + 'logs\\LogFileHMMNo17_%d_%m_%Y_%H_%M.log')
logging.basicConfig(filename=LOG_FILENAME, level=logging.INFO)


def main():
    logging.info('{}: Train list is : {}, test list is : {}'
                 .format(time.asctime(time.localtime(time.time())), chrome_train_list, chrome_test_list))
    print('{}: Start creating HMM'.format(time.asctime(time.localtime(time.time()))))
    logging.info('{}: Start creating HMM'.format(time.asctime(time.localtime(time.time()))))
    lambda1 = 0.8
    lambda2 = 0.1
    lambda3 = 0.1

    hmm_class = HMM(chrome_train_list, lambda1, lambda2, lambda3, is_smooth=False)

    use_stop_prob = False
    logging.info('{}: use stop probability is: {}'.format(time.asctime(time.localtime(time.time())), use_stop_prob))

    for chrome in chrome_test_list:
        test_file = directory + 'labels150\\chr' + chrome + '_label.csv'
        print('{}: Start viterbi for chrome: {}'.format((time.asctime(time.localtime(time.time()))), chrome))
        viterbi_class = viterbi(hmm_class, 'hmm', data_file=test_file, is_log=False, use_stop_prob=use_stop_prob,
                              phase_number=1, use_majority_vote=False)
        viterbi_result = viterbi_class.viterbi_all_data(chrome)

        write_file_name = datetime.now().strftime(directory + 'file_results\\chr' + chrome +
                                                  '_resultTry150_%d_%m_%Y_%H_%M.csv')
        confusion_file_name = datetime.now().strftime(directory + 'confusion_files\\chr' + chrome +
                                                      '_CMTry150_%d_%m_%Y_%H_%M.xls')
        seq_confusion_file_name = datetime.now().strftime(directory + 'confusion_files\\chr' + chrome +
                                                          '_sqeCMTry150_%d_%m_%Y_%H_%M.xls')
        seq_labels_file_name = directory + 'sample_labels150\\chr' + chrome + '_sample_label.xlsx'
        logging.info('{}: Related results files are: \n {} \n {} \n {}'.format(time.asctime(time.localtime(time.time())),
                     write_file_name, confusion_file_name, seq_confusion_file_name))
        evaluate_class = print_save_results(hmm_class, 'hmm', test_file, viterbi_result, write_file_name,
                                          confusion_file_name, seq_labels_file_name, seq_confusion_file_name)
        word_results_dictionary, seq_results_dictionary = evaluate_class.run()

        print(word_results_dictionary)
        print(seq_results_dictionary)
        logging.info('{}: Evaluation results for chrome number: {} are: \n {} \n {} \n'.
                     format(time.asctime(time.localtime(time.time())), chrome, word_results_dictionary,
                            seq_results_dictionary))

        logging.info('-----------------------------------------------------------------------------------')

if __name__ == "__main__":
    all_chromes = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16']  #, '17']
    for test_chrome in range(1, 18):
        chrome_train_list = [x for x in all_chromes if x != str(test_chrome)]
        print(chrome_train_list)
        chrome_test_list = [str(test_chrome)]
        print(chrome_test_list)
        main()


