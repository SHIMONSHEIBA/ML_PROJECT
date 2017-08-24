from HMM import HMM
from viterbi_ML import viterbi
import time
from Print_and_save_results import print_save_results
from hmm_learn import HMM_learn


def main():
    chrome_train_list = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12']
    chrome_test_list = ['13', '14', '15', '16', '17']
    print '{}: Start creating HMM'.format((time.asctime(time.localtime(time.time()))))
    lambda1 = 0.8
    lambda2 = 0.1
    lambda3 = 0.1
    # hmm_learn = HMM_learn(train_file, lambda1, lambda2, lambda3, is_smooth=False, train_data_file=train_data_file)
    hmm = HMM(chrome_train_list, lambda1, lambda2, lambda3, is_smooth=False)

    for chrome in chrome_test_list:
        test_file = 'C:\\gitprojects\\ML_PROJECT\\labels150\\chr' + chrome + '_label.csv'
        print '{}: Start viterbi for chrome: {}'.format((time.asctime(time.localtime(time.time()))), chrome)
        viterbi_obj = viterbi(hmm, 'hmm', data_file=test_file, is_log=False, use_stop_prob=False)
        viterbi_result = viterbi_obj.viterbi_all_data()

        print 'start evaluation'
        write_file_name = 'C:\\gitprojects\\ML_PROJECT\\file_results\\chr' + chrome + '_resultNoStop.csv'
        confusion_file_name = 'C:\\gitprojects\\ML_PROJECT\\confusion_files\\chr' + chrome + '_CMNoStop.xls'
        seq_confusion_file_name = 'C:\\gitprojects\\ML_PROJECT\\confusion_files\\chr' + chrome + '_sqeCMNoStop.xls'
        # seq_labels_file_name = 'C:/gitprojects/ML project/samples_small_data/chr1_sample_label.xlsx'
        seq_labels_file_name = 'C:\\gitprojects\\ML_PROJECT\\sample_labels150\\chr' + chrome + '_sample_label.xlsx'
        evaluate_obj = print_save_results(hmm, 'hmm', test_file, viterbi_result, write_file_name,
                                          confusion_file_name, seq_labels_file_name, seq_confusion_file_name)
        evaluate_result = evaluate_obj.run()

        print evaluate_result

if __name__ == "__main__":
    main()
