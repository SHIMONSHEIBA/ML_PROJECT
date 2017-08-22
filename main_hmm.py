from HMM import HMM
from viterbi_ML import viterbi
import time
from Print_and_save_results import print_save_results


def main():
    chrome_number = '1'
    # train_file = 'C:\\gitprojects\\ML_PROJECT\\labels\\chr' + chrome_number + '_label.csv'
    train_file = 'C:/gitprojects/ML project/samples_small_data/label_small.csv'
    print '{}: Start creating HMM'.format((time.asctime(time.localtime(time.time()))))
    lambda1 = 0.2
    lambda2 = 0.5
    lambda3 = 0.3
    hmm = HMM(train_file, lambda1, lambda2, lambda3)

    print '{}: Start viterbi'.format((time.asctime(time.localtime(time.time()))))
    viterbi_obj = viterbi(hmm, 'hmm', train_file, is_log=True)
    viterbi_result = viterbi_obj.viterbi_all_data()

    print 'start evaluation'
    write_file_name = 'C:\\gitprojects\\ML_PROJECT\\file_results\\chr' + chrome_number + '_result.csv'
    confusion_file_name = 'C:\\gitprojects\\ML_PROJECT\\confusion_files\\chr' + chrome_number + '_CM.xls'
    seq_confusion_file_name = 'C:\\gitprojects\\ML_PROJECT\\confusion_files\\chr' + chrome_number + '_sqeCM.xls'
    seq_labels_file_name = 'C:/gitprojects/ML project/samples_small_data/chr1_sample_label.xlsx'
    evaluate_obj = print_save_results(hmm, 'hmm', train_file, viterbi_result, write_file_name,
                                      confusion_file_name, seq_labels_file_name, seq_confusion_file_name)
    evaluate_result = evaluate_obj.run()

    print evaluate_result

if __name__ == "__main__":
    main()
