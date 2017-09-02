from HMM import HMM
from viterbi_ML import viterbi
from Print_and_save_results import print_save_results
import time
from datetime import datetime
import logging
from MEMM_try import MEMM
from gradient_try import Gradient
from collections import Counter
from NonStructureFeatures_perBase import NonStructureFeatures_perBase
import csv
from Check_non_structure_classifiers import Classifier
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import cross_val_predict, LeaveOneOut
from sklearn.svm import SVC


directory = 'C:\\gitprojects\\ML_PROJECT\\'

logging.getLogger('').handlers = []
LOG_FILENAME = datetime.now().strftime(directory + 'logs\\LogFileMajority2_%d_%m_%Y_%H_%M.log')
logging.basicConfig(filename=LOG_FILENAME, level=logging.INFO)


def write_majority_doc(chrome_result, compare_list, sequence_index):
    write_file_name = directory + 'majority_vote2\\chr' + chrome_result + '_majority_vote_results_first_base.csv'
    with open(write_file_name, 'a') as csv_file:
        writer = csv.writer(csv_file)
        if sequence_index == 0:
            writer.writerow(
                ['prediction_hmm', 'prediction_memm', 'prediction_firstNonStructure',
                 'prediction_secondNonStructure', 'third_secondNonStructure', 'sequence_index'])
        prediction_list = [prediction for prediction in compare_list]
        prediction_list.append(sequence_index)
        writer.writerow(prediction_list)

    return


def main():
    print('{}: Train list is: {}, test list is: {}'.format(time.asctime(time.localtime(time.time()))
                                                           , chrome_train_list, chrome_test_list))
    logging.info('{}: Train list is: {}, test list is: {}'
                 .format(time.asctime(time.localtime(time.time())), chrome_train_list, chrome_test_list))
    print('{}: Start creating HMM'.format(time.asctime(time.localtime(time.time()))))
    logging.info('{}: Start creating HMM'.format(time.asctime(time.localtime(time.time()))))
    lambda1 = 0.8
    lambda2 = 0.1
    lambda3 = 0.1
    # Train HMM on training data
    hmm_class = HMM(chrome_train_list, lambda1, lambda2, lambda3, is_smooth=False)
    # Train MEMM on training data
    features_combination_list = ['feature_word_tag', 'feature_word', 'feature_tag', 'feature_1', 'feature_2',
                                 'feature_3', 'feature_4', 'feature_5', 'feature_6', 'feature_7', 'feature_8']
    memm_class = MEMM(chrome_train_list, features_combination_list)
    gradient_class = Gradient(memm=memm_class, lamda=1)
    gradient_result = gradient_class.gradient_descent()
    weights = gradient_result.x
    # Train non-structure classifier
    # need to return a dictionary that each seq in chrome_test_list have the first base prediction
    # in the format: {seq_index:base_tag}
    # svm_results - SVM(chrome_train_list, chrome_test_list)
    NonStructureFeatures_perBase_train_obj = NonStructureFeatures_perBase(majority=True)
    NonStructureModels = {}
    for clf, name in (
            (RidgeClassifier(tol=1e-2, solver="sag"), "Ridge Classifier"),
            (Perceptron(n_iter=50), "Perceptron"),
            (MultinomialNB(alpha=.01), 'MultinomialNB')):
        NonStructureModels[name] = clf.fit(NonStructureFeatures_perBase_train_obj.X_train,
                                           NonStructureFeatures_perBase_train_obj.Y_train)

    use_stop_prob = False
    logging.info('{}: use stop probability is: {}'.format(time.asctime(time.localtime(time.time())), use_stop_prob))

    for chrome in chrome_test_list:
        test_file = directory + 'labels150\\chr' + chrome + '_label.csv'
        print('{}: Start viterbi HMM for chrome: {} phase 1'.format((time.asctime(time.localtime(time.time()))), chrome))
        viterbi_class_hmm = viterbi(hmm_class, 'hmm', data_file=test_file, is_log=False, use_stop_prob=use_stop_prob,
                              phase_number=1, use_majority_vote=False, use_majority2=True)
        # need to return a dictionary that each seq in chrome_test_list have the first base prediction
        # in the format: {seq_index:base_tag}
        hmm_viterbi_result = viterbi_class_hmm.viterbi_all_data(chrome)
        print('{}: Start viterbi MEMM for chrome: {} phase 1'.\
            format((time.asctime(time.localtime(time.time()))), chrome))
        viterbi_class_memm = viterbi(memm_class, 'memm', data_file=test_file, is_log=False, use_stop_prob=use_stop_prob,
                              phase_number=1, use_majority_vote=False, w=weights, use_majority2=True)
        # need to return a dictionary that each seq in chrome_test_list have the first base prediction
        # in the format: {seq_index:base_tag}
        memm_viterbi_result = viterbi_class_memm.viterbi_all_data(chrome)
        print('{}: Start train non-structure classifier for chrome: {}'.\
            format((time.asctime(time.localtime(time.time()))), chrome))
        # Train non-structure classifier
        # need to return a dictionary that each seq in chrome_test_list have the first base prediction
        # in the format: {seq_index:base_tag}
        chrome_len = len(memm_viterbi_result.keys())
        NonStructurePredictions = {k: [] for k in range(chrome_len)}
        NonStructureFeatures_perBase_test_class = \
            NonStructureFeatures_perBase(is_train=False, chrome_test_list=[chrome],
                                         train_object=NonStructureFeatures_perBase_train_obj)
        for name, model in NonStructureModels.items():
            prediction = model.predict(NonStructureFeatures_perBase_test_class.X_test)
            for sequence_inner_index in range(NonStructureFeatures_perBase_test_class.X_test.shape[0]):
                NonStructurePredictions[sequence_inner_index].append(prediction[sequence_inner_index])

        most_common_tags_first_base = {}
        for sequence_index in range(chrome_len):
            compare_list = []
            # add each model prediction for the first base:
            compare_list.append(hmm_viterbi_result[sequence_index])
            compare_list.append(memm_viterbi_result[sequence_index])
            word_in_index = compare_list[0].split('_')[0]
            for tags in NonStructurePredictions[sequence_index]:
                if tags == 1:
                    compare_list.append(word_in_index + '_' + hmm_class.word_tag_dict[word_in_index[0]][0])
                elif tags == -1:
                    compare_list.append(word_in_index + '_' + hmm_class.word_tag_dict[word_in_index[0]][1])
            # compare_list.append(svm_results[sequence_index])
            count = Counter(compare_list)

            most_common_tags_first_base[sequence_index] = count.most_common()[0][0]
            write_majority_doc(chrome, compare_list, sequence_index)
        print('{}: Start viterbi HMM for chrome: {} phase 2'.format((time.asctime(time.localtime(time.time()))), chrome))
        viterbi_class_phase2_hmm = viterbi(hmm_class, 'hmm', data_file=test_file, is_log=False,
                                           use_stop_prob=use_stop_prob, phase_number=2, use_majority_vote=False,
                                           prediction_for_phase2=most_common_tags_first_base)
        phase2_viterbi_result_hmm = viterbi_class_phase2_hmm.viterbi_all_data(chrome)

        write_file_name = datetime.now().strftime(directory + 'file_results\\chr' + chrome +
                                                  '_resultMajority2HMM_%d_%m_%Y_%H_%M.csv')
        confusion_file_name = datetime.now().strftime(directory + 'confusion_files\\chr' + chrome +
                                                      '_CMMajority2HMM_%d_%m_%Y_%H_%M.xls')
        seq_confusion_file_name = datetime.now().strftime(directory + 'confusion_files\\chr' + chrome +
                                                          '_sqeCMMajority2HMM_%d_%m_%Y_%H_%M.xls')
        seq_labels_file_name = directory + 'sample_labels150\\chr' + chrome + '_sample_label.xlsx'
        logging.info('{}: Related results files are: \n {} \n {} \n {}'.
                     format(time.asctime(time.localtime(time.time())), write_file_name, confusion_file_name,
                            seq_confusion_file_name))
        evaluate_obj = print_save_results(hmm_class, 'hmm', test_file, phase2_viterbi_result_hmm, write_file_name,
                                          confusion_file_name, seq_labels_file_name, seq_confusion_file_name)
        word_results_dictionary, seq_results_dictionary = evaluate_obj.run()

        print(word_results_dictionary)
        print(seq_results_dictionary)
        logging.info('{}: Evaluation results for chrome number: {}, after freeze the first base are: \n {} \n {} \n'.
                     format(time.asctime(time.localtime(time.time())), chrome, word_results_dictionary,
                            seq_results_dictionary))

        logging.info('-----------------------------------------------------------------------------------')

        print('{}: Start viterbi MEMM for chrome: {} phase 2'.format((time.asctime(time.localtime(time.time()))),
                                                                    chrome))
        viterbi_class_phase2_memm = viterbi(memm_class, 'memm', data_file=test_file, is_log=False,
                                            use_stop_prob=use_stop_prob, phase_number=2, use_majority_vote=False,
                                            w=weights, prediction_for_phase2=most_common_tags_first_base)
        phase2_viterbi_result_memm = viterbi_class_phase2_memm.viterbi_all_data(chrome)

        print('start evaluation')
        write_file_nameMEMM = datetime.now().strftime(directory + 'file_results\\chr' + chrome +
                                                  '_resultMajority2MEMM_%d_%m_%Y_%H_%M.csv')
        confusion_file_nameMEMM = datetime.now().strftime(directory + 'confusion_files\\chr' + chrome +
                                                      '_CMMajority2MEMM_%d_%m_%Y_%H_%M.xls')
        seq_confusion_file_nameMEMM = datetime.now().strftime(directory + 'confusion_files\\chr' + chrome +
                                                          '_sqeCMMajority2MEMM_%d_%m_%Y_%H_%M.xls')
        seq_labels_file_name = directory + 'sample_labels150\\chr' + chrome + '_sample_label.xlsx'
        logging.info('{}: Related results files are: \n {} \n {} \n {}'.
                     format(time.asctime(time.localtime(time.time())), write_file_nameMEMM, confusion_file_nameMEMM,
                            seq_confusion_file_name))
        evaluate_class = print_save_results(memm_class, 'memm', test_file, phase2_viterbi_result_memm,
                                          write_file_nameMEMM, confusion_file_nameMEMM, seq_labels_file_name,
                                          seq_confusion_file_nameMEMM)
        word_results_dictionary, seq_results_dictionary = evaluate_class.run()

        print(word_results_dictionary)
        print(seq_results_dictionary)
        logging.info('{}: Evaluation results for chrome number: {}, after freeze the first base are: \n {} \n {} \n'.
                     format(time.asctime(time.localtime(time.time())), chrome, word_results_dictionary,
                            seq_results_dictionary))

        logging.info('-----------------------------------------------------------------------------------')


if __name__ == "__main__":
    # all_chromes = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17']
    # for test_chrome in range(1, 18):
    #     chrome_train_list = [x for x in all_chromes if x != str(test_chrome)]
    #     print chrome_train_list
    #     chrome_test_list = [str(test_chrome)]
    #     print chrome_test_list
    chrome_train_list = ['17']
    chrome_test_list = ['17']
    main()
