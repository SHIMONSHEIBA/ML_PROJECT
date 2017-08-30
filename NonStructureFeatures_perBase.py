from MEMM_try import MEMM
import time
from Print_and_save_results import print_save_results
import logging
from datetime import datetime
from Check_non_structure_classifiers import Classifier
import pandas as pd
import numpy as np

LOG_FILENAME = datetime.now().strftime('C:\\gitprojects\\ML_PROJECT\\logs\\LogFileNonStructure_%d_%m_%Y_%H_%M.log')
logging.basicConfig(filename=LOG_FILENAME, level=logging.INFO)


class NonStructureFeatures_perBase:
    def __init__(self, chrome_train_list, chrome_test_list=None):
        # chore to use as training data
        self.chrome_train_list = chrome_train_list
        self.chrome_test_list = chrome_test_list

        features_list = ['feature_word', 'feature_3', 'feature_4', 'feature_5', 'feature_6', 'feature_7',
                         'feature_8']

        logging.info('{}: Train list is: {}, test list is: {}'
                     .format(time.asctime(time.localtime(time.time())), chrome_train_list, chrome_test_list))
        print('{}: Start creating Features (using MEMM)'.format(time.asctime(time.localtime(time.time()))))
        logging.info('{}: Start creating Features (using MEMM)'.format(time.asctime(time.localtime(time.time()))))
        logging.info('MEMM for features : {}'.format(features_list))

        # just create the feature vector indexes
        NonStructureFeaturesIndexes = MEMM(chrome_train_list, features_list, history_tag_feature_vector=True)
        self.all_train_samples_features = self.create_feature_vector(NonStructureFeaturesIndexes, train=True,
                                                                     majority=False)
        if chrome_test_list is not None:
            self.all_test_samples_features = self.create_feature_vector(NonStructureFeaturesIndexes, train=False,
                                                                        majority=True)

    def create_feature_vector(self, NonStructureFeaturesIndexes, train=True, majority=True):

            if train:
                chrome_list = self.chrome_train_list
            else:
                chrome_list = self.chrome_test_list

            print('{}: starting building feature vector for chrome_list: {}'.
                  format(time.asctime(time.localtime(time.time())), chrome_list))
            all_data_index = 0
            for chrome in chrome_list:
                training_file = 'C:\\gitprojects\\ML_PROJECT\\labels150\\chr' + chrome + '_label.csv'
                # seq_labels_file_name = 'C:\\gitprojects\\ML_PROJECT\\first_base_label\\chr' + chrome + '_first_label.xlsx'
                # seq_label = pd.read_excel(seq_labels_file_name, header=None)
                # seq_label_array = seq_label.as_matrix()

                with open(training_file, 'r') as training:

                    sequence_index = 1
                    for sequence in training:
                        print(sequence_index)
                        if sequence_index == 1150:
                            reut = 1

                        word_tag_list = sequence.split(',')

                        if '\n' in word_tag_list[len(word_tag_list) - 1]:
                            word_tag_list[len(word_tag_list) - 1] = word_tag_list[len(word_tag_list) - 1].replace('\n',
                                                                                                                  '')
                        while '' in word_tag_list:
                            word_tag_list.remove('')
                        while ' ' in word_tag_list:
                            word_tag_list.remove(' ')
                        while '\n' in word_tag_list:
                            word_tag_list.remove('\n')
                        while ',' in word_tag_list:
                            word_tag_list.remove(',')

                        #print("working on sequence {} :".format(sequence_index))
                        #print(word_tag_list)

                        # define three first word_tags for some features
                        first_tag = '#'
                        second_tag = '#'

                        zero_word = '#'
                        first_word = '#'
                        second_word = '#'
                        plus_one_word = ''
                        plus_two_word = ''
                        plus_three_word = ''
                        more_than_3 = True

                        # tags_list = seq_label_array[sequence_index]
                        # first_tag = tags_list[0]
                        # if first_tag in range(1, 5):  # first base is part of a gene
                        #     first_tag = 1
                        # elif first_tag in range(5, 9):  # first base is not part of a gene
                        #     first_tag = -1
                        # else:
                        #     print('Error: tag for sequence {} in chrome {} not in (1,8)'.format(sequence_index, chrome))

                        for word_in_seq_index, word_tag in enumerate(word_tag_list):

                            word_tag_tuple = word_tag.split('_')

                            if '\n' in word_tag_tuple[1]:
                                word_tag_tuple[1] = word_tag_tuple[1][:1]

                            current_word = word_tag_tuple[0]
                            current_tag = word_tag_tuple[1]
                            if len(word_tag_list) - word_in_seq_index > 3:
                                plus_one_word = word_tag_list[word_in_seq_index + 1][0]
                                plus_two_word = word_tag_list[word_in_seq_index + 2][0]
                                plus_three_word = word_tag_list[word_in_seq_index + 3][0]
                            elif more_than_3:
                                plus_one_word = word_tag_list[word_in_seq_index + 1][0]
                                plus_two_word = word_tag_list[word_in_seq_index + 2][0]
                                plus_three_word = '#'
                                more_than_3 = False

                            indexes_vector = NonStructureFeaturesIndexes.calculate_history_tag_indexes\
                                (first_tag, second_tag, zero_word, first_word, second_word, plus_one_word,
                                 plus_two_word, plus_three_word, current_word, current_tag)

                            first_tag = second_tag
                            second_tag = current_tag
                            zero_word = first_word
                            first_word = second_word
                            second_word = current_word
                            if not more_than_3:
                                plus_one_word = plus_two_word
                                plus_two_word = plus_three_word

                            labels = range(0, indexes_vector.shape[0])
                            labels.append('IsGen')
                            if current_tag in ['1', '2', '3', '4']:  # first base is part of a gene
                                current_tag = 1
                            elif current_tag in ['5', '6', '7', '8']:  # first base is not part of a gene
                                current_tag = -1
                            else:
                                print(
                                'Error: tag for sequence {} in chrome {} not in (1,8)'.format(sequence_index, chrome))
                            indexes_vector = np.append(indexes_vector, current_tag)

                            featuresDF = pd.Series(indexes_vector, index=labels)

                            if all_data_index == 0:
                                all_samples_features = featuresDF
                            else:
                                all_samples_features = pd.concat([featuresDF, all_samples_features], axis=1)
                            # update indexes - after each sequence
                            all_data_index += 1
                        sequence_index += 1
                        # for test data and majority: need features just for the first base
                        if not train and sequence_index == 0 and majority:
                            break
            return all_samples_features.T


def main():
    # chrome_train_list = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12']
    # chrome_test_list = ['13', '14', '15', '16', '17']

    NonStructureFeatures_perBase_obj = NonStructureFeatures_perBase(chrome_train_list, chrome_test_list)
    classifier = Classifier(NonStructureFeatures_perBase_obj, use_CV=False)
    classifier.ModelsIteration()

    # print('start evaluation')
    # write_file_name = datetime.now().strftime\
    #     ('C:\\gitprojects\\\ML_PROJECT\\file_results\\chr' + chrome + '_resultMEMM_%d_%m_%Y_%H_%M.csv')
    # confusion_file_name = datetime.now().strftime\
    #     ('C:\\gitprojects\\ML_PROJECT\\confusion_files\\chr' + chrome + '_CMMEMM_%d_%m_%Y_%H_%M.xls')
    # seq_confusion_file_name = datetime.now().strftime\
    #     ('C:\\gitprojects\\ML_PROJECT\\confusion_files\\chr' + chrome + '_sqeCMMEMM_%d_%m_%Y_%H_%M.xls')
    # # seq_labels_file_name = 'C:/gitprojects/ML project/samples_small_data/chr1_sample_label.xlsx'
    # seq_labels_file_name = 'C:\\gitprojects\\ML_PROJECT\\sample_labels150\\chr' + chrome + '_sample_label.xlsx'
    # evaluate_obj = print_save_results(memm, 'memm', test_file, viterbi_result, write_file_name,
    #                                   confusion_file_name, seq_labels_file_name, seq_confusion_file_name)
    # word_results_dictionary, seq_results_dictionary = evaluate_obj.run()
    #
    # print(word_results_dictionary)
    # print(seq_results_dictionary)
    # logging.info('Following Evaluation results for features {}'.format(features_combination))
    # logging.info('{}: Evaluation results for chrome number: {} are: \n {} \n {} \n'.
    #              format(time.asctime(time.localtime(time.time())), chrome, word_results_dictionary,
    #                     seq_results_dictionary))
    # logging.info('-----------------------------------------------------------------------------------')

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
