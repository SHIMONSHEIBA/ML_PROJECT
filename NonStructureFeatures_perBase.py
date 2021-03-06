from MEMM_try import MEMM
import time
import logging
from datetime import datetime
from Check_non_structure_classifiers import Classifier
import pandas as pd
import numpy as np
import csv

directory = 'C:\\gitprojects\\ML_PROJECT\\'


class NonStructureFeatures_perBase:
    def __init__(self, majority=False, load_data=True):
        self.chrome_list = ['1', '2', '2.1', '2.2', '3', '4', '4.1', '4.2', '4.3', '4.4', '5', '5.1', '6',
                            '7', '7.1', '7.2', '8', '8.1', '9', '9.1', '10', '10.1', '11', '11.1', '12', '12.1', '12.2',
                            '12.3', '13', '13.1', '13.2', '14', '14.1', '14.2', '15', '15.1', '15.2', '15.3', '16',
                            '16.1', '16.2', '17']

        self.chrome_list_not_done = []
        self.majority = majority
        # just for initialize
        self.X_train = ''
        self.Y_train = ''
        self.X_test = ''
        self.Y_test = ''

        features_list = ['feature_word', 'feature_3', 'feature_4', 'feature_5', 'feature_6', 'feature_7', 'feature_8']
        logging.info('{}: Features are: {}'
                     .format(time.asctime(time.localtime(time.time())), features_list))
        print('{}: Non structure: Start creating Features (using MEMM)'.format(time.asctime(time.localtime(time.time()))))
        logging.info('{}: Non structure: Start creating Features (using MEMM)'.format(time.asctime(time.localtime(time.time()))))
        logging.info('MEMM for features : {}'.format(features_list))

        # just create the feature vector indexes
        if load_data:
            self.NonStructureFeaturesIndexes = MEMM(self.chrome_list, features_list, history_tag_feature_vector=True)
            self.all_samples_features = self.create_feature_vector()
            self.all_samples_features['chrome'].replace \
                ([2.1, 2.2, 4.1, 4.2, 4.3, 4.4, 5.1, 7.1, 7.2, 8.1, 9.1, 10.1, 11.1, 12.1, 12.2, 12.3, 13.1, 13.2, 14.1,
                  14.2, 15.1, 15.2, 15.3, 16.1, 16.2], [2, 2, 4, 4, 4, 4, 5, 7, 7, 8, 9, 10, 11, 12, 12, 12, 13, 13, 14, 14,
                                                        15, 15, 15, 16, 16], inplace=True)

            # self.all_samples_features['chrome'].replace([2.1, 2.2], [2, 2], inplace=True)
            self.all_samples_features['chrome'] = self.all_samples_features['chrome'].astype(int)
            self.all_samples_features.to_hdf(directory + 'vectors\\all_features.hdf', key='df', mode='w')
        else:
            self.all_samples_features = pd.read_hdf(directory + 'vectors\\all_features.pkl', key='df')

    def create_train_test_dataframes(self, chrome_train_list, chrome_test_list):
        train_samples = self.all_samples_features.loc[self.all_samples_features['chrome'].isin(chrome_train_list)]
        if not self.majority:  # not majority: need all samples
            test_samples = self.all_samples_features.loc[self.all_samples_features['chrome'].isin(chrome_test_list)]
        else:  # majority: need only first base prediction
            test_samples = self.all_samples_features.loc[(self.all_samples_features['word_index'] == 0) &
                                                         self.all_samples_features['chrome'].isin(chrome_test_list)]
        print('size of train sample is: {}, test data: {}, all data: {}'.format(train_samples.shape[0],
                                                                                test_samples.shape[0],
                                                                                self.all_samples_features.shape[0]))
        # create train data frame - data and labels
        self.X_train = train_samples.drop(['IsGen', 'chrome', 'word_index'], axis=1)
        self.Y_train = train_samples['IsGen']
        # create test data frame data and labels
        self.X_test = test_samples.drop(['IsGen', 'chrome', 'word_index'], axis=1)
        self.Y_test = test_samples['IsGen']

    def create_feature_vector(self):

        print('{}: starting building feature vector for chrome_list: {}'.
              format(time.asctime(time.localtime(time.time())), self.chrome_list))
        all_samples_index = 0  # number of samples (bases) in the train/test data --> index of all_samples_features
        for chrome in self.chrome_list_not_done:
            print('{}: starting building feature vector for chrome: {}'.
                  format(time.asctime(time.localtime(time.time())), chrome))
            # training_file = directory + 'labels150\\chr' + chrome + '_label.csv'
            training_file = directory + 'labels150_non\\chr' + chrome + '_label.csv'
            # seq_labels_file_name = 'C:\\gitprojects\\ML_PROJECT\\first_base_label\\chr' + chrome + '_first_label.xlsx'
            # seq_label = pd.read_excel(seq_labels_file_name, header=None)
            # seq_label_array = seq_label.as_matrix()

            with open(training_file, 'r') as training:
                all_samples_features = []

                sequence_index = 1
                for sequence in training:
                    print('{}: starting working on sequence: {} in chrome {}'.
                          format(time.asctime(time.localtime(time.time())), sequence_index, chrome))

                    word_tag_list = sequence.split(',')

                    while '' in word_tag_list:
                        word_tag_list.remove('')
                        print('while loop with empty')
                    while ' ' in word_tag_list:
                        word_tag_list.remove(' ')
                        print('while loop with space')
                    while '\n' in word_tag_list:
                        word_tag_list.remove('\n')
                        print('while loop with n')
                    while ',' in word_tag_list:
                        word_tag_list.remove(',')
                        print('while loop with ,')
                    if '\n' in word_tag_list[len(word_tag_list) - 1]:
                        word_tag_list[len(word_tag_list) - 1] = word_tag_list[len(word_tag_list) - 1].replace('\n', '')

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

                    for word_in_seq_index, word_tag in enumerate(word_tag_list):
                        # index of all the words in each seq --> for the majority part - test only the first one

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

                        indexes_vector = self.NonStructureFeaturesIndexes.calculate_history_tag_indexes \
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

                        if current_tag in ['1', '2', '3', '4']:  # base is part of a gene
                            current_tag = 1
                        elif current_tag in ['5', '6', '7', '8']:  # base is not part of a gene
                            current_tag = -1
                        else:
                            print('Error: tag for sequence {} in chrome {} not in (1,8)'.format(sequence_index, chrome))
                        indexes_vector = np.append(indexes_vector, current_tag)
                        indexes_vector = np.append(indexes_vector, chrome)
                        indexes_vector = np.append(indexes_vector, str(word_in_seq_index))
                        indexes_vector = list(indexes_vector)
                        all_samples_features.append(indexes_vector)
                        # first base of the seq, if majority: create feature just for that base
                        # if word_index == 0 and majority:
                        #     all_samples_index += 1
                        #     break
                        all_samples_index += 1

                    sequence_index += 1

            headers = [i for i in range(len(indexes_vector) - 3)]
            headers.append('IsGen')
            headers.append('chrome')
            headers.append('word_index')
            # data = np.array(all_samples_features.values())
            all_samples_featuresDF = pd.DataFrame(data=all_samples_features, columns=headers)
            chrome_vector_file_name = directory + 'vectors\\chr' + chrome + '.csv'
            all_samples_featuresDF.to_csv(chrome_vector_file_name, encoding='utf-8', index=False)
            all_samples_featuresDF.drop(all_samples_featuresDF.index, inplace=True)

            # with open(chrome_vector_file_name, "wb") as f:
            #     writer = csv.writer(f)
            #     writer.writerows(all_samples_features)
        print('{}: starting loading feature vectors to create data frame'.format(time.asctime(time.localtime(time.time()))))
        for chrome in self.chrome_list:
            chrome_vector = pd.read_csv(directory + 'vectors\\chr' + chrome + '.csv')
            if chrome == '1':
                all_features = chrome_vector
            else:
                all_features = pd.concat([all_features, chrome_vector], ignore_index=True, axis=0)

        return all_features


def main():
    NonStructureFeatures_perBase_obj = NonStructureFeatures_perBase(majority=False, load_data=False)
    # chorme to use as training data
    chrome_list_train_test = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    for test_chrome in range(1, 17):
        chrome_train_list = [x for x in chrome_list_train_test if x != test_chrome]
        chrome_test_list = [test_chrome]
        logging.info('{}: Train list is: {}, test list is: {}'
                     .format(time.asctime(time.localtime(time.time())), chrome_train_list, chrome_test_list))
        print('{}: Train list is: {}, test list is: {}'
              .format(time.asctime(time.localtime(time.time())), chrome_train_list, chrome_test_list))
        NonStructureFeatures_perBase_obj.create_train_test_dataframes(chrome_train_list, chrome_test_list)
        classifier = Classifier(NonStructureFeatures_perBase_obj, use_CV=False)
        classifier.ModelsIteration()

if __name__ == "__main__":
    logging.getLogger('').handlers = []
    LOG_FILENAME = datetime.now().strftime(directory + 'non_structure\\'
                                           'LogFileNonStructurePerBase_%d_%m_%Y_%H_%M.log')
    logging.basicConfig(filename=LOG_FILENAME, level=logging.INFO)
    # for test_chrome in range(1, 18):
    #     chrome_train_list = [x for x in all_chromes if x != str(test_chrome)]
    #     chrome_test_list = [str(test_chrome)]
    main()
