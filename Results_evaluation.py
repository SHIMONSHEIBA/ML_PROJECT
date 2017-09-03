import csv
import time


directory = 'C:\\gitprojects\\ML_PROJECT\\'

class ResultsEvaluation:
    def __init__(self):
        self.chrome_list = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17']
        self.chrome_info = {k: [0] * 17 for k in range(1, len(self.chrome_list)+1)}
        for chrome_index, chrome in enumerate(self.chrome_list):
            self.data_chrome = {}
            self.results_chrome = {}
            self.create_dictionary(chrome)
            for sequence_index in range(len(self.data_chrome.keys())):  # go over all the sequences in the chrome
                wrong_on_first, was_gene_now_no_gene, was_gene_now_gene, wasnt_gene_now_gene, wasnt_gene_now_no_gene,\
                number_of_good_switches, number_of_wrong_switches, total_number_of_switches, index_first_wrong,\
                index_last_correct, index_last_wrong, number_of_miss, first_wrong_on_switch, predict_both,\
                first_correct_on_switch = self.switch_on_wrong_and_first(sequence_index)
                sequence_lenght = min(len(self.data_chrome[sequence_index]), len(self.results_chrome[sequence_index]))
                # miss1: wrong on first base and all the seq
                if wrong_on_first and sequence_lenght == number_of_miss:
                    self.chrome_info[chrome_index+1][0] += 1
                # miss2: right on the first and first mistake where there was switch
                elif not wrong_on_first and number_of_miss > 0 and first_wrong_on_switch:
                    self.chrome_info[chrome_index+1][1] += 1
                # miss3: wrong on the first and first correct where there was switch
                elif wrong_on_first and number_of_miss < sequence_lenght and first_correct_on_switch:
                    self.chrome_info[chrome_index+1][2] += 1
                # miss4: start being wrong in the middle of the seq and then was correct:
                if 0 < index_first_wrong < index_last_wrong < index_last_correct and predict_both:
                    self.chrome_info[chrome_index+1][3] += 1
                # miss5: predict all the seq the same, but there are more than one switches so he correct and than wrong
                # and than correct again, or he wrong and than correct and then wrong again
                if 0 < index_first_wrong < index_last_wrong < index_last_correct and not predict_both:
                    self.chrome_info[chrome_index+1][4] += 1
                # predict all seq correct and there was switch - only one switch
                if predict_both and number_of_good_switches == 1 \
                        and number_of_good_switches == total_number_of_switches:
                    self.chrome_info[chrome_index + 1][5] += 1
                # predict all seq correct and there was switch- more than one switch
                if predict_both and number_of_good_switches > 1 \
                        and number_of_good_switches == total_number_of_switches:
                    self.chrome_info[chrome_index + 1][6] += 1
                # predict some of the switches good
                if predict_both and number_of_good_switches > 0 and number_of_good_switches != total_number_of_switches:
                    self.chrome_info[chrome_index + 1][7] += 1
                # start insert all the other numbers
                insert_index = 8
                self.chrome_info[chrome_index+1][insert_index] += was_gene_now_no_gene
                insert_index += 1
                self.chrome_info[chrome_index+1][insert_index] += was_gene_now_gene
                insert_index += 1
                self.chrome_info[chrome_index+1][insert_index] += wasnt_gene_now_gene
                insert_index += 1
                self.chrome_info[chrome_index+1][insert_index] += wasnt_gene_now_no_gene
                insert_index += 1
                self.chrome_info[chrome_index+1][insert_index] += number_of_good_switches
                insert_index += 1
                self.chrome_info[chrome_index+1][insert_index] += number_of_wrong_switches
                insert_index += 1
                self.chrome_info[chrome_index+1][insert_index] += total_number_of_switches
                insert_index += 1
                if predict_both:  # predict both gene and non gene in the same sequence
                    self.chrome_info[chrome_index+1][insert_index] += 1
                insert_index += 1
                if number_of_miss == 0:
                    self.chrome_info[chrome_index + 1][insert_index] += 1
            print('{}: evaluation numbers for chrome: {} are: {}'.format(time.asctime(time.localtime(time.time())),
                                                                         chrome, self.chrome_info[chrome_index+1]))
        write_file_name = directory + '\\evaluation1.csv'
        with open(write_file_name, 'ab') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(['miss1', 'miss2', 'miss3', 'miss4', 'miss5', 'predict_all_switches_good_one_switch',
                             'predict_all_switches_good_more_one_switch', 'predict_some_of_the_switches_good',
                             'was_gene_now_no_gene', 'was_gene_now_gene', 'wasnt_gene_now_gene',
                             'wasnt_gene_now_no_gene', 'number_of_good_switches', 'number_of_wrong_switches',
                             'total_number_of_switches', 'predict_both', 'all_correct', 'chrome'])
            for key, value in self.chrome_info.items():
                value.append(key)
                writer.writerow(value)

    def create_dictionary(self, chrome):
        print '{}: Start upload chrome: {}'.format(time.asctime(time.localtime(time.time())), chrome)
        files = [[directory + 'labels150\\chr' + chrome + '_label.csv', 'data'],
                 [directory + 'results\\chr' + chrome + '_result.csv', 'results']]
        for file_type in files:
            file_name = file_type[0]
            type_name = file_type[1]
            with open(file_name, 'r') as file_open:
                sequence_index = 0
                for sequence in file_open:
                    word_tag_list = sequence.split(',')
                    # handel , in the end of the sequence:
                    if '\n' in word_tag_list[len(word_tag_list) - 1]:
                        word_tag_list[len(word_tag_list) - 1] = word_tag_list[len(word_tag_list) - 1].replace('\n', '')
                    while ' ' in word_tag_list:
                        word_tag_list.remove(' ')
                    while '' in word_tag_list:
                        word_tag_list.remove('')
                    while '\n' in word_tag_list:
                        word_tag_list.remove('\n')
                    if type_name == 'data':
                        self.data_chrome[sequence_index] = word_tag_list
                    elif type_name == 'results':
                        self.results_chrome[sequence_index] = word_tag_list
                    sequence_index += 1

    def switch_on_wrong_and_first(self, sequence_index):
        gene_list = ['1', '2', '3', '4']
        no_gene_list = ['5', '6', '7', '8']
        was_gene_now_no_gene = 0
        was_gene_now_gene = 0
        wasnt_gene_now_gene = 0
        wasnt_gene_now_no_gene = 0
        last_tag = ''
        last_predict = ''
        number_of_good_switches = 0
        number_of_wrong_switches = 0
        wrong_on_first = False
        index_first_wrong = -1
        index_last_wrong = 0
        index_last_correct = 0
        miss_number = 0
        first_wrong_on_switch = False
        first_correct_on_switch = False
        predict_gene = False
        predict_non_gene = False
        predict_both = False
        seq_lenght = min(len(self.data_chrome[sequence_index]), len(self.results_chrome[sequence_index]))
        if len(self.data_chrome[sequence_index]) != len(self.results_chrome[sequence_index]):
            print('Error: lenght of seq is different in sequence number: {}'.format(sequence_index))
        for pos in range(seq_lenght):
            real_pos_tag = self.data_chrome[sequence_index][pos]
            predict_pos_tag = self.results_chrome[sequence_index][pos]
            if real_pos_tag != predict_pos_tag:  # start to predict wrong
                index_last_wrong = pos
                miss_number += 1
                if pos == 0:
                    wrong_on_first = True
                    index_first_wrong = 0
                elif index_first_wrong == -1:
                    index_first_wrong = pos
                tag = real_pos_tag.split('_')[1]
                predict_tag = predict_pos_tag.split('_')[1]
                if predict_tag in gene_list:
                    predict_gene = True
                elif predict_tag in no_gene_list:
                    predict_non_gene = True
                if tag in gene_list and last_tag in gene_list and pos != 0:  # was gene, now gene
                    was_gene_now_gene += 1
                elif tag in gene_list and last_tag in no_gene_list and pos != 0:  # wasn't gene, now gene
                    wasnt_gene_now_gene += 1
                    # there was a switch and no switch in the prediction
                    if last_predict in no_gene_list and predict_tag in no_gene_list:
                        number_of_wrong_switches += 1
                    if index_first_wrong == pos:
                        first_wrong_on_switch = True
                elif tag in no_gene_list and last_tag in gene_list and pos != 0:  # was gene, now no gene
                    was_gene_now_no_gene += 1
                    # there was a switch and no switch in the prediction
                    if last_predict in gene_list and predict_tag in gene_list:
                        number_of_wrong_switches += 1
                    if index_first_wrong == pos:
                        first_wrong_on_switch = True
                elif tag in no_gene_list and last_tag in no_gene_list and pos != 0:  # wasn't gene, now no gene
                    wasnt_gene_now_no_gene += 1
                else:
                    if pos != 0:
                        print('Error: tag is: {}, last result is: {}').format(tag, last_tag)
                last_tag = tag
                last_predict = predict_tag

            elif real_pos_tag == predict_pos_tag:  # predict ok
                index_last_correct = pos
                if index_last_correct - index_last_wrong == 1:  # first time he is correct
                    first_correct_on_switch = True
                tag = real_pos_tag.split('_')[1]
                predict_tag = predict_pos_tag.split('_')[1]
                if predict_tag in gene_list:
                    predict_gene = True
                elif predict_tag in no_gene_list:
                    predict_non_gene = True
                if tag in gene_list and last_tag in no_gene_list and pos != 0:  # wasn't gene, now gene
                    # there was a switch and no switch in the prediction
                    if last_predict in gene_list and predict_tag in gene_list:
                        number_of_wrong_switches += 1
                    # there was a switch and the model switch the prediction
                    elif last_predict in no_gene_list and predict_tag in gene_list:
                        number_of_good_switches += 1
                elif tag in no_gene_list and last_tag in gene_list and pos != 0:  # was gene, now no gene
                    # there was a switch and no switch in the prediction
                    if last_predict in no_gene_list and predict_tag in no_gene_list:
                        number_of_wrong_switches += 1
                    # there was a switch and the model switch the prediction
                    elif last_predict in gene_list and predict_tag in no_gene_list:
                        number_of_good_switches += 1
                last_tag = tag
                last_predict = predict_tag

        total_number_of_switches = number_of_good_switches + number_of_wrong_switches
        if predict_gene and predict_non_gene:
            predict_both = True

        return wrong_on_first, was_gene_now_no_gene, was_gene_now_gene, wasnt_gene_now_gene, wasnt_gene_now_no_gene,\
               number_of_good_switches, number_of_wrong_switches, total_number_of_switches, index_first_wrong,\
               index_last_correct, index_last_wrong, miss_number, first_wrong_on_switch, predict_both,\
               first_correct_on_switch

def main():
    ResultsEvaluation_class = ResultsEvaluation()


if __name__ == '__main__':
    main()
