from MEMM import MEMM
from viterbi import Viterbi
from gradient import Gradient
from evaluate import Evaluate
import numpy as np
import datetime
import time
import os
import csv
import platform

def main():

    cur_time = time.time()

    if os.name == 'nt':
        train_file = '.\\data\\small_train_1.wtag'
        test_file = '.\\data\\small_test_1.wtag'
        comp_file = '.\\data\\comp.wtag'
        write_file = '.\\data\\test_doc\\test_tag' + str(cur_time)
        confusion_file = '.\\data\\confusion_matrix\\confusion_test_tag' + str(cur_time)
        # gradient_file = '.\\data\\' + str(cur_time) + '.csv'
    else:
        if os.name =='posix':
            train_file = './data/train.wtag'
            test_file = './data/test.wtag'
            comp_file = './data/comp.wtag'
            write_file = './data/test_doc/test_tag' + str(cur_time)
            confusion_file = './data/confusion_matrix/confusion_test_tag' + str(cur_time)
            # gradient_file = './data/' + str(cur_time) + '.csv'
        else:
            print 'undefined os.name'
            return

    evaluate_train_data = True
    evaluate_test_data = True
    write_test_doc = True
    write_confusion_test_doc = True
    feature_list = ['100', '103', '104']
    lamda_accuracy_res = []

    lamda_list = [0.05, 0.1, 0.25, 0.5, 0.75, 1, 2, 3, 4, 5, 10, 20]
    # lamda_list = [1, 2, 4]

    # create memm obj
    memm_obj = MEMM(evaluate_train_data, evaluate_test_data, feature_list, train_file, test_file, comp_file)

    evaluation_result_dict = {}
    evaluation_result_str = {}

    for lamda in lamda_list:

        # create gradient obj and run grad-descent
        gradient_obj = Gradient(memm_obj=memm_obj, PARAM_LAMBDA=lamda)
        gradient_result = gradient_obj.gradientDescent()
        v = gradient_result.x

        # create viterbi obj and run viterbi algorithm
        viterbi_obj = Viterbi(memm_obj, type='test', v=v)
        viterbi_result = viterbi_obj.run_viterbi_on_all_storage()

        # determine path for write files
        if os.name == 'nt':
            confusion_file = '.\\data\\confusion_matrix\\confusion_test_tag_lamda=' + str(lamda) + '_' + str(cur_time) + '_'
            evaluation_file = '.\\data\\confusion_matrix\\evaluation_file_' + str(time.time()) + '.csv'

        elif os.name == 'posix':
            confusion_file = './data/confusion_matrix/confusion_test_tag_lamda=' + str(lamda) + '_' + str(cur_time) + '_'
            evaluation_file = './data/confusion_matrix/evaluation_file_' + str(time.time()) + '.csv'

        # create evaluation obj and perform evaluation
        evaluate_obj = Evaluate(memm_obj, viterbi_result, test_file, write_test_doc, write_confusion_test_doc,
                                write_file, confusion_file)
        evaluate_obj.run()
        evaluate_result = evaluate_obj.eval_res
        evaluation_result_dict[lamda] = evaluate_result
        accuracy_str = '%.5f' % evaluate_result['Accuracy']
        lamda_accuracy_res.append([repr(lamda), accuracy_str])

        print '# Finished lamda = ' + str(lamda)

    print '## results ##'
    # write evaluation results to file

    with open(evaluation_file, 'wb') as f:
        writer = csv.writer(f)
        for lamda_list in lamda_accuracy_res:
            writer.writerows(lamda_list)

if __name__ == "__main__":
    main()