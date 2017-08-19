from MEMM import MEMM
from viterbi import Viterbi
from gradient import Gradient
from evaluate import Evaluate
import numpy as np
import datetime
import time
import os
import platform

def main():

    evaluate_train_data = True
    evaluate_test_data = True
    write_test_doc = True
    write_confusion_test_doc = True
    feature_list = ['100', '101', '102', '103', '104', '105']# '106', '107']
    cur_time = str(time.time())

    if os.name == 'nt':
        train_file = '.\\NLP\\data\\train.wtag'
        test_file = '.\\NLP\\data\\test.wtag'
        comp_file = '.\\NLP\\data\\comp.wtag'
        write_file = '.\\NLP\\data\\test_doc\\test_tag' + str(cur_time)
        confusion_file = '.\\NLP\\data\\confusion_matrix\\confusion_test_tag' + str(cur_time)
        gradient_file = '.\\NLP\\data\\' + str(cur_time) + '.csv'
    else:
        if os.name =='posix':
            train_file = './data/train.wtag'
            test_file = './data/test.wtag'
            comp_file = './data/comp.wtag'
            write_file = './data/test_doc/test_tag' + str(cur_time)
            confusion_file = './data/confusion_matrix/confusion_test_tag' + str(cur_time)
            gradient_file = './data/' + str(cur_time) + '.csv'
        else:
            print 'undefined os.name'
            return

    print 'start MEMM'
    print datetime.datetime.now()
    memm_obj = MEMM(evaluate_train_data, evaluate_test_data, feature_list, train_file, test_file, comp_file)

    # PARAM_LAMBDA
    print 'start gradient'
    print datetime.datetime.now()

    gradient_obj = Gradient(memm_obj=memm_obj, PARAM_LAMBDA=1)
    gradient_result = gradient_obj.gradientDescent()
    v=gradient_result.x
    np.savetxt(gradient_file,v, delimiter=",")

    #v = np.loadtxt('C:\\gitprojects\\nlp\\hw1\\data\\1481878095.37.csv')

    print 'finish gradient'
    print datetime.datetime.now()
    #v = np.ones_like(np.arange(memm_obj.feature_vec_type_counter['All']))

    viterbi_obj = Viterbi(memm_obj, type='test', v=v)

    # viterbi_obj return test_file.wtag and dict of sentences list
    #viterbi_obj = Viterbi(memm_obj, type='test', v=gradient_result.x)
    #viterbi_result = viterbi_obj.run_viterbi_algorithm_on_sentence()
    #sentence_1 = memm_obj.test_sentences_storage[0]  # load only 1 sentences

    # TODO wrap all document

    #viterbi_obj.run_viterbi_algorithm_on_sentence(sentence=sentence_1, sentence_index=0)
    print 'start vetrbi'
    viterbi_result = viterbi_obj.run_viterbi_on_all_storage()

    print 'start evaluation'
    # memm_obj, viterbi_result, test_file'''
    #viterbi_result = {0: ['About_*', '400,000_PRP$', 'commuters_VBD'], 1: ['In_*', 'other_PRP$', 'words_VBD', ',_VBG', 'it_PRP$', '._VBD']}
    evaluate_obj = Evaluate(memm_obj, viterbi_result, test_file, write_test_doc, write_confusion_test_doc, write_file, confusion_file)
    evaluate_result = evaluate_obj.run()

    print evaluate_result
    print 'exit main'

    #print viterbi_obj

    #gradient_dcesent_res =  gradient_obj.gradientDescent()

if __name__ == "__main__":
    main()
