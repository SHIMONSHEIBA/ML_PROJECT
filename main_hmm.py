from HMM import HMM
from hmm_viterbi import viterbi
import numpy as np
import datetime
import time


def main():
    chrome_number = 1
    train_file = 'C:\gitprojects\ML_PROJECT\\labels\\chr' + chrome_number + '_label.csv'
    print '{}: Start creating HMM'.format((time.asctime(time.localtime(time.time()))))
    hmm = HMM(train_file=train_file)

    print '{}: Start viterbi'.format((time.asctime(time.localtime(time.time()))))
    viterbi_obj = viterbi(hmm)
    viterbi_result = viterbi_obj.run_viterbi_all_data('viterbi')
    viterbiL_result = viterbi_obj.run_viterbi_all_data('viterbiL')

if __name__ == "__main__":
    main()
