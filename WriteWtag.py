import numpy as np
import scipy
from scipy.sparse import csr_matrix

# import scipy
class Evaluate:

    def __init__(self,memm_obj, viterbi_result, test_file):

        # file names
        self.test_file_name = test_file
        self.viterbi_result = viterbi_result
        self.tag_dict = memm_obj.tag_dict

    def run(self):
        return

    def write_result_to_wtag_file(self, predicted_w_t, real_file):
        return
