import math
import numpy as np
from scipy.sparse import csr_matrix
from scipy.optimize import minimize

class Gradient(object):

    def __init__(self, memm, lamda):


        self.lamda = lamda
        self.index_of_objective = 1
        self.gradient_iter = 1
        self.memm = memm
        self.history_tag_feature_vector_train = memm.history_tag_feature_vector_train
        self.history_tag_feature_vector_denominator = memm.history_tag_feature_vector_denominator
        self.tags_dict = memm.tags_dict
        self.iteration_counter = 0
        self.w_init = np.zeros(shape=len(memm.features_vector), dtype = int)

    def objectiveFunction(self, v):

        first_part = 0
        second_part = 0
        third_part = 0

        for history_tag, feature_vector in self.history_tag_feature_vector_train.items():

            # 3: 1-to-n v*f_v
            third_part += float(feature_vector.dot(v))

            # 1: 1-to-n log of sum of exp. of v*f(x,y') for all y' in Y
            first_part_inner = 0
            counter_miss_tag = 0
            for tag in self.tags_dict:

                if (history_tag[0], tag) in self.history_tag_feature_vector_denominator:
                    feature_vector_current = self.history_tag_feature_vector_denominator[history_tag[0], tag]
                    cur_res = feature_vector_current.dot(v)
                    if cur_res != 0:
                        stop = 5
                    first_part_inner += math.exp(cur_res)
                else:
                    counter_miss_tag +=1
            first_part += math.log(first_part_inner)

        # 2: L2-norm of v
        second_part += 0.5*pow(np.linalg.norm(v), 2)

        print('objective iter finish')
        print(self.index_of_objective)
        self.index_of_objective+=1
        return first_part + self.lamda*second_part - third_part

    def gradient(self, v):

        #grad_1 = np.zeros_like(v)
        #grad_2 = np.zeros_like(v)

        first_part = csr_matrix(np.zeros_like(v)) #np.zeros_like(v)
        second_part = 0
        third_part = np.copy(v)

        for history_tag, feature_vector in self.history_tag_feature_vector_train.items():
            first_part = np.add(first_part, feature_vector)

        for history_tag, feature_vector in self.history_tag_feature_vector_train.items():

            # save time - calculate only one time exp(v*f(x,y))
            tag_exp_dict = {}
            sum_dict_denominator = 0
            for tag_prime, flag in self.tags_dict.items():
                if (history_tag[0], tag_prime) in self.history_tag_feature_vector_denominator:
                    feature_vector_current = self.history_tag_feature_vector_denominator[history_tag[0], tag_prime]   # history[0] - x vector
                    cur_res = math.exp(feature_vector_current.dot(v))
                    sum_dict_denominator += cur_res
                    tag_exp_dict[tag_prime] = cur_res

            second_part_inner = 0
            for tag_prime, flag in self.tags_dict.items():
                if (history_tag[0], tag_prime) in self.history_tag_feature_vector_denominator:
                    right_var = tag_exp_dict[tag_prime] / sum_dict_denominator
                    second_part_inner = second_part_inner + (self.history_tag_feature_vector_denominator[history_tag[0], tag_prime] * right_var)
            second_part += second_part_inner

        print('gradient iter finish')
        print(self.gradient_iter)
        self.gradient_iter += 1

        first_part = first_part.toarray()
        second_part = second_part.toarray()

        return (- first_part + second_part + self.lamda * third_part).transpose()

    def gradientDescent(self):

        result = minimize(fun=self.objectiveFunction, x0=self.w_init,method='L-BFGS-B',jac=self.gradient,
                          options={'disp': True, 'maxiter': 20, 'factr': 1e2})
        print('finish gradient Descent')
        print(result.x)
        return result