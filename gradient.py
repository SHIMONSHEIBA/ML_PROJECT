import numpy as np
from scipy.optimize import minimize
from scipy.sparse import csr_matrix
import math


class Gradient(object):

    def __init__(self, memm_obj, PARAM_LAMBDA):

        self.PARAM_LAMBDA = PARAM_LAMBDA
        self.objective_iter = 1
        self.gradient_iter = 1

        # MEMM Object members
        self.memm_obj = memm_obj
        self.train_f_v = memm_obj.train_f_v
        self.train_f_v_prime = memm_obj.train_f_v_prime
        self.tag_dict = memm_obj.tag_dict
        self.iteration_counter = 0


        # TODO get currect size from guy
        self.v0 = np.zeros_like(np.arange(self.memm_obj.feature_vec_type_counter['All']))

    def objectiveFunction(self, v):

        sum_1 = 0
        sum_2 = 0
        sum_3 = 0

        for history, f_v in self.train_f_v.iteritems():

            # 3: 1-to-n v*f_v
            sum_3 += float(f_v.dot(v))

            # 1: 1-to-n log of sum of exp. of v*f(x,y') for all y' in Y
            sum_1_inner = 0
            counter_miss_tag = 0
            for tag in self.tag_dict:

                if (history[0], tag) in self.train_f_v_prime:
                    f_v_current = self.train_f_v_prime[history[0], tag]
                    cur_res = f_v_current.dot(v)
                    if cur_res != 0:                    # TODO check float casting
                        stop = 5
                    sum_1_inner += math.exp(cur_res)
                else:
                    counter_miss_tag +=1
            sum_1 += math.log(sum_1_inner)

        # 2: L2-norm of v
        sum_2 += 0.5*pow(np.linalg.norm(v), 2)

        print 'objective iter finish'
        print self.objective_iter
        self.objective_iter+=1
        return sum_1 + self.PARAM_LAMBDA*sum_2 - sum_3

    def gradient(self, v):

        #grad_1 = np.zeros_like(v)
        #grad_2 = np.zeros_like(v)

        sum_1 = csr_matrix(np.zeros_like(v)) #np.zeros_like(v)
        sum_2 = 0
        sum_3 = np.copy(v)

        for history, f_v in self.train_f_v.iteritems():     # TODO check if sum not be binary
            sum_1 = np.add(sum_1, f_v)

        for history, f_v in self.train_f_v.iteritems():

            # save time - calculate only one time exp(v*f(x,y))
            tag_exp_dict = {}
            sum_dict_denominator = 0
            for tag_prime, flag in self.tag_dict.iteritems():
                if (history[0], tag_prime) in self.train_f_v_prime:
                    f_v_current = self.train_f_v_prime[history[0], tag_prime]   # history[0] - x vector
                    cur_res = math.exp(f_v_current.dot(v))
                    sum_dict_denominator += cur_res
                    tag_exp_dict[tag_prime] = cur_res

            sum_2_inner = 0
            for tag_prime, flag in self.tag_dict.iteritems():
                if (history[0], tag_prime) in self.train_f_v_prime:
                    right_var = tag_exp_dict[tag_prime] / sum_dict_denominator
                    sum_2_inner = sum_2_inner + (self.train_f_v_prime[history[0], tag_prime] * right_var)
            sum_2 += sum_2_inner

        print 'gradient iter finish'
        print self.gradient_iter
        self.gradient_iter += 1

        sum_1 = sum_1.toarray()
        sum_2 = sum_2.toarray()

        return (- sum_1 + sum_2 + self.PARAM_LAMBDA * sum_3).transpose()  # flip gradient

    def gradientDescent(self):
        res = minimize(fun=self.objectiveFunction, x0=self.v0,
                       method='L-BFGS-B',jac=self.gradient,
                       options={'disp': True, 'maxiter': 15, 'factr': 1e2})
        print 'finish gradient Descent'
        print res.x
        return res