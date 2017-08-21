import itertools
from itertools import product
import numpy as np

word_tag_dict = {'A': ['1', '5'], 'C': ['2', '6'], 'G': ['3', '7'], 'T': ['4', '8'], '#':['#']}

feature_vector_length = 158

print(np.zeros(shape=len(word_tag_dict), dtype = int))

#print(np.zeros_like(np.arange(6)))


#print(np.arange(3))