import itertools
from itertools import product
import numpy as np

word_tag_dict = {'A': ['1', '5'], 'C': ['2', '6'], 'G': ['3', '7'], 'T': ['4', '8'], '#':['#']}

first_word='A'
second_word = 'G'
current_word = 'A'

possible_tags = [word_tag_dict[first_word], word_tag_dict[second_word], word_tag_dict[current_word]]

for possible_tag_comb in list(product(*possible_tags)):
    print(possible_tag_comb)