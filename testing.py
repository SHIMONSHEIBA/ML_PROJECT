
import itertools
features_combination_list_sub = ['feature_word_tag', 'feature_word', 'feature_tag', 'feature_1', 'feature_2',
                             'feature_3', 'feature_4', 'feature_5', 'feature_6', 'feature_7',
                             'feature_8']

print(''.join(features_combination_list_sub))

#features_combination_list = []
#for perm in itertools.combinations(features_combination_list_sub, 5):
#    features_combination_list.append(list(perm))
#for perm in itertools.combinations(features_combination_list_sub, 6):
#    features_combination_list.append(list(perm))
#for perm in itertools.combinations(features_combination_list_sub, 7):
#    features_combination_list.append(list(perm))
#for perm in itertools.combinations(features_combination_list_sub, 8):
#    features_combination_list.append(list(perm))
#for perm in itertools.combinations(features_combination_list_sub, 9):
#    features_combination_list.append(list(perm))
#for perm in itertools.combinations(features_combination_list_sub, 10):
#    features_combination_list.append(list(perm))
#print(features_combination_list)


#from itertools import chain, combinations
#
#def powerset(iterable):
#    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
#    s = list(iterable)
#    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))
#
#for result in powerset([1, 2, 3]):
#    print(result)

#results = list(powerset([1, 2, 3]))
#print(results)

