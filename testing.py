from itertools import product

permutations_list = product('ACGT', repeat=7)

permutations_list_one_t = product('ACGT', repeat=6)
permutation_list_one = []
for permutation in permutations_list_one_t:
    permutation_list_one.append(''.join(permutation) + '#')
    permutation_list_one.append('#' + ''.join(permutation))
del (permutations_list_one_t)

permutations_list_two_t = product('ACGT', repeat=5)
permutation_list_two = []
for permutation in permutations_list_two_t:
    permutation_list_two.append(''.join(permutation) + '##')
    permutation_list_two.append('##' + ''.join(permutation))
del (permutations_list_two_t)

permutation_list_one += permutation_list_two
del (permutation_list_two)
for permutetion in permutations_list:
    permutation_list_one.append(permutetion)
del (permutations_list)

print(len(permutation_list_one))



