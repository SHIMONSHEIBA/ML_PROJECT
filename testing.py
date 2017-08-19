

features_vector = {}
features_vector_mapping = {}
index = 0

word_tag_dict = {'A': ['1', '5'], 'C': ['2', '6'], 'G': ['3', '7'], 'T': ['4', '8']}
for word, tag_list in word_tag_dict.items():
    for tag in tag_list:
        key = word + '_' + tag
        features_vector[key] = index
        features_vector_mapping[index] = key
        index += 1


print(features_vector)
print(features_vector_mapping)



