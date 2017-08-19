import time



class MEMM:
    """ Base class of modeling MEMM logic on the data"""

    # shared among all instances of the class'
    amino_mapping = {'TTT' : 'Phe','TTC' : 'Phe','TTA' : 'Leu','TTG' : 'Leu','CTT' : 'Leu','CTC' : 'Leu',
                     'CTA' : 'Leu','CTG' : 'Leu','ATT' : 'Ile','ATC' : 'Ile','ATA' : 'Ile','ATG' : 'Met',
                     'GTT' : 'Val','GTC' : 'Val','GTA' : 'Val','GTG' : 'Val','TCT' : 'Ser','TCC' : 'Ser',
                     'TCA' : 'Ser','TCG' : 'Ser','CCT' : 'Pro','CCC' : 'Pro','CCA' : 'Pro','CCG' : 'Pro',
                     'ACT' : 'Thr','ACC' : 'Thr','ACA' : 'Thr','ACG' : 'Thr','GCT' : 'Ala','GCC' : 'Ala',
                     'GCA' : 'Ala','GCG' : 'Ala','TAT' : 'Tyr','TAC' : 'Tyr','TAA' : 'stop','TAG' : 'stop',
                     'CAT' : 'His','CAC' : 'His','CAA' : 'Gin','CAG' : 'Gin','AAT' : 'Asn','AAC' : 'Asn',
                     'AAA' : 'Lys','AAG' : 'Lys','GAT' : 'Asp','GAC' : 'Asp','GAA' : 'Glu','GAG' : 'Glu',
                     'TGT' : 'Cys','TGC' : 'Cys','TGA' : 'stop','TGG' : 'Trp','CGT' : 'Arg','CGC' : 'Arg',
                     'CGA' : 'Arg','CGG' : 'Arg','AGT' : 'Ser','AGC' : 'Ser','AGA' : 'Arg','AGG' : 'Arg',
                     'GGT' : 'Gly','GGC' : 'Gly','GGA' : 'Gly','GGG' : 'Gly' }

    stop_keys = ['TGA','TAA', 'TAG']

    start_keys = ['ATG']

    tags_dict = {'1' : [0,'A+'] , '2' : [0,'C+'] ,'3' : [0,'G+'] , '4' : [0,'T+'] ,'5' : [0,'A-'] , '6' : [0,'C-'] ,
                 '7' : [0,'G-'] , '8' : [0,'T-'] }

    words_dict = { 'A' : 0 , 'T' : 0 , 'C' : 0 , 'G' : 0 }

    def __init__(self, trainingfile):

        self.training_file = trainingfile
        self.feature_1 = {}
        self.feature_2 = {}
        self.feature_3 = {}
        self.feature_4 = {}
        self.feature_5 = {}
        self.feature_6 = {}
        self.feature_7 = {}
        self.feature_8 = {}

        self.build_features_from_train()


    def build_features_from_train(self):

        start_time = time.time()
        print('starting building features from train')

        with open(self.training_file) as training:

            sequence_index = 1
            for sequence in training:

                word_tag_list = sequence.split(',')

                print("working on sequence {} :".format(sequence_index))
                print(word_tag_list)

                # define two first word_tags for some features
                first_tag = '#'
                second_tag = '#'

                zero_word= ''
                first_word = ''
                second_word = ''
                plus_one_word = ''
                plus_two_word = ''
                plus_three_word = ''

                for word_in_seq_index , word_tag in enumerate(word_tag_list):

                    word_tag_tuple = word_tag.split('_')

                    if '\n' in word_tag_tuple[1]:
                        word_tag_tuple[1] = word_tag_tuple[1][:1]

                    # count number of instances for each word in train set
                    self.words_dict[word_tag_tuple[0]] += 1

                    # count number of instances for each tag in train set
                    self.tags_dict[word_tag_tuple[1]][0] += 1

                    current_word = word_tag_tuple[0]
                    current_tag = word_tag_tuple[1]

                    feature_3_key = ''
                    feature_4_key = ''

                    # build feature_1 of three tags instances
                    feature_1_key = first_tag + second_tag + current_tag
                    if feature_1_key not in self.feature_1:
                        self.feature_1[feature_1_key] = 1
                    else:
                        self.feature_1[feature_1_key] += 1

                    # build feature_2 of two tags instances
                    feature_2_key = second_tag + current_tag
                    if feature_2_key not in self.feature_2:
                        self.feature_2[feature_2_key] = 1
                    else:
                        self.feature_2[feature_2_key] += 1

                    if word_in_seq_index > 1:
                        first_word = word_tag_list[word_in_seq_index-2][0]
                        second_word = word_tag_list[word_in_seq_index-1][0]
                        feature_3_key = first_word + second_word + current_word
                        feature_4_key =  self.amino_mapping[feature_3_key]

                    # build feature_3 of three words instances
                    if feature_3_key:
                        if feature_3_key not in self.feature_3:
                            self.feature_3[feature_3_key] = 1
                        else:
                            self.feature_3[feature_3_key] += 1

                    # build feature_4 of amino acids instances
                    if feature_4_key:
                        if feature_4_key not in self.feature_4:
                            self.feature_4[feature_4_key] = 1
                        else:
                            self.feature_4[feature_4_key] += 1

                    # build feature_5 of stop codon before current word
                    if word_in_seq_index > 2:
                        zero_word = word_tag_list[word_in_seq_index-3][0]
                        #first_word = word_tag_list[word_in_seq_index-2][0]
                        #second_word = word_tag_list[word_in_seq_index-1][0]
                        feature_5_key = zero_word + first_word + second_word
                        if feature_5_key in self.stop_keys:
                            if feature_5_key not in self.feature_5:
                                self.feature_5[feature_5_key] = 1
                            else:
                                self.feature_5[feature_5_key] += 1

                    # build feature_6 of stop codon after current word
                    if len(word_tag_list)-word_in_seq_index > 3:
                        plus_one_word = word_tag_list[word_in_seq_index + 1][0]
                        plus_two_word = word_tag_list[word_in_seq_index + 2][0]
                        plus_three_word = word_tag_list[word_in_seq_index + 3][0]
                        feature_6_key = plus_one_word + plus_two_word + plus_three_word
                        if feature_6_key in self.stop_keys:
                            if feature_6_key not in self.feature_6:
                                self.feature_6[feature_6_key] = 1
                            else:
                                self.feature_6[feature_6_key] += 1
                    # TODO: validate words are updated per itaration
                    # build feature_7 of start codon before current word
                    if word_in_seq_index > 2:
                        #zero_word = word_tag_list[word_in_seq_index - 3][0]
                        # first_word = word_tag_list[word_in_seq_index-2][0]
                        # second_word = word_tag_list[word_in_seq_index-1][0]
                        feature_7_key = zero_word + first_word + second_word
                        if feature_7_key in self.start_keys:
                            if feature_7_key not in self.feature_7:
                                self.feature_7[feature_7_key] = 1
                            else:
                                self.feature_7[feature_7_key] += 1

                    # build feature_8 of start codon after current word
                    if len(word_tag_list) - word_in_seq_index > 3:
                        #plus_one_word = word_tag_list[word_in_seq_index + 1][0]
                        #plus_two_word = word_tag_list[word_in_seq_index + 2][0]
                        #plus_three_word = word_tag_list[word_in_seq_index + 3][0]
                        feature_8_key = plus_one_word + plus_two_word + plus_three_word
                        if feature_8_key in self.start_keys:
                            if feature_8_key not in self.feature_8:
                                self.feature_8[feature_8_key] = 1
                            else:
                                self.feature_8[feature_8_key] += 1


                    # update words
                    #zero_word = first_word
                    #first_word = second_word
                    #second_word = current_word
                    #plus_one_word = plus_two_word
                    #plus_two_word = plus_three_word

                    # update tags
                    first_tag = second_tag
                    second_tag = current_tag
                sequence_index += 1

        print('finished building features in : {}'.format(time.time()-start_time))
        return


if __name__ == '__main__':


    MEMM = MEMM("C:\\Users\\shimo\\Desktop\\STRUCTURED_PREDICTION\\ML_PROJECT\\training_example.csv")
