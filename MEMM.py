import numpy as np
import operator
import scipy
from scipy.sparse import csr_matrix
import datetime

# import scipy
class MEMM:

    def __init__(self, evaluate_train, evaluate_test, feature_list, train_file, test_file, comp_file):

        # file names
        # self.train_file_name = 'C:\\gitprojects\\nlp\\hw1\\data\\train.wtag'
        #self.test_file_name = 'C:\\gitprojects\\nlp\\hw1\\data\\small_test.wtag'
        # self.test_file_name = 'C:\\gitprojects\\nlp\\hw1\\data\\test.wtag'

        self.train_file_name = train_file
        self.test_file_name = test_file
        self.competition_file_name = comp_file

        # class variables
        self.word_dictionary_count = {}     # all trainings word, and their tags
        self.tag_dict = {}                  # all tags seen in training data
        self.common_tag_list = []           # top 5 tags in train data

        self.f_101 = {}                     # Ratnaparkhi feature number 101 - 3 start char
        self.f_102 = {}                     # Ratnaparkhi feature number 102 - 3 last char
        self.f_103 = {}                     # Ratnaparkhi feature number 103
        self.f_104 = {}                     # Ratnaparkhi feature number 104 - last tag
        self.f_105 = {}                     # Ratnaparkhi feature number 105 - two last tag
        self.f_106 = {}                     # Ratnaparkhi feature number 106 - last word and current tag
        self.f_107 = {}                     # Ratnaparkhi feature number 107 - next word and current tag


        self.feature_vec_meta = {}          # feature key and his idx
        self.feature_vec_i_tag_meta = {}    # feature idx key and his feature description
        self.feature_vec_type_counter = {}  # counter of each feature type 100,103 e.g.
        self.sentences_storage = {}         # all sentences mapping by their place in training data
        self.unseen_dict = {}
        self.seen_dict = {}

        self.test_sentences_storage = {}
        self.comp_sentences_storage = {}

        self.train_f_v = {}             # train data - feature vector for all word and their known tags
        self.test_f_v = {}
        self.train_f_v_prime = {}       # train data - feature vector for all word and all y'
        self.test_f_v_prime = {}        # test data - feature vector for all word and all y'

        self.miss_100 = 0
        self.miss_101 = 0
        self.miss_102 = 0
        self.miss_103 = 0
        self.miss_104 = 0
        self.miss_105 = 0
        self.miss_106 = 0
        self.miss_107 = 0
        # TODO: add feature - prefix, number classes (garbage collector)

        # main function
        self.feature_using = feature_list           # e.g. ['100', '103' ,'104']

        self.create_word_dict()                     # create word_tag dict. understand training data
        self.find_most_common_tags()
        self.determine_entrance_feature_vector()    # create structure of feature vector

        if (evaluate_train):
            self.create_train_feature_vector(self.sentences_storage)      # generate f_v for each train data words
            print 'start train prime'

            self.create_train_feature_vector_prime_reduce(self.sentences_storage)
            print 'end train prime'

        if (evaluate_test):
            self.create_relevant_storage(self.test_file_name, self.test_sentences_storage)
            print datetime.datetime.now()
            self.create_test_feature_vector(self.test_sentences_storage)       # generate f_v for each train data words
            print datetime.datetime.now()
            print 'start test prime'

            self.create_test_feature_vector_prime(self.test_sentences_storage)

        print 'self.miss_100'
        print self.miss_100
        print 'self.miss_101'
        print self.miss_101
        print 'self.miss_102'
        print self.miss_102
        print 'self.miss_103'
        print self.miss_103
        print 'self.miss_104'
        print self.miss_104
        print 'self.miss_105'
        print self.miss_105
        print 'self.miss_106'
        print self.miss_106
        print 'self.miss_107'
        print self.miss_107
        print 'finished building MEMM'
        # self.eval_test_data()
        # self.create_sentence_vector()

    def create_word_dict(self):
        # TODO remove duplication from last word in sentences case
        idx_sentences = 0

        with open(self.train_file_name) as fd:
            for line in fd:

                sentence_list = line.split(' ')
                #print sentence_list
                m_2_t = '*'
                m_1_t = '*'
                prev_word = ''
                next_word = ''

                cur_sentences = []

                for i, val in enumerate(sentence_list):

                    w_t_arr = val.split('_')

                    if '\n' in val:     # end of sentence - skip (TODO analyze last word - 'STOP')

                        cur_sentences.append(w_t_arr[0])
                        w_t_arr[1] = w_t_arr[1][:-1]  # 'STOP'

                        if w_t_arr[1] not in self.tag_dict:
                            self.tag_dict[w_t_arr[1]] = 1
                        else:
                            self.tag_dict[w_t_arr[1]] += 1

                        m_0_t = w_t_arr[1]  # STOP
                        key_3_t = m_2_t + '_' + m_1_t + '_' + m_0_t     # last 3 tags
                        key_2_t = m_1_t + '_' + m_0_t
                        key_1_t = m_0_t
                        prev_word = sentence_list[i - 1].split('_')[0]
                        key_f_106 = prev_word + '_' + m_0_t


                        if '101' in self.feature_using:
                            # get suffix
                            if len(w_t_arr[0]) > 2:
                                suf_3_w = w_t_arr[0][-3:]
                                key_suf_3_w = suf_3_w + '_' + m_0_t  # create key
                                if key_suf_3_w not in self.f_101:
                                    self.f_101[key_suf_3_w] = 1
                                else:
                                    self.f_101[key_suf_3_w] += 1

                        if '102' in self.feature_using:
                            # get prefixes
                            if len(w_t_arr[0]) > 2:
                                pre_3_w = w_t_arr[0][:3]
                                key_pre_3_w = pre_3_w + '_' + m_0_t  # create key
                                if key_pre_3_w not in self.f_102:
                                    self.f_102[key_pre_3_w] = 1
                                else:
                                    self.f_102[key_pre_3_w] += 1

                        if '103' in self.feature_using:
                            if key_3_t not in self.f_103:
                                self.f_103[key_3_t] = 1
                            else:
                                self.f_103[key_3_t] += 1

                        if '104' in self.feature_using:
                            if key_2_t not in self.f_104:
                                self.f_104[key_2_t] = 1
                            else:
                                self.f_104[key_2_t] += 1

                        if '105' in self.feature_using:
                            if key_1_t not in self.f_105:
                                self.f_105[key_1_t] = 1
                            else:
                                self.f_105[key_1_t] += 1

                        if '106' in self.feature_using:
                            # prev_word + '_' + m_0_t
                            if prev_word != '':
                                if key_f_106 not in self.f_106:
                                    self.f_106[key_f_106] = 1
                                else:
                                    self.f_106[key_f_106] += 1

                        # f_107 not relevant here, next word is out of range


                        # build word dictionary
                        if w_t_arr[0] in self.word_dictionary_count:
                            self.word_dictionary_count[w_t_arr[0]]['cnt'] += 1
                        else:
                            self.word_dictionary_count[w_t_arr[0]] = {}
                            self.word_dictionary_count[w_t_arr[0]]['cnt'] = 1
                        self.word_dictionary_count[w_t_arr[0]][w_t_arr[1]] = 1

                    else:
                        cur_sentences.append(w_t_arr[0])
                        if w_t_arr[1] not in self.tag_dict:
                            self.tag_dict[w_t_arr[1]] = 1
                        else:
                            self.tag_dict[w_t_arr[1]]+=1

                        # tag occurrence dictionary

                        m_0_t = w_t_arr[1]
                        key_3_t = m_2_t + '_' + m_1_t + '_' + m_0_t
                        key_2_t = m_1_t + '_' + m_0_t
                        key_1_t = m_0_t

                        if '101' in self.feature_using:
                            # get suffix
                            if len(w_t_arr[0]) > 2:
                                suf_3_w = w_t_arr[0][-3:]
                                key_suf_3_w = suf_3_w + '_' + m_0_t  # create key
                                if key_suf_3_w not in self.f_101:
                                    self.f_101[key_suf_3_w] = 1
                                else:
                                    self.f_101[key_suf_3_w] += 1

                        if '102' in self.feature_using:
                            # get prefixes
                            if len(w_t_arr[0]) > 2:
                                pre_3_w = w_t_arr[0][:3]
                                key_pre_3_w = pre_3_w + '_' + m_0_t  # create key
                                if key_pre_3_w not in self.f_102:
                                    self.f_102[key_pre_3_w] = 1
                                else:
                                    self.f_102[key_pre_3_w] += 1

                        if '103' in self.feature_using:
                            if key_3_t not in self.f_103:
                                self.f_103[key_3_t] = 1
                            else:
                                self.f_103[key_3_t] += 1

                        if '104' in self.feature_using:
                            if key_2_t not in self.f_104:
                                self.f_104[key_2_t] = 1
                            else:
                                self.f_104[key_2_t] += 1

                        if '105' in self.feature_using:
                            if key_1_t not in self.f_105:
                                self.f_105[key_1_t] = 1
                            else:
                                self.f_105[key_1_t] += 1

                        if '106' in self.feature_using:
                            # prev_word + '_' + m_0_t
                            if i > 0 :
                                prev_word = sentence_list[i - 1].split('_')[0]
                                key_f_106 = prev_word + '_' + m_0_t
                                if key_f_106 not in self.f_106:
                                    self.f_106[key_f_106] = 1
                                else:
                                    self.f_106[key_f_106] += 1

                        if '107' in self.feature_using:
                            if (i+1!=len(sentence_list)):
                                next_word = sentence_list[i + 1].split('_')[0]  # insurance exist, i is not last word
                                key_f_107 = next_word + '_' + m_0_t
                                if key_f_107 not in self.f_107:
                                    self.f_107[key_f_107] = 1
                                else:
                                    self.f_107[key_f_107] += 1

                        # update currents tag
                        m_2_t = m_1_t
                        m_1_t = m_0_t

                        # build word dictionary
                        if w_t_arr[0] in self.word_dictionary_count:
                            self.word_dictionary_count[w_t_arr[0]]['cnt'] += 1
                        else:
                            self.word_dictionary_count[w_t_arr[0]] = {}
                            self.word_dictionary_count[w_t_arr[0]]['cnt'] = 1
                        self.word_dictionary_count[w_t_arr[0]][w_t_arr[1]] = 1

                self.sentences_storage[idx_sentences] = cur_sentences
                idx_sentences += 1

        return

    def find_most_common_tags(self):
        tmp_list = sorted(self.tag_dict.items(), key=lambda x: x[1])
        tmp_list.reverse()
        cut_top_5 = 0
        while cut_top_5<5:
            self.common_tag_list.append(tmp_list[cut_top_5][0])
            cut_top_5+=1
        print self.common_tag_list
        self.common_tag_list = ['NN', 'IN', 'NNP', 'CD', 'NNS', 'JJ']
        print self.common_tag_list
        return

    def determine_entrance_feature_vector(self):

        # feature_vec_meta
        # add special tags to tg _dict -> TODO insert in init
        self.tag_dict['*'] = 1
        self.tag_dict['STOP'] = 1

        f_c = 0
        f_c_100 = 0
        f_c_101 = 0
        f_c_102 = 0
        f_c_103 = 0
        f_c_104 = 0
        f_c_105 = 0
        f_c_106 = 0
        f_c_107 = 0

        # f.100
        # only word_tag which exist in training
        if '100' in self.feature_using:
            for word, dic in self.word_dictionary_count.iteritems():
                for tag, num in dic.iteritems():
            #    for tag, num in self.tag_dict.iteritems():     # if all word-tag combination needed
                    if tag=='cnt':      # skip counter
                        continue
                    key = word + '_' + tag
                    self.feature_vec_i_tag_meta[f_c] = key
                    self.feature_vec_meta[key] = f_c
                    f_c += 1
                    f_c_100 += 1

        # f.101
        if '101' in self.feature_using:
            for key_3_suf, amount in self.f_101.iteritems():
                if amount > 3:
                    self.feature_vec_i_tag_meta[f_c] = key_3_suf
                    self.feature_vec_meta[key_3_suf] = f_c
                    f_c += 1
                    f_c_101 += 1

        # f.102
        if '102' in self.feature_using:
            for key_3_pre, amount in self.f_102.iteritems():
                if amount > 3:
                    self.feature_vec_i_tag_meta[f_c] = key_3_pre
                    self.feature_vec_meta[key_3_pre] = f_c
                    f_c += 1
                    f_c_102 += 1

        # f.103
        if '103' in self.feature_using:
            for key_3_tag, amount in self.f_103.iteritems():
                self.feature_vec_i_tag_meta[f_c] = key_3_tag
                self.feature_vec_meta[key_3_tag] = f_c
                f_c += 1
                f_c_103 += 1

        # f.104
        if '104' in self.feature_using:
            for key_2_tag, amount in self.f_104.iteritems():
                self.feature_vec_i_tag_meta[f_c] = key_2_tag
                self.feature_vec_meta[key_2_tag] = f_c
                f_c += 1
                f_c_104 += 1

        # f.105
        if '105' in self.feature_using:
            for key_1_tag, amount in self.f_105.iteritems():
                self.feature_vec_i_tag_meta[f_c] = key_1_tag
                self.feature_vec_meta[key_1_tag] = f_c
                f_c += 1
                f_c_105 += 1

        # f.106
        if '106' in self.feature_using:
            for prev_word_cur_tag, amount in self.f_106.iteritems():
                self.feature_vec_i_tag_meta[f_c] = prev_word_cur_tag
                self.feature_vec_meta[prev_word_cur_tag] = f_c
                f_c += 1
                f_c_106 += 1

        # f.107
        if '107' in self.feature_using:
            for prev_word_cur_tag, amount in self.f_107.iteritems():
                self.feature_vec_i_tag_meta[f_c] = prev_word_cur_tag
                self.feature_vec_meta[prev_word_cur_tag] = f_c
                f_c += 1
                f_c_107 += 1

        self.feature_vec_type_counter['All'] = f_c
        self.feature_vec_type_counter['100'] = f_c_100
        self.feature_vec_type_counter['101'] = f_c_101
        self.feature_vec_type_counter['102'] = f_c_102
        self.feature_vec_type_counter['103'] = f_c_103
        self.feature_vec_type_counter['104'] = f_c_104
        self.feature_vec_type_counter['105'] = f_c_105
        self.feature_vec_type_counter['106'] = f_c_106
        self.feature_vec_type_counter['107'] = f_c_107
        print 'self.feature_vec_type_counter'
        print self.feature_vec_type_counter
        return f_c

    def create_train_feature_vector(self, storage):

        idx_sentences = 0

        with open(self.train_file_name) as fd:
            for line in fd:
                # new sentences generate vectors
                sentence_list = line.split(' ')
                m_2_t = '*'
                m_1_t = '*'
                cur_sentences = []
                for word_i, val in enumerate(sentence_list):
                    w_t_arr = val.split('_')
                    if '\n' in val:             # end of sentences
                        w_t_arr[1] = w_t_arr[1][:-1]  # 'STOP'

                    word_f = self.generate_feature_vector(m_2_t, m_1_t, idx_sentences, word_i, w_t_arr[1], storage)
                    #self.train_f_v.append(word_f)
                    self.train_f_v[(m_2_t, m_1_t, idx_sentences, word_i), w_t_arr[1]] = word_f
                    m_2_t = m_1_t
                    m_1_t = w_t_arr[1]
                idx_sentences +=1

        return

    def create_train_feature_vector_prime_reduce(self, storage):
        # self.train_f_v_prime

        # self.test_f_v_prime
        idx_sentences = 0

        with open(self.train_file_name) as fd:
            for line in fd:
                # new sentences generate vectors
                sentence_list = line.split(' ')
                m_2_t = ['*']
                m_1_t = ['*']
                m_t = []
                if idx_sentences%50 == 0:
                    print 'line number: ' + str(idx_sentences)

                for word_i, val in enumerate(sentence_list):
                    # print str(val) + ': ' + str(word_i)
                    w_t_arr = val.split('_')
                    if '\n' in val:  # end of sentences
                        w_t_arr[1] = w_t_arr[1][:-1]  # 'STOP'      # not important in this function actually

                    # word exist in train data - run over all her tags.

                    if (w_t_arr[0] in self.word_dictionary_count):
                        m_t = []
                        for tag, flag in self.word_dictionary_count[w_t_arr[0]].iteritems():
                            if tag == 'cnt':
                                continue
                            else:
                                m_t.append(tag)
                                for idx_outer, outer_tag in enumerate(m_2_t):
                                    for idx_middle, middle_tag in enumerate(m_1_t):
                                        word_f = self.generate_feature_vector(outer_tag, middle_tag, idx_sentences,
                                                                              word_i, tag, storage)
                                        self.train_f_v_prime[
                                            (outer_tag, middle_tag, idx_sentences, word_i), tag] = word_f
                        m_2_t = m_1_t
                        m_1_t = m_t
                    else:
                        print 'bug word not in dict (train data)'

                idx_sentences += 1

        return

    def create_relevant_storage(self, file_name, storage_name):
        idx_sentences = 0
        with open(file_name) as fd:
            for line in fd:
                sentence_list = line.split(' ')
                cur_sentences = []
                for i, val in enumerate(sentence_list):
                    w_t_arr = val.split('_')
                    #if '\n' in val:  # end of sentence - skip (TODO analyze last word - 'STOP')
                    #    continue
                    #else:
                    cur_sentences.append(w_t_arr[0])

                storage_name[idx_sentences] = cur_sentences
                idx_sentences += 1
        return

    def create_test_feature_vector(self, storage):
        idx_sentences = 0

        with open(self.test_file_name) as fd:
            for line in fd:
                # new sentences generate vectors
                sentence_list = line.split(' ')
                m_2_t = '*'
                m_1_t = '*'
                cur_sentences = []
                for word_i, val in enumerate(sentence_list):
                    w_t_arr = val.split('_')
                    if '\n' in val:  # end of sentences
                        w_t_arr[1] = w_t_arr[1][:-1]  # 'STOP'
                    word_f = self.generate_feature_vector(m_2_t, m_1_t, idx_sentences, word_i, w_t_arr[1], storage)
                    #self.test_f_v.append(word_f)
                    self.test_f_v[(m_2_t, m_1_t, idx_sentences, word_i), w_t_arr[1]] = word_f
                    m_2_t = m_1_t
                    m_1_t = w_t_arr[1]
                idx_sentences += 1

        return

    def create_test_feature_vector_prime(self, storage):
        # self.test_f_v_prime
        idx_sentences = 0
        word_seen = 0
        word_unseen = 0

        with open(self.test_file_name) as fd:
            for line in fd:
                # new sentences generate vectors
                sentence_list = line.split(' ')
                m_2_t = ['*']
                m_1_t = ['*']
                m_t = []

                #print 'line number: ' + str(idx_sentences)
                #print datetime.datetime.now()

                for word_i, val in enumerate(sentence_list):
                    #print str(val) + ': ' + str(word_i)
                    w_t_arr = val.split('_')
                    if '\n' in val:  # end of sentences
                        w_t_arr[1] = w_t_arr[1][:-1]  # 'STOP'      # not important in this function actually

                    # 2 cases:
                    # a. if word exist in train data - run over all her tags.
                    # b. word does not exist - run over five common tags

                    if (w_t_arr[0] in self.word_dictionary_count):
                        # option a
                        if w_t_arr[0] not in self.seen_dict:
                            self.seen_dict[w_t_arr[0]] = 0
                        self.seen_dict[w_t_arr[0]] += 1

                        word_seen +=1
                        m_t = []
                        for tag, flag in self.word_dictionary_count[w_t_arr[0]].iteritems():
                            if tag == 'cnt':
                                continue
                            else:
                                m_t.append(tag)
                                for idx_outer, outer_tag in enumerate(m_2_t):
                                    for idx_middle, middle_tag in enumerate(m_1_t):
                                        word_f = self.generate_feature_vector(outer_tag, middle_tag, idx_sentences, word_i, tag, storage)
                                        self.test_f_v_prime[(outer_tag, middle_tag, idx_sentences, word_i), tag] = word_f
                        m_2_t = m_1_t
                        m_1_t = m_t
                    else:

                        if w_t_arr[0] not in self.unseen_dict:
                            self.unseen_dict[w_t_arr[0]] = 0
                        self.unseen_dict[w_t_arr[0]] +=1

                        word_unseen += 1
                        # run over 5 common tags
                        for idx_outer, outer_tag in enumerate(m_2_t):
                            for idx_middle, middle_tag in enumerate(m_1_t):
                                for idx_tag, tag in enumerate(self.common_tag_list):
                                    word_f = self.generate_feature_vector(outer_tag, middle_tag, idx_sentences, word_i, tag, storage)
                                    self.test_f_v_prime[(outer_tag, middle_tag, idx_sentences, word_i), tag] = word_f

                        m_2_t = m_1_t
                        m_1_t = self.common_tag_list

                    # finish word f_v update tags - TODO think if necessary for test
                    #m_2_t = m_1_t
                    #m_1_t = w_t_arr[1]

                idx_sentences += 1
            print 'w unseen'
            print word_unseen
            print 'w seen'
            print word_seen
        return

    def generate_feature_vector(self, m_2_t, m_1_t, s_i, w_i, w_t, relevant_storage):

        # x_vector type: t-2, t-1, sentence_idx, word_index, word_tag

        word_f = {}
        feature_vector_length = self.feature_vec_type_counter['All']  # length of inner array
        f_array = np.zeros_like(np.arange(feature_vector_length))
        #f_array = csr_matrix(self.feature_vec_type_counter['All'])

        if '100' in self.feature_using:
            key_f_100 = relevant_storage[s_i][w_i] + '_' + w_t        # create key
            if key_f_100 in self.feature_vec_meta:
                feature_idx = self.feature_vec_meta[key_f_100]
                word_f[feature_idx] = 1
                f_array[feature_idx] = 1
            else:
                # did not see this feature
                self.miss_100 +=1

        if '101' in self.feature_using:
            cur_w = relevant_storage[s_i][w_i]
            if len(cur_w) > 2:
                cur_w = cur_w[0][-3:]
                key_f_101 = cur_w + '_' + w_t  # create key
                if key_f_101 in self.feature_vec_meta:
                    feature_idx = self.feature_vec_meta[key_f_101]
                    word_f[feature_idx] = 1
                    f_array[feature_idx] = 1
                else:
                    # did not see this feature
                    self.miss_101 += 1

        if '102' in self.feature_using:
            cur_w = relevant_storage[s_i][w_i]
            if len(cur_w) > 2:
                cur_w = cur_w[0][:3]
                key_f_102 = cur_w + '_' + w_t  # create key
                if key_f_102 in self.feature_vec_meta:
                    feature_idx = self.feature_vec_meta[key_f_102]
                    word_f[feature_idx] = 1
                    f_array[feature_idx] = 1
                else:
                    # did not see this feature
                    self.miss_102 += 1

        if '103' in self.feature_using:
            key_f_103 = m_2_t + '_' + m_1_t + '_' + w_t# create key
            if key_f_103 in self.feature_vec_meta:
                feature_idx = self.feature_vec_meta[key_f_103]
                word_f[feature_idx] = 1
                f_array[feature_idx] = 1
            else:
                # did not see this feature
                self.miss_103 += 1

        if '104' in self.feature_using:
            key_f_104 = m_1_t + '_' + w_t  # create key
            if key_f_104 in self.feature_vec_meta:
                feature_idx = self.feature_vec_meta[key_f_104]
                word_f[feature_idx] = 1
                f_array[feature_idx] = 1
            else:
                # did not see this feature
                self.miss_104 += 1

        if '105' in self.feature_using:
            key_f_105 = w_t  # create key
            if key_f_105 in self.feature_vec_meta:
                feature_idx = self.feature_vec_meta[key_f_105]
                word_f[feature_idx] = 1
                f_array[feature_idx] = 1
            else:
                # did not see this feature
                self.miss_105 += 1

        if '106' in self.feature_using:
            if w_i != 0:    #feature contain last word -> current first word - continue
                prev_word = relevant_storage[s_i][int(w_i)-1]
                key_f_106 = prev_word + '_' + w_t  # create key
                if key_f_106 in self.feature_vec_meta:
                    feature_idx = self.feature_vec_meta[key_f_106]
                    word_f[feature_idx] = 1
                    f_array[feature_idx] = 1
                else:
                    # did not see this feature
                    self.miss_106 += 1

        if '107' in self.feature_using:
            if w_i < len(relevant_storage[s_i])-2:  # feature contain next word -> current last word - continue
                next_word = relevant_storage[s_i][int(w_i) + 1]
                key_f_107 = next_word + '_' + w_t  # create key
                if key_f_107 in self.feature_vec_meta:
                    feature_idx = self.feature_vec_meta[key_f_107]
                    word_f[feature_idx] = 1
                    f_array[feature_idx] = 1
                else:
                    # did not see this feature
                    self.miss_107 += 1

        f_compact = csr_matrix(f_array)     # represent sparse matrix
        return f_compact

    def eval_test_data(self):

        with open(self.test_file_name) as fd:
            for line in fd:
                sentence_list = line.split(' ')
                clean_sentence = []
                for idx, value in enumerate(sentence_list):
                    clean_sentence.append(value.split('_')[0])

                # print sentence_list
                m_2_t = '*'
                m_1_t = '*'

                for i, val in enumerate(sentence_list):

                    w_t_arr = val.split('_')

                    # create vector representation
                    cur_vector = []
                    cur_vector.append(m_2_t)
                    cur_vector.append(m_1_t)
                    cur_vector.append(clean_sentence)
                    cur_vector.append(i)

                    for tag, flag in self.tag_dict.iteritems(): # TODO eval '*' 'STOP' tags too
                        self.create_sentence_vector(cur_vector, tag)

                    m_2_t = m_1_t       # update to next word last two tags
                    m_1_t = w_t_arr[1]

    def create_sentence_vector(self,cur_vector, tag):

        cur_vector.append(tag)
        cur_word = cur_vector[2][cur_vector[3]]
        cur_feature_vector = {}

        # find binary vector
        # feature order: f.100, f.103, f.104

        # f.100 - lookup word - tag pairs
        # TODO check if exist
        if cur_word in self.word_dictionary_count:
            for tag, flag in self.word_dictionary_count.iteritems():
                if tag != 'cnt':
                    key = cur_word + '_' + tag
                    feature_num = self.feature_vec_meta[key]
                    cur_feature_vector[feature_num] = 1

        # f.103 - trigram tags TODO  for each trigram and bigram
        return


def main():

    print 'start MEMM'

    evaluate_train_data = True
    evaluate_test_data = True

    # file names
    train_file = 'C:\\gitprojects\\nlp\\hw1\\data\\small_train.wtag'
    test_file = 'C:\\gitprojects\\nlp\\hw1\\data\\small_test.wtag'
    comp_file = 'C:\\gitprojects\\nlp\\hw1\\data\\comp.words'

    MEMM(evaluate_train_data, evaluate_test_data, ['104'], train_file, test_file, comp_file)


if __name__ == "__main__":
    main()