import pandas as pd
import numpy as np
from time import time
import time
import random
import math


class DataCreation:
    def __init__(self):
        for i in range(1, 18):
            self.create_chrom('chr'+str(i))

    def create_chrom(self, file_name):
        # load data and genes file for the specific chrome
        print '{}: Start loading: {}'.format((time.asctime(time.localtime(time.time()))), file_name)
        data_file_path = 'C:\\gitprojects\\ML project\\DNA_data\\' + file_name + '.xlsx'
        gene_file_path = 'C:\\gitprojects\\ML project\\DNA_data\\genes\\' + file_name + '.xlsx'
        chrome = pd.read_excel(data_file_path)
        genes = pd.read_excel(gene_file_path, sheetname=1)
        # print chrome.iloc[[0]]
        # print chrome.shape
        chrome = pd.DataFrame(chrome.values.reshape(1, -1))
        chrome = chrome.loc[:, (chrome != 0).any(axis=0)]
        print chrome.iloc[[0]]
        print chrome.shape

        # labels dictionary:
        # gene labels
        gene_dict_A = {u'A': 'A_1'}
        gene_dict_C = {u'C': 'C_2'}
        gene_dict_G = {u'G': 'G_3'}
        gene_dict_T = {u'T': 'T_4'}

        # non gene labels
        non_gene_dict_A = {u'A': 'A_5'}
        non_gene_dict_C = {u'C': 'C_6'}
        non_gene_dict_G = {u'G': 'G_7'}
        non_gene_dict_T = {u'T': 'T_8'}

        # create label vector
        # labels = pd.DataFrame(0, index=np.arange(chrome.shape[1]))
        print '{}: Start Creating gene labels'.format(time.asctime(time.localtime(time.time())))
        labels = chrome.copy()
        # print labels.shape
        t1 = time.time()
        for index, gene in genes.iterrows():
            start = gene['cdsStart']
            end = gene['cdsEnd']
            # print labels.iloc[:, start:end]
            labels.iloc[:, start:end].replace(gene_dict_A, inplace=True)
            labels.iloc[:, start:end].replace(gene_dict_C, inplace=True)
            labels.iloc[:, start:end].replace(gene_dict_G, inplace=True)
            labels.iloc[:, start:end].replace(gene_dict_T, inplace=True)
            # print labels.iloc[:, start:end]
            print '{}: Finish creating {} gene labels'.format(time.asctime(time.localtime(time.time())), index)
        t2 = time.time()
        print '{}: Finish creating gene labels. Time to create gene labels: {}'.\
            format(time.asctime(time.localtime(time.time())), t2-t1)
        print '{}: Start Creating non-gene labels'.format(time.asctime(time.localtime(time.time())))
        labels.replace(non_gene_dict_A, inplace=True)
        labels.replace(non_gene_dict_C, inplace=True)
        labels.replace(non_gene_dict_G, inplace=True)
        labels.replace(non_gene_dict_T, inplace=True)
        t3 = time.time()
        print '{}: Finish creating non-gene labels. Time to create non-gene labels: {}'.\
            format(time.asctime(time.localtime(time.time())), t3-t2)

        self.split_chrome(file_name, chrome, labels)

        return

    def split_chrome(self, file_name, chrome, labels):
        print '{}: Start split chrome: {}'.format(time.asctime(time.localtime(time.time())), file_name)
        t0 = time.time()
        # split chrome and label to samples
        chrome_size = chrome.shape[1]
        size_sum = 0  # will be the sum of the sizes of parts
        lower = 800  # lower bound to random
        upper = 1200  # upper bound to random
        array_shape = int(math.ceil(1.0*chrome_size/lower))  # maximum size of array of samples
        # will be an array of the chrome split to samples
        split_chrome = np.chararray(shape=(array_shape, 1), itemsize=(2*upper)+1, unicode=True)
        # will be an array of the labels split to samples
        split_label = np.chararray(shape=(array_shape, 1), itemsize=(4*upper)+1, unicode=True)
        # will be an array of labels for the whole sample (seq)
        samples_label = np.empty(shape=(array_shape, 1), dtype=int)
        index = 0
        while size_sum < chrome_size:
            size = random.randint(lower, upper)  # choose random size
            # for the last part, if size will be larger than the rest of the chrome
            size = min(size, chrome_size - size_sum)
            # create chrome and label parts according to size
            chrome_part = chrome.iloc[:, size_sum:size_sum + size].copy().iloc[0]
            label_part = labels.iloc[:, size_sum:size_sum + size].copy().iloc[0]
            # create string out of the part
            chrome_part_string = chrome_part.str.cat(sep=',')
            label_part_string = label_part.str.cat(sep=',')
            if ('A_1' in label_part_string) or ('C_2' in label_part_string) or ('G_3' in label_part_string) or \
                    ('T_4' in label_part_string):
                sample_label = 1
            else:
                sample_label = -1
            split_chrome[index] = chrome_part_string
            split_label[index] = label_part_string
            samples_label[index] = sample_label
            index += 1
            size_sum += size
        # save split data and label to csv and all chrome data to numpy
        np.savetxt('C:\gitprojects\ML_PROJECT\\data\\'+file_name+'_data.csv', split_chrome[0:index, :],
                   delimiter=",", fmt='%s')
        np.savetxt('C:\gitprojects\ML_PROJECT\\labels\\'+file_name + '_label.csv', split_label[0:index, :],
                   delimiter=",", fmt='%s')
        np.savetxt('C:\gitprojects\ML_PROJECT\\sample_labels\\'+file_name + '_sample_label.csv',
                   samples_label[0:index, :], delimiter=",", fmt='%i')
        np.save('C:\gitprojects\ML_PROJECT\\numpy\\'+file_name + '_data.npy', chrome)
        np.save('C:\gitprojects\ML_PROJECT\\numpy\\'+file_name + '_label.npy', labels)
        t1 = time.time()
        print '{}: Finish split chrome: {} and save files. The split took: {}'.\
            format(time.asctime(time.localtime(time.time())), file_name, t1-t0)

        return


if __name__ == '__main__':
    data = DataCreation()

