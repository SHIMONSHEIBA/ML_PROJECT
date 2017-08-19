
class MEMM:
   """Base class of modeling MEMM logic on the data"""
   # """shared among all instances of the class'
   amino_mapping = { 'TTT' : 'Phe' , 'TTC' : 'Phe','TTA' : 'Leu','TTG' : 'Leu','CTT' : 'Leu','CTC' : 'Leu',
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

   tags_dict = { '1' : [0,'A+'] , '2' : [0,'C+'] ,'3' : [0,'G+'] , '4' : [0,'T+'] ,'5' : [0,'A-'] , '6' : [0,'C-'] ,
                 '7' : [0,'G-'] , '8' : [0,'T-'] }

   words_dict = { 'A' : 0 , 'T' : 0 , 'C' : 0 , 'G' : 0 }

    def __init__(self, trainingfile):

       self.trainingFile = trainingfile
       self.feature_1 = {}  # feature
       self.feature_2 = {}  # feature
       self.feature_3 = {}  # feature
       self.feature_4 = {}  # feature
       self.feature_5 = {}  # feature
       self.feature_6 = {}  # feature
       self.feature_7 = {}  # feature

       self.build_features_from_train()


   def build_features_from_train(self):

       print("starting building features from train")

       with open(self.train_file_name) as fd:


        return


