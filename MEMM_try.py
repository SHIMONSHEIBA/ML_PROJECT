
class MEMM:
   """Base class of modeling MEMM logic on the data"""

   'shared among all instances of the class'
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
       MEMM.classVariable += 1

   def display_count(self):
       print("Total MEMM instances: %d" % MEMM.classVariable)

   def display_training_file(self):
       print("training file : ", self.trainingfile)

