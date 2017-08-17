import xlwt
from random import randint

class Evaluate:

    def __init__(self,memm_obj, viterbi_result, test_file, write_test_doc, write_confusion_test_doc, write_file_name, confusion_file_name):

        self.test_file_name = test_file
        self.viterbi_result = viterbi_result
        self.tag_dict = memm_obj.tag_dict
        self.memm_obj = memm_obj
        self.write_test_doc = write_test_doc
        self.write_confusion_test_doc = write_confusion_test_doc
        self.write_file_name = write_file_name
        self.confusion_file_name = confusion_file_name
        self.unseen_dict = memm_obj.unseen_dict
        self.seen_dict = memm_obj.seen_dict

        self.confusion_matrix = {}
        self.confusion_matrix_seen = {}
        self.confusion_matrix_unseen = {}
        self.confusion_matrix_active = True
        self.eval_res = {}

    def run(self):
        self.eval_res = self.eval_test_results(self.viterbi_result, self.test_file_name)
        if self.write_test_doc == True:
            self.write_result_doc()
        if self.write_confusion_test_doc == True:
            self.write_confusion_doc()


    def eval_test_results(self, predicted_w_t, real_file):
        # predicted_values -
        miss = 0
        miss_seen = 0
        miss_unseen = 0
        hit = 0
        hit_seen = 0
        hit_unseen = 0

        if self.confusion_matrix_active:  # build confusion matrix - analyze result later
            for outer_tag, outer_flag in self.tag_dict.iteritems():
                for inner_tag, inner_flag in self.tag_dict.iteritems():
                    cur_key = outer_tag + '_' + inner_tag
                    if cur_key not in self.confusion_matrix:
                        self.confusion_matrix[cur_key] = 0

                    if cur_key not in self.confusion_matrix_seen:
                        self.confusion_matrix_seen[cur_key] = 0

                    if cur_key not in self.confusion_matrix_unseen:
                        self.confusion_matrix_unseen[cur_key] = 0


        idx_sentences = 0
        a=3
        with open(real_file) as fd:  # real values
            for line in fd:
                sentence_list = line.split(' ')

                for i, val in enumerate(sentence_list):
                    w_t_arr = val.split('_')
                    if w_t_arr[1].endswith('\n'):
                        w_t_arr[1] = w_t_arr[1][:-1]

                    if a==4:
                        a=5
                    else:
                        p_key_val = predicted_w_t[idx_sentences][i].split('_')
                        p_w = p_key_val[0]  # our predicted tag
                        p_t = p_key_val[1]
                        if p_w != w_t_arr[0]:
                            print 'problem miss between prediction and test word indexes'
                        if p_t != w_t_arr[1]:  # tag miss
                            miss += 1
                            if self.confusion_matrix_active:
                                confusion_t_key = str(w_t_arr[1]) + '_' + str(p_t)  # real tag _ prediction tag
                                self.confusion_matrix[confusion_t_key] += 1

                                if p_w in self.unseen_dict:
                                    miss_unseen +=1
                                    self.confusion_matrix_unseen[confusion_t_key] += 1
                                else:
                                    miss_seen += 1
                                    self.confusion_matrix_seen[confusion_t_key] += 1

                        else:
                            hit += 1
                            if self.confusion_matrix_active:
                                confusion_t_key = str(w_t_arr[1]) + '_' + str(p_t)  # trace add
                                self.confusion_matrix[confusion_t_key] += 1

                                if p_w in self.unseen_dict:
                                    hit_unseen += 1
                                    self.confusion_matrix_unseen[confusion_t_key] += 1
                                else:
                                    hit_seen += 1
                                    self.confusion_matrix_seen[confusion_t_key] += 1

                idx_sentences += 1

        print 'Miss'
        print miss
        print 'Hit'
        print hit
        print 'Accuracy'
        print float(hit)/float(miss+hit)

        print 'Miss_unseen'
        print miss_unseen
        print 'Hit_unseen'
        print hit_unseen
        print 'Accuracy_unseen'
        print float(hit_unseen) / float(miss_unseen + hit_unseen)

        print 'Miss_seen'
        print miss_seen
        print 'Hit_seen'
        print hit_seen
        print 'Accuracy_seen'
        print float(hit_seen) / float(miss_seen + hit_seen)

        return {
            'confusion_matrix' : self.confusion_matrix,
            'Miss' : miss,
            'Hit' : hit,
            'Accuracy' : float(hit)/float(miss+hit),
            'Miss_unseen': miss_unseen,
            'Hit_unseen': hit_unseen,
            'Accuracy_unseen': float(hit_unseen) / float(miss_unseen + hit_unseen),
            'Miss_seen': miss_seen,
            'Hit_seen': hit_seen,
            'Accuracy_seen': float(hit_seen) / float(miss_seen + hit_seen),
        }

    def write_result_doc(self):

        file_name = self.write_file_name + '_' + '.wtag'
        f = open(file_name, 'w')

        for idx_sentences, sentence_list in self.viterbi_result.iteritems():
            for idx_inner, string_w_t in enumerate(sentence_list):
                f.write(string_w_t+' ')
            f.write('\n')                                           # finish sentences
        f.close()

        return

    def write_confusion_doc(self):
        # build confusion matrix doc
        # build structure of line and columns

        file_name = self.confusion_file_name + '.xls'
        book = xlwt.Workbook(encoding="utf-8")

        sheet1 = book.add_sheet("Confusion Matrix")
        sheet2 = book.add_sheet("Confusion Matrix seen")
        sheet3 = book.add_sheet("Confusion Matrix Unseen")

        column_rows_structure = self.tag_dict.keys()

        pattern = xlwt.Pattern()  # Create the Pattern
        pattern.pattern = xlwt.Pattern.SOLID_PATTERN  # May be: NO_PATTERN, SOLID_PATTERN, or 0x00 through 0x12
        pattern.pattern_fore_colour = 22  # May be: 8 through 63. 0 = Black, 1 = White, 2 = Red
        style = xlwt.XFStyle()  # Create the Pattern
        style.pattern = pattern  #

        pattern_mistake = xlwt.Pattern()  # Create the Pattern
        pattern_mistake.pattern = xlwt.Pattern.SOLID_PATTERN  # May be: NO_PATTERN, SOLID_PATTERN, or 0x00 through 0x12
        pattern_mistake.pattern_fore_colour = 2
        style_mistake = xlwt.XFStyle()  # Create the Pattern
        style_mistake.pattern = pattern_mistake

        pattern_good = xlwt.Pattern()  # Create the Pattern
        pattern_good.pattern = xlwt.Pattern.SOLID_PATTERN  # May be: NO_PATTERN, SOLID_PATTERN, or 0x00 through 0x12
        pattern_good.pattern_fore_colour = 3
        style_good = xlwt.XFStyle()  # Create the Pattern
        style_good.pattern = pattern_good


        sheet1.write(0, 0, ' ', style)
        for idx_tag, cur_tag in enumerate(column_rows_structure):
            sheet1.write(0, idx_tag+1, cur_tag, style)

        for row_tag_idx, row_tag in enumerate(column_rows_structure):
            sheet1.write(row_tag_idx+1, 0, row_tag, style)
            for col_tag_idx, col_tag in enumerate(column_rows_structure):
                cur_value = self.confusion_matrix[str(row_tag) + '_' + str(col_tag)]    # confusion_matrix_unseen
                if cur_value == 0:
                    sheet1.write(row_tag_idx + 1, col_tag_idx+1, str(cur_value))
                else:
                    if row_tag_idx == col_tag_idx:
                        sheet1.write(row_tag_idx + 1, col_tag_idx + 1, str(cur_value), style_good)
                    else:
                        sheet1.write(row_tag_idx + 1, col_tag_idx + 1, str(cur_value), style_mistake)


        sheet2.write(0, 0, ' ', style)
        for idx_tag, cur_tag in enumerate(column_rows_structure):
            sheet2.write(0, idx_tag + 1, cur_tag, style)

        for row_tag_idx, row_tag in enumerate(column_rows_structure):
            sheet2.write(row_tag_idx + 1, 0, row_tag, style)
            for col_tag_idx, col_tag in enumerate(column_rows_structure):
                cur_value = self.confusion_matrix_seen[str(row_tag) + '_' + str(col_tag)]
                if cur_value == 0:
                    sheet2.write(row_tag_idx + 1, col_tag_idx + 1, str(cur_value))
                else:
                    if row_tag_idx == col_tag_idx:
                        sheet2.write(row_tag_idx + 1, col_tag_idx + 1, str(cur_value), style_good)
                    else:
                        sheet2.write(row_tag_idx + 1, col_tag_idx + 1, str(cur_value), style_mistake)

        sheet3.write(0, 0, ' ', style)
        for idx_tag, cur_tag in enumerate(column_rows_structure):
            sheet3.write(0, idx_tag + 1, cur_tag, style)

        for row_tag_idx, row_tag in enumerate(column_rows_structure):
            sheet3.write(row_tag_idx + 1, 0, row_tag, style)
            for col_tag_idx, col_tag in enumerate(column_rows_structure):
                cur_value = self.confusion_matrix_unseen[str(row_tag) + '_' + str(col_tag)]  # confusion_matrix_unseen
                if cur_value == 0:
                    sheet3.write(row_tag_idx + 1, col_tag_idx + 1, str(cur_value))
                else:
                    if row_tag_idx == col_tag_idx:
                        sheet3.write(row_tag_idx + 1, col_tag_idx + 1, str(cur_value), style_good)
                    else:
                        sheet3.write(row_tag_idx + 1, col_tag_idx + 1, str(cur_value), style_mistake)



        book.save(file_name)

def main():
    Evaluate('a', 'b', 'c', 'd', 'd', 'e')

if __name__ == "__main__":
    main()
