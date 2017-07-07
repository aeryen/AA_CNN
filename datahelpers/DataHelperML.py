import pkg_resources
import os
import numpy as np
import logging

from datahelpers.DataHelper import DataHelper
from datahelpers.Data import AAData
# from utils import featuremaker
import errno


class DataHelperML(DataHelper):
    def __init__(self, doc_level, embed_type, embed_dim, target_doc_len, target_sent_len,
                 total_fold, t_fold_index, train_csv_file, data_dir="ml_mulmol"):
        super(DataHelperML, self).__init__(doc_level=doc_level, embed_type=embed_type, embed_dim=embed_dim,
                                           target_doc_len=target_doc_len, target_sent_len=target_sent_len,
                                           total_fold=total_fold, t_fold_index=t_fold_index)

        self.training_data_dir = pkg_resources.resource_filename('datahelpers', 'data/' + data_dir + '/')
        self.train_csv_file = train_csv_file
        self.train_label_file_path = self.training_data_dir + "_new_label/" + train_csv_file
        self.val_label_file_path = self.training_data_dir + "_new_label/val.csv"
        self.test_label_file_path = self.training_data_dir + "_new_label/test.csv"

        self.train_data = None
        self.test_data = None

    def temp_write_channel_file(self, author_code, file_name, file_text_content):
        fm = featuremaker.FeatureMaker(file_text_content)
        poss = fm.fast_pos_tag()
        word_len = fm.per_word_length()
        [prefix_2, prefix_3, suffix_2, suffix_3] = fm.prefix_suffix()

        if not os.path.exists(os.path.dirname("./data/ml_mulmol/" + author_code + "/")):
            try:
                os.makedirs(os.path.dirname("./data/ml_mulmol/" + author_code + "/"))
            except OSError as exc:  # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise

        original_txt = open("./data/ml_mulmol/" + author_code + "/" + file_name, "w")
        for line in file_text_content:
            original_txt.write(" ".join(line) + "\n")

        pos_file = open("./data/ml_mulmol/" + author_code + "/" + "pos_" + file_name, "w")
        for line in poss:
            pos_file.write(" ".join(line) + "\n")

        wl_file = open("./data/ml_mulmol/" + author_code + "/" + "wl_" + file_name, "w")
        for line in word_len:
            wl_file.write(" ".join(map(str, line)) + "\n")

        p2_file = open("./data/ml_mulmol/" + author_code + "/" + "pre2_" + file_name, "w")
        for line in prefix_2:
            p2_file.write(" ".join(line) + "\n")

        p3_file = open("./data/ml_mulmol/" + author_code + "/" + "pre3_" + file_name, "w")
        for line in prefix_3:
            p3_file.write(" ".join(line) + "\n")

        s2_file = open("./data/ml_mulmol/" + author_code + "/" + "suf2_" + file_name, "w")
        for line in suffix_2:
            s2_file.write(" ".join(line) + "\n")

        s3_file = open("./data/ml_mulmol/" + author_code + "/" + "suf3_" + file_name, "w")
        for line in suffix_3:
            s3_file.write(" ".join(line) + "\n")

        return poss, word_len, prefix_2, prefix_3, suffix_2, suffix_3

    @staticmethod
    def load_raw_file(data_dir, author_name, file_name):
        if not os.path.exists(os.path.dirname(data_dir + author_name + "/")):
            logging.error("error: " + author_name + " does not exit")
            return
        file_content = open(data_dir + author_name + "/txt/txt-preprocessed/" + file_name, "r").readlines()
        content = []
        paragraph = []
        for line in file_content:
            line = line.strip()
            if len(line) == 0 and len(paragraph) > 0:  # end of paragraph, split and push
                paragraph = " ".join(paragraph)
                content.extend(DataHelper.split_sentence(paragraph))
                paragraph = []
            elif len(line.split()) <= 2:  # too short
                pass
            else:  # keep adding to paragraph
                paragraph.append(line)
        return content

    def load_raw_dir(self, csv_file):
        authors, file_ids, label_matrix = DataHelperML.load_csv(csv_file_path=csv_file)
        self.num_of_classes = label_matrix.shape[1]

        logging.info("LABEL MATRIX HAS SHAPE: " + str(label_matrix.shape))

        data = AAData(name="ML", size=len(file_ids))
        data.file_id = file_ids

        origin_list = [None] * data.size
        doc_size = [None] * data.size

        folder_list = os.listdir(self.training_data_dir)
        for author in folder_list:
            f = self.training_data_dir + author + "/txt/txt-preprocessed/"
            if os.path.isdir(f):
                sub_file_list = os.listdir(f)
                for file_name in sub_file_list:
                    if file_name in data.file_id:
                        index = data.file_id.index(file_name)
                        file_content = DataHelperML.load_raw_file(data_dir=self.training_data_dir,
                                                                  author_name=author, file_name=file_name)
                        # self.temp_write_channel_file(author, file_name, file_content)
                        origin_list[index] = file_content
                        doc_size[index] = len(file_content)

        doc_size = np.array(doc_size)

        data.raw = origin_list
        data.label_doc = label_matrix
        data.doc_size = doc_size

        return data

    @staticmethod
    def load_proced_file(data_dir, author_code, file_name):
        if not os.path.exists(os.path.dirname(data_dir + author_code + "/")):
            logging.error("error: " + author_code + " does not exit")
            return
        file_content = open(data_dir + author_code + "/" + file_name, "r").readlines()
        file_content = [line.split() for line in file_content]
        return file_content

    def load_proced_dir(self, csv_file):
        authors, file_ids, label_matrix = DataHelper.load_csv(csv_file_path=csv_file)
        self.num_of_classes = label_matrix.shape[1]

        logging.info("LABEL MATRIX HAS SHAPE: " + str(label_matrix.shape))

        data = AAData(name="ML", size=len(file_ids))
        data.file_id = file_ids

        origin_list = [None] * data.size
        doc_size = [None] * data.size

        folder_list = os.listdir(self.training_data_dir)
        for author in folder_list:
            f = self.training_data_dir + author
            if os.path.isdir(f):
                sub_file_list = os.listdir(f)
                for file_name in sub_file_list:
                    if file_name in data.file_id:
                        index = data.file_id.index(file_name)
                        file_content = DataHelperML.load_proced_file(data_dir=self.training_data_dir,
                                                                     author_code=author, file_name=file_name)
                        origin_list[index] = file_content
                        doc_size[index] = len(file_content)

        doc_size = np.array(doc_size)

        data.raw = origin_list
        data.label_doc = label_matrix
        data.doc_size = doc_size

        return data

    def get_comb_count(self, x):
        n = int(len(x) / 50)
        print("n = " + str(n))
        if len(x) - (n * 50) > 0:
            n += 1
        return n

    def multi_sent_combine(self, x):
        n = self.get_comb_count(x)
        comb_list = []
        for i in range(n):
            comb_list.append(x[i * 50:(i + 1) * 50])
        return comb_list

    def comb_all_doc(self, data):
        x_comb = []
        label_comb = []
        comb_size = []

        [comb_size.append(self.get_comb_count(doc)) for doc in data.raw]
        [x_comb.extend(self.multi_sent_combine(doc)) for doc in data.raw]

        for comb_index in range(len(data.raw)):
            label_comb.extend(np.tile(data.label_doc[comb_index], [comb_size[comb_index], 1]))
            print("number of comb in document: " + str(comb_size[comb_index]))

        data.raw = x_comb
        data.label_instance = label_comb
        data.comb_size = comb_size

        return x_comb, label_comb, comb_size