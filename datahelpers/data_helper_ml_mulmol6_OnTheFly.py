import collections
import errno
import math
import os
import logging
import pkg_resources
import numpy as np

from utils import featuremaker
from datahelpers.DataHelperML import DataHelperML
from datahelpers.Data import LoadMethod
from datahelpers.Data import AAData


class DataHelperMulMol6(DataHelperML):
    Record = collections.namedtuple('Record', ['file', 'author', 'content'])
    problem_name = "ML"

    def __init__(self, doc_level, embed_type, embed_dim, target_doc_len, target_sent_len, train_csv_file="train.csv"):
        logging.info("Data Helper: " + __file__ + " initiated.")

        super(DataHelperMulMol6, self).__init__(doc_level=doc_level, embed_type=embed_type, embed_dim=embed_dim,
                                                target_doc_len=target_doc_len, target_sent_len=target_sent_len,
                                                train_csv_file=train_csv_file, data_dir="ml_dataset")

        self.training_data_dir = pkg_resources.resource_filename('datahelpers', 'data/ml_dataset/')
        self.truth_file_path = self.training_data_dir + train_csv_file

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



    def load_data(self):
        # o = DataHelper(file_to_load)
        train_data = self.load_raw_dir(csv_file=self.train_label_file_path)
        val_data = self.load_raw_dir(csv_file=self.val_label_file_path)
        test_data = self.load_raw_dir(csv_file=self.test_label_file_path)

        # x_concat_exp = np.concatenate([train_data.raw, val_data.raw, test_data.raw], axis=0)
        self.vocab, self.vocab_inv = self.build_vocab([train_data, val_data, test_data], self.vocabulary_size)

        if not self.doc_level_data:
            train_data = self.flatten_doc_to_sent(train_data)
            val_data = self.flatten_doc_to_sent(val_data)
            test_data = self.flatten_doc_to_sent(test_data)
            self.train = train_data
            self.val = val_data
            self.test = test_data

        if self.embed_type == "glove":
            self.embed_matrix = self.build_glove_embedding(self.vocab_inv)
        else:
            self.embed_matrix = self.build_w2v_embedding(self.vocab_inv)

        self.x_train = DataHelperMulMol6.build_input_data(self.x_train)
        self.x_train = self.pad_sentences(self.x_train, target_length=self.target_sent_len)

        self.x_test = DataHelperMulMol6.build_input_data(self.x_test)
        self.x_test = self.pad_sentences(self.x_test, target_length=self.target_sent_len)

        if self.doc_level_data:
            self.longest_sentence(self.x_train, False)
            self.x_train = self.pad_document(self.x_train, target_length=self.target_doc_len)
            self.x_test = self.pad_document(self.x_test, target_length=self.target_doc_len)

        return [self.x_train, self.labels_train, self.vocab, self.vocab_inv, self.embed_matrix]

    def load_test_data(self):
        if self.x_test is not None:
            if self.doc_level_data:
                return [self.x_test, self.labels_test, self.vocab, self.vocab_inv, self.doc_size_test]
            else:
                return [self.x_test, self.labels_test, self.vocab, self.vocab_inv, self.doc_size_test]
        else:
            print("nope")

    def get_file_id_test(self):
        return self.file_id_test

    def get_doc_label(self):
        return self.labels_test


if __name__ == "__main__":
    o = DataHelperMulMol6(doc_level=LoadMethod.SENT, embed_type="glove", embed_dim=100, target_doc_len=400, target_sent_len=100)
    o.load_data()
    # o.load_test_data()
    print("o")
