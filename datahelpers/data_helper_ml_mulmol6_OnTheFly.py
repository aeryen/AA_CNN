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

        # overrides the csv path, even though the content is the same
        self.training_data_dir = pkg_resources.resource_filename('datahelpers', 'data/ml_dataset/')
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

    def load_data(self):
        # train_data = self.load_raw_dir(csv_file=self.train_label_file_path)
        # val_data = self.load_raw_dir(csv_file=self.val_label_file_path)
        # test_data = self.load_raw_dir(csv_file=self.test_label_file_path)

        all_file_csv_path = self.training_data_dir + "_old_label/labels.csv"
        all_data = self.load_raw_dir(csv_file=all_file_csv_path)


        # x_concat_exp = np.concatenate([train_data.raw, val_data.raw, test_data.raw], axis=0)
        # self.vocab, self.vocab_inv = self.build_vocab([train_data, val_data, test_data], self.vocabulary_size)
        self.vocab, self.vocab_inv = self.build_vocab([all_data], self.vocabulary_size)

        self.embed_matrix = self.build_embedding(self.vocab_inv)

        all_data = self.build_content_vector(all_data)
        all_data = self.pad_sentences(all_data)

        if self.doc_level_data == LoadMethod.DOC or self.doc_level_data == LoadMethod.COMB:
            all_data = self.pad_document(all_data, target_length=self.target_doc_len)

        [train_data, test_data] = DataHelperML.split_by_fold_2(5, 0, all_data)

        if self.doc_level_data == LoadMethod.SENT:
            train_data = self.flatten_doc_to_sent(train_data)
            test_data = self.flatten_doc_to_sent(test_data)

        self.train_data = train_data
        self.test_data = test_data

        return [self.train_data, self.vocab, self.vocab_inv, self.embed_matrix]

    def load_test_data(self):
        if self.test_data is not None:
            return [self.test_data, self.vocab, self.vocab_inv]
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
