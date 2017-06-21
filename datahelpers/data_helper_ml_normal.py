import collections
import numpy as np
import pickle
import os
import math
import logging
import pkg_resources

from datahelpers.DataHelperML import DataHelperML
from datahelpers.Data import AAData
from datahelpers.Data import LoadMethod


class DataHelperMLNormal(DataHelperML):
    Record = collections.namedtuple('Record', ['file', 'author', 'content'])
    problem_name = "ML"

    def __init__(self, doc_level, embed_type, embed_dim, target_doc_len, target_sent_len, train_csv_file="labels.csv"):
        logging.info("Data Helper: " + __file__ + " initiated.")

        super(DataHelperML, self).__init__(doc_level=doc_level, embed_type=embed_type, embed_dim=embed_dim,
                                           target_doc_len=target_doc_len, target_sent_len=target_sent_len)

        self.training_data_dir = pkg_resources.resource_filename('datahelpers', 'data/ml_mulmol/')
        self.train_label_file_path = self.training_data_dir + "_new_label/" + train_csv_file
        self.val_label_file_path = self.training_data_dir + "_new_label/val.csv"
        self.test_label_file_path = self.training_data_dir + "_new_label/test.csv"

        self.load_data()

    def load_data(self):
        all_file_csv_path = self.training_data_dir + "_old_label/labels.csv"
        all_data = self.load_proced_dir(csv_file=all_file_csv_path)

        self.vocab, self.vocab_inv = self.build_vocab([all_data], self.vocabulary_size)
        self.embed_matrix = self.build_embedding(self.vocab_inv)

        if self.doc_level_data == LoadMethod.COMB:
            all_data = self.comb_all_doc(all_data)

        all_data = self.build_content_vector(all_data)
        all_data = self.pad_sentences(all_data)

        if self.doc_level_data == LoadMethod.COMB:
            all_data = self.pad_document(all_data, 50)  # TODO 50
        elif self.doc_level_data == LoadMethod.DOC:
            all_data = self.pad_document(all_data, target_length=self.target_doc_len)

        [train_data, test_data] = DataHelperML.split_by_fold_2(5, 0, all_data)

        if self.doc_level_data == LoadMethod.SENT:
            train_data = self.flatten_doc_to_sent(train_data)
            test_data = self.flatten_doc_to_sent(test_data)

        self.train_data = train_data
        self.test_data = test_data

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
            comb_list.append(x[i*50:(i+1)*50])
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

    def get_train_data(self):
        return [self.train_data, self.vocab, self.vocab_inv, self.embed_matrix]

    def get_test_data(self):
        return [self.test_data, self.vocab, self.vocab_inv]
