import collections
import numpy as np
import os
import logging
import pkg_resources

from datahelpers.DataHelperML import DataHelperML
from datahelpers.Data import LoadMethod


class DataHelperML2CH(DataHelperML):
    Record = collections.namedtuple('Record', ['file', 'author', 'content'])
    problem_name = "ML"

    def __init__(self, doc_level, embed_type, embed_dim, target_doc_len, target_sent_len, total_fold, t_fold_index,
                 train_csv_file="labels.csv"):
        logging.info("Data Helper: " + __file__ + " initiated.")

        if embed_type != "both":
            print('embed_type should be both')
            raise RuntimeError
        super(DataHelperML2CH, self).__init__(doc_level=doc_level, embed_type="both", embed_dim=embed_dim,
                                              target_doc_len=target_doc_len, target_sent_len=target_sent_len,
                                              total_fold=total_fold, t_fold_index=t_fold_index,
                                              train_csv_file=train_csv_file)

        self.embed_matrix_glv = None
        self.embed_matrix_w2v = None

        self.training_data_dir = pkg_resources.resource_filename('datahelpers', 'data/ml_mulmol/')
        self.train_label_file_path = self.training_data_dir + "_new_label/" + train_csv_file
        self.val_label_file_path = self.training_data_dir + "_new_label/val.csv"
        self.test_label_file_path = self.training_data_dir + "_new_label/test.csv"

        self.load_data()

    def load_data(self):
        # o = DataHelper(file_to_load)
        all_file_csv_path = self.training_data_dir + "_old_label/" + self.train_csv_file
        all_data = self.load_proced_dir(csv_file=all_file_csv_path)

        self.vocab, self.vocab_inv = self.build_vocab([all_data], self.vocabulary_size)
        self.embed_matrix_glv = self.build_glove_embedding(self.vocab_inv)
        self.embed_matrix_w2v = self.build_w2v_embedding(self.vocab_inv)

        if self.doc_level_data == LoadMethod.COMB:
            all_data = self.comb_all_doc(all_data)  # TODO

        all_data = self.build_content_vector(all_data)
        all_data = self.pad_sentences(all_data)

        if self.doc_level_data == LoadMethod.COMB:
            all_data = self.pad_document(all_data, 50)  # TODO 50
        elif self.doc_level_data == LoadMethod.DOC:
            all_data = self.pad_document(all_data, target_length=self.target_doc_len)

        [train_data, test_data] = DataHelperML.split_by_fold_2(self.total_fold, self.t_fold_index, all_data)

        if self.doc_level_data == LoadMethod.SENT:
            train_data = self.flatten_doc_to_sent(train_data)
            test_data = self.flatten_doc_to_sent(test_data)
        elif self.doc_level_data == LoadMethod.DOC:
            train_data.label_instance = train_data.label_doc
            test_data.label_instance = test_data.label_doc

        self.train_data = train_data
        self.train_data.embed_matrix = self.embed_matrix_glv
        self.train_data.embed_matrix_w2v = self.embed_matrix_w2v
        self.train_data.vocab = self.vocab
        self.train_data.vocab_inv = self.vocab_inv
        self.test_data = test_data
        self.test_data.embed_matrix = self.embed_matrix_glv
        self.test_data.embed_matrix_w2v = self.embed_matrix_w2v
        self.test_data.vocab = self.vocab
        self.test_data.vocab_inv = self.vocab_inv
