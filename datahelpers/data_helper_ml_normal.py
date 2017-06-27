import collections
import logging
import pkg_resources

from datahelpers.DataHelperML import DataHelperML
from datahelpers.Data import AAData
from datahelpers.Data import LoadMethod


class DataHelperMLNormal(DataHelperML):
    Record = collections.namedtuple('Record', ['file', 'author', 'content'])
    problem_name = "ML"

    def __init__(self, doc_level, embed_type, embed_dim, target_doc_len, target_sent_len, total_fold, t_fold_index,
                 train_csv_file="labels.csv"):
        logging.info("Data Helper: " + __file__ + " initiated.")

        super(DataHelperMLNormal, self).__init__(doc_level=doc_level, embed_type=embed_type, embed_dim=embed_dim,
                                                 target_doc_len=target_doc_len, target_sent_len=target_sent_len,
                                                 total_fold=total_fold, t_fold_index=t_fold_index,
                                                 train_csv_file=train_csv_file)

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
        self.test_data = test_data

    def get_train_data(self):
        return [self.train_data, self.vocab, self.vocab_inv, self.embed_matrix]

    def get_test_data(self):
        return [self.test_data, self.vocab, self.vocab_inv]
