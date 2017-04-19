import collections
import numpy as np
import pickle
import os
import math
import logging
import pkg_resources

from datahelpers.DataHelper import DataHelper


class DataHelperML(DataHelper):
    Record = collections.namedtuple('Record', ['file', 'author', 'content'])
    problem_name = "ML"

    vocabulary_size = 20000
    embedding_dim = 100

    train_size = None
    test_size = None

    file_id_train = None
    file_id_test = None
    labels_train = None
    labels_test = None
    x_train = None
    x_test = None
    doc_size_train = None
    doc_size_test = None

    doc_labels_test = None  # document level label backup (used in evaler)

    vocab = None
    vocab_inv = None
    embed_matrix = None

    doc_level_data = "doc"  # doc, comb, sent
    target_sent_len = None
    target_doc_len = None

    def __init__(self, doc_level="comb", embed_type="glove", embed_dim=100, target_doc_len=100, target_sent_len=220,
                 train_holdout=-1, num_fold=5, fold_index=0, truth_file="labels.csv"):
        logging.info("Data Helper: " + __file__ + " initiated.")

        super(DataHelperML, self).__init__(doc_level=doc_level, embed_type=embed_type, embed_dim=embed_dim,
                                           target_doc_len=target_doc_len, target_sent_len=target_sent_len,
                                           train_holdout=train_holdout, num_fold=num_fold, fold_index=fold_index)

        self.training_data_dir = pkg_resources.resource_filename('datahelpers', 'data/ml_mulmol/')
        self.truth_file_path = self.training_data_dir + truth_file

    def load_original_file(self, author_code, file_name):
        if not os.path.exists(os.path.dirname(self.training_data_dir + author_code + "/")):
            logging.error("error: " + author_code + " does not exit")
            return
        original_txt = open(self.training_data_dir + author_code + "/" + file_name, "r").readlines()
        original_txt = [line.split() for line in original_txt]
        return original_txt

    def __load_data(self):
        file_id_list = []
        label_matrix = []

        truth_file_content = open(self.truth_file_path, "r").readlines()
        self.author_list = truth_file_content[0].split(",")[1:]
        if self.num_of_classes is None:
            self.num_of_classes = len(truth_file_content[1].split(",")[1:])
        for line in truth_file_content[1:]:
            line = line.split(",")
            file_id_list.append(line[0])
            label_vector = list(map(int, line[1:]))
            label_matrix.append(np.array(label_vector))
        label_matrix = np.array(label_matrix)

        doc_count = len(file_id_list)
        doc_size = [None] * doc_count
        origin_list = [None] * doc_count

        folder_list = os.listdir(self.training_data_dir)
        if self.num_of_classes is None:
            self.num_of_classes = len(truth_file_content[1].split(",")[1:])
        for author in folder_list:
            f = self.training_data_dir + author
            if os.path.isdir(f):
                sub_file_list = os.listdir(f)
                for file_name in sub_file_list:
                    if file_name in file_id_list:
                        index = file_id_list.index(file_name)
                        original_txt = self.load_original_file(author, file_name)
                        origin_list[index] = original_txt  # document level array instead of all sentence list
                        doc_size[index] = len(original_txt)

        doc_size = np.array(doc_size)

        return [file_id_list, label_matrix, doc_size, origin_list]

    @staticmethod
    def build_input_data(docs, vocabulary, doc_level):
        unk = vocabulary["<UNK>"]
        if doc_level == "doc" or doc_level == "comb":
            x = np.array([[[vocabulary.get(word, unk) for word in sent] for sent in doc] for doc in docs])
        else:
            x = np.array([[vocabulary.get(word, unk) for word in doc] for doc in docs])
        return x

    def pad_sentences(self, docs, padding_word="<PAD>", target_length=-1):
        """
        Pads all sentences to the same length. The length is defined by the longest sentence.
        Returns padded sentences.
        """
        if target_length > 0:
            max_length = target_length
        else:
            sent_lengths = [len(x) for x in docs]
            max_length = max(sent_lengths)
            logging.info("longest sentence: " + str(max_length))

        if self.doc_level_data == "sent":
            padded_doc = []
            for i in range(len(docs)):
                sent = docs[i]
                if len(sent) <= max_length:
                    num_padding = max_length - len(sent)
                    new_sentence = np.concatenate([sent, np.zeros(num_padding, dtype=np.int)])
                else:
                    new_sentence = sent[:max_length]
                padded_doc.append(new_sentence)
            return np.array(padded_doc)
        else:
            padded_docs = []
            for doc in docs:
                padded_doc = []
                for i in range(len(doc)):
                    sent = doc[i]
                    if len(sent) <= max_length:
                        num_padding = max_length - len(sent)
                        new_sentence = np.concatenate([sent, np.zeros(num_padding, dtype=np.int)])
                    else:
                        new_sentence = sent[:max_length]
                    padded_doc.append(new_sentence)
                padded_docs.append(np.array(padded_doc))
            return padded_docs

    def pad_document(self, docs, padding_word="<PAD>", target_length=-1):
        """
        Pads all sentences to the same length. The length is defined by the longest sentence.
        Returns padded sentences.
        """
        if target_length > 0:
            tar_length = target_length
        else:
            doc_lengths = [len(d) for d in docs]
            tar_length = max(doc_lengths)
            logging.info("longest doc: " + str(tar_length))

        padded_doc = []
        sent_length = len(docs[0][0])
        for i in range(len(docs)):
            d = docs[i]
            if len(d) <= tar_length:
                num_padding = tar_length - len(d)
                if len(d) > 0:
                    new_doc = np.concatenate([d, np.zeros([num_padding, sent_length], dtype=np.int)])
                else:
                    new_doc = np.zeros([num_padding, sent_length], dtype=np.int)
            else:
                new_doc = d[:tar_length]
            padded_doc.append(new_doc)
        return np.array(padded_doc)

    def train_test_shuf_split(self, file_id, labels, doc_size, origin):
        rs = np.random.RandomState(10)
        shuffle_i = rs.permutation(np.arange(len(labels)))

        file_id_shuffled = [file_id[i] for i in shuffle_i]
        labels_shuffled = labels[shuffle_i]
        doc_size_shuffled = doc_size[shuffle_i]
        origin_shuffled = [origin[i] for i in shuffle_i]

        data_size = len(file_id)
        file_id_train, file_id_test = self.split_by_fold(self.num_fold, self.fold_index, data_size, file_id_shuffled)
        labels_train, labels_test = self.split_by_fold(self.num_fold, self.fold_index, data_size, labels_shuffled)
        doc_size_train, doc_size_test = self.split_by_fold(self.num_fold, self.fold_index, data_size, doc_size_shuffled)

        origin_train, origin_test = self.split_by_fold(self.num_fold, self.fold_index, data_size, origin_shuffled)

        return [file_id_train, file_id_test, labels_train, labels_test, doc_size_train, doc_size_test,
                origin_train, origin_test]

    def expand_origin_and_label_to_sentence(self, x, y):
        expand_x = []
        for x_doc in x:
            expand_x.extend(x_doc)
        expand_y = []
        for i in range(len(y)):
            expand_y.extend(np.tile(y[i], [len(x[i]), 1]))
        return [expand_x, expand_y]

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

    def comb_all_doc(self, x, label):
        x_comb = []
        label_comb = []
        comb_size = []

        [comb_size.append(self.get_comb_count(doc)) for doc in x]
        [x_comb.extend(self.multi_sent_combine(doc)) for doc in x]

        for comb_index in range(len(x)):
            label_comb.extend(np.tile(label[comb_index], [comb_size[comb_index], 1]))
            print("number of comb in document: " + str(comb_size[comb_index]))
    
        return x_comb, label_comb, comb_size

    def load_data(self):
        # o = DataHelper(file_to_load)
        file_name_ordered, label_matrix_ordered, doc_size, origin_list = self.__load_data()

        [self.file_id_train, self.file_id_test, self.labels_train, self.labels_test,
         self.doc_size_train, self.doc_size_test, self.x_train, self.x_test] = \
            self.train_test_shuf_split(file_id=file_name_ordered, labels=label_matrix_ordered, doc_size=doc_size, origin=origin_list)

        self.doc_labels_test = self.labels_test

        x_training_exp, labels_training_exp = self.expand_origin_and_label_to_sentence(self.x_train, self.labels_train)
        x_test_exp, labels_test_exp = self.expand_origin_and_label_to_sentence(self.x_test, self.labels_test)

        vocab_file = DataHelper.get_vocab_path(file_name=__file__, embed_type=self.embed_type, embed_dim=self.embedding_dim)
        x_concat_exp = np.concatenate([x_training_exp, x_test_exp], axis=0)
        # self.longest_sentence(x_concat_exp, True)
        self.vocab, self.vocab_inv = self.build_vocab(x_concat_exp, self.vocabulary_size)
        # pickle.dump([self.vocab, self.vocab_inv], open(vocab_file, "wb"))

        if self.doc_level_data == "sent":
            self.x_train = x_training_exp
            self.x_test = x_test_exp
            self.labels_train = labels_training_exp
            self.labels_test = labels_test_exp

        if self.embed_type == "glove":
            [glove_words, glove_vectors] = self.load_glove_vector(self.glove_path)
            self.embed_matrix = self.build_glove_embedding(self.vocab_inv, glove_words, glove_vectors, self.embedding_dim)
        else:
            self.word2vec_model = self.load_w2v_vector()
            self.embed_matrix = self.build_w2v_embedding(self.vocab_inv, self.word2vec_model, self.embedding_dim)

        if self.doc_level_data == "comb":
            [self.x_train, self.labels_train, self.doc_size_train] = \
                self.comb_all_doc(self.x_train, self.labels_train)

            [self.x_test, self.labels_test, self.doc_size_test] = \
                self.comb_all_doc(self.x_test, self.labels_test)

        self.x_train = DataHelperML.build_input_data(self.x_train, self.vocab, self.doc_level_data)
        self.x_train = self.pad_sentences(self.x_train, target_length=self.target_sent_len)
        self.x_test = DataHelperML.build_input_data(self.x_test, self.vocab, self.doc_level_data)
        self.x_test = self.pad_sentences(self.x_test, target_length=self.target_sent_len)

        if self.doc_level_data == "doc" or self.doc_level_data == "comb":
            self.longest_sentence(self.x_train, False)  # find document with most sentences
            self.x_train = self.pad_document(self.x_train, target_length=self.target_doc_len)
            self.x_test = self.pad_document(self.x_test, target_length=self.target_doc_len)

        return [self.x_train, self.labels_train, self.vocab, self.vocab_inv, self.embed_matrix]

    def load_test_data(self):
        if self.x_test is not None:
            return [self.x_test, self.labels_test, self.vocab, self.vocab_inv, self.doc_size_test]
        else:
            print("nope")

    def get_file_id_test(self):
        return self.file_id_test

    def get_doc_label(self):
        return self.labels_test

