import collections
import errno
import itertools
import math
import os
import pickle
import pkg_resources
import logging

import numpy as np

from utils import featuremaker
from datahelpers.DataHelper import DataHelper

# THIS FILE LOADS PAN11 DATA
# CODE 0 IS SMALL TRAINING AND TESTING, CODE 1 IS LARGE TRAINING AND TESTING


class DataHelperMulMol6(DataHelper):
    Record = collections.namedtuple('Record', ['file', 'author', 'content'])
    problem_name = "ML"

    training_data_dir = "./data/ml_dataset/"
    truth_file_path = "./data/ml_dataset/labels.csv"

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

    vocab = None
    vocab_inv = None
    embed_matrix = None

    doc_level_data = False
    target_sent_len = None
    target_doc_len = None

    def __init__(self, doc_level=False, embed_dim=100, target_sent_len=220, target_doc_len=100):
        logging.info("Data Helper: " + __file__ + " initiated.")

        super(DataHelperMulMol6, self).__init__(doc_level=doc_level, embed_type=embed_type, embed_dim=embed_dim,
                                                target_doc_len=target_doc_len, target_sent_len=target_sent_len,
                                                train_holdout=train_holdout)

        self.training_data_dir = pkg_resources.resource_filename('datahelpers', 'data/ml_mulmol/')
        self.truth_file_path = self.training_data_dir + "labels.csv"

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
        # label_matrix = np.matrix(label_matrix)

        file_id_ordered = []
        label_matrix_ordered = []
        data_list = []
        doc_size = []

        folder_list = os.listdir(self.training_data_dir)
        for author in folder_list:
            f = self.training_data_dir + author
            if os.path.isdir(f):
                f += "/txt/txt-preprocessed/"
                sub_file_list = os.listdir(f)
                for file_name in sub_file_list:
                    if file_name in file_id_list:
                        content = DataHelperMulMol6.read_one_file(f + file_name)
                        data_list.append(content)  # document level array instead of all sentence
                        # self.write_channel_file(author, file_name, content)
                        # data_list.extend(content)
                        index = file_id_list.index(file_name)
                        file_id_ordered.append(file_name)
                        label_matrix_ordered.append(label_matrix[index])  # document level array
                        # label_matrix_ordered.extend(np.tile(label_matrix[index], [len(content), 1]))
                        doc_size.append(len(content))

        label_matrix_ordered = np.array(label_matrix_ordered)
        doc_size = np.array(doc_size)

        return [file_id_ordered, label_matrix_ordered, data_list, doc_size]

    def build_embedding(self, vocabulary_inv, glove_words, glove_vectors):
        np.random.seed(10)
        embed_matrix = []
        std = np.std(glove_vectors[0, :])
        for word in vocabulary_inv:
            if word in glove_words:
                word_index = glove_words.index(word)
                embed_matrix.append(glove_vectors[word_index, :])
            else:
                embed_matrix.append(np.random.normal(loc=0.0, scale=std, size=self.embedding_dim))
        embed_matrix = np.array(embed_matrix)
        return embed_matrix

    @staticmethod
    def build_input_data(docs, vocabulary, doc_level):
        unk = vocabulary["<UNK>"]
        if doc_level:
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
            print("longest doc: " + str(max_length))

        if not self.doc_level_data:
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
            print("longest doc: " + str(tar_length))

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

    def train_test_split(self, file_id, labels, x, doc_size):
        np.random.seed(10)
        shuffle_i = np.random.permutation(np.arange(len(labels)))

        file_id_shuffled = [file_id[i] for i in shuffle_i]
        labels_shuffled = labels[shuffle_i]
        x_shuffled = [x[i] for i in shuffle_i]
        doc_size_shuffled = doc_size[shuffle_i]

        self.train_size = int(math.floor(len(labels) * 0.75))
        self.test_size = len(file_id) - self.train_size
        file_id_training, file_id_test = file_id_shuffled[:self.train_size], file_id_shuffled[self.train_size:]
        labels_training, labels_test = labels_shuffled[:self.train_size], labels_shuffled[self.train_size:]
        x_training, x_test = x_shuffled[:self.train_size], x_shuffled[self.train_size:]
        doc_size_training, doc_size_test = doc_size_shuffled[:self.train_size], doc_size_shuffled[self.train_size:]

        return [file_id_training, file_id_test, labels_training, labels_test,
                x_training, x_test, doc_size_training, doc_size_test]

    def expand_sentence(self, x, y):
        expand_x = []
        for x_doc in x:
            expand_x.extend(x_doc)
        expand_y = []
        for i in range(len(y)):
            expand_y.extend(np.tile(y[i], [len(x[i]), 1]))
        return [expand_x, expand_y]

    def load_data(self):
        # o = DataHelper(file_to_load)
        file_id, labels, data_list, doc_size = self.__load_data()

        [self.file_id_train, self.file_id_test, self.labels_train, self.labels_test,
         self.x_train, self.x_test, self.doc_size_train, self.doc_size_test] = \
            self.train_test_split(file_id, labels, data_list, doc_size)

        x_training_exp, labels_training_exp = self.expand_sentence(self.x_train, self.labels_train)
        x_test_exp, labels_test_exp = self.expand_sentence(self.x_test, self.labels_test)

        x_concat_exp = np.concatenate([x_training_exp, x_test_exp], axis=0)
        self.longest_sentence(x_concat_exp, True)
        self.vocab, self.vocab_inv = self.build_vocab(x_concat_exp, self.vocabulary_size)
        pickle.dump([self.vocab, self.vocab_inv], open("ml_vocabulary.pickle", "wb"))

        if not self.doc_level_data:
            self.x_train = x_training_exp
            self.labels_train = labels_training_exp
            self.x_test = x_test_exp
            self.labels_test = labels_test_exp

        [glove_words, glove_vectors] = self.load_glove_vector(self.glove)
        self.embed_matrix = self.build_embedding(self.vocab_inv, glove_words, glove_vectors)

        self.x_train = DataHelperMulMol6.build_input_data(self.x_train, self.vocab, self.doc_level_data)
        self.x_train = self.pad_sentences(self.x_train, target_length=self.target_sent_len)

        self.x_test = DataHelperMulMol6.build_input_data(self.x_test, self.vocab, self.doc_level_data)
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
    o = DataHelperMulMol6(doc_level=True)
    o.load_data()
    # o.load_test_data()
    print("o")
