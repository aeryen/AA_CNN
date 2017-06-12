import numpy as np
import re
from collections import Counter
import itertools
import gensim
import logging
import pkg_resources
import os
import math

from datahelpers.Data import AAData

class DataHelper(object):
    def __init__(self, doc_level, embed_type, embed_dim, target_doc_len, target_sent_len):
        logging.info("setting: %s is %s", "doc_level", doc_level)
        logging.info("setting: %s is %s", "embed_type", embed_type)
        logging.info("setting: %s is %s", "embed_dim", embed_dim)
        logging.info("setting: %s is %s", "target_doc_len", target_doc_len)
        logging.info("setting: %s is %s", "target_sent_len", target_sent_len)

        assert doc_level is not None
        assert embed_type is not None
        assert embed_dim is not None
        assert target_doc_len is not None
        assert target_sent_len is not None

        self.num_of_classes = None

        self.doc_level_data = doc_level
        self.embed_type = embed_type
        self.embedding_dim = embed_dim
        self.target_doc_len = target_doc_len
        self.target_sent_len = target_sent_len

        self.vocab = None
        self.vocab_inv = None
        self.embed_matrix = None
        self.vocabulary_size = 20000

        self.glove_dir = pkg_resources.resource_filename('datahelpers', 'glove/')
        self.glove_path = self.glove_dir + "glove.6B." + str(self.embedding_dim) + "d.txt"
        self.w2v_path = "./datahelpers/w2v/GoogleNews-vectors-negative300.bin"

        if self.embed_type == "glove":
            [self.glove_words, self.glove_vectors] = self.load_glove_vector()
        elif self.embed_type == "w2v":
            self.w2v_model = self.load_w2v_vector()
        elif self.embed_type == "both":
            [self.glove_words, self.glove_vectors] = self.load_glove_vector()
            self.w2v_model = self.load_w2v_vector()

    @staticmethod
    def clean_str(string):
        string = re.sub("\'", " \' ", string)
        string = re.sub("\"", " \" ", string)
        string = re.sub("-", " - ", string)
        string = re.sub("/", " / ", string)

        string = re.sub("[\d]+\.?[\d]*", "123", string)
        string = re.sub("[\d]+/[\d]+/[\d]{4}", "123", string)

        string = re.sub("[-]{4,}", " <<DLINE>> ", string)
        string = re.sub("-", " - ", string)
        string = re.sub(r"[~]+", " ~ ", string)

        string = re.sub(r",", " , ", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r":", " : ", string)
        string = re.sub(r"\.", " . ", string)
        string = re.sub(r"[(\[{]", " ( ", string)
        string = re.sub(r"[)\]}]", " ) ", string)
        string = re.sub(r"\?", " ? ", string)
        string = re.sub(r"\s{2,}", " ", string)

        return string.strip().lower().split()

    @staticmethod
    def split_sentence(paragraph):
        paragraph = paragraph.split(". ")
        paragraph = [e + ". " for e in paragraph if len(e) > 5]
        if paragraph:
            paragraph[-1] = paragraph[-1][:-2]
            paragraph = [DataHelper.clean_str(e) for e in paragraph]
        return paragraph

    def load_glove_vector(self):
        glove_lines = list(open(self.glove_path, "r", encoding="utf-8").readlines())
        glove_lines = [s.split(" ", 1) for s in glove_lines if (len(s) > 0 and s != "\n")]
        glove_words = [s[0] for s in glove_lines]
        vector_list = [s[1] for s in glove_lines]
        glove_vectors = np.array([np.fromstring(line, dtype=float, sep=' ') for line in vector_list])
        return [glove_words, glove_vectors]

    def load_w2v_vector(self):
        if os.path.exists(self.w2v_path):
            word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(self.w2v_path, binary=True)
        else:
            word2vec_model = gensim.models.KeyedVectors.load_word2vec_format('.' + self.w2v_path,  # one level up
                                                                             binary=True)
        return word2vec_model

    def build_glove_embedding(self, vocabulary_inv):
        np.random.seed(10)
        embed_matrix = []
        std = np.std(self.glove_vectors[0, :])
        for word in vocabulary_inv:
            if word in self.glove_words:
                word_index = self.glove_words.index(word)
                embed_matrix.append(self.glove_vectors[word_index, :])
            else:
                embed_matrix.append(np.random.normal(loc=0.0, scale=std, size=self.embedding_dim))
        embed_matrix = np.array(embed_matrix)
        return embed_matrix

    def build_w2v_embedding(self, vocabulary_inv):
        np.random.seed(10)
        embed_matrix = []
        std = np.std(self.w2v_model["the"])
        for word in vocabulary_inv:
            if word in self.w2v_model:
                embed_matrix.append(self.w2v_model[word])
            else:
                embed_matrix.append(np.random.normal(loc=0.0, scale=std, size=self.embedding_dim))
        embed_matrix = np.array(embed_matrix)
        return embed_matrix

    def build_embedding(self, vocabulary_inv):
        if self.embed_type == "glove":
            self.embed_matrix = self.build_glove_embedding(self.vocab_inv)
        else:
            self.embed_matrix = self.build_w2v_embedding(self.vocab_inv)

    @staticmethod
    def longest_sentence(input_list, print_content):
        sent_lengths = [len(x) for x in input_list]
        result_index = sorted(list(range(len(sent_lengths))), key=lambda i: sent_lengths[i])[-30:]
        for i in result_index:
            s = input_list[i]
            print(len(s))
            if print_content:
                print(s)

    @staticmethod
    def line_concat(data_list):
        """connect sentences in a record into a single string"""
        content_len = []
        for record in data_list:
            for l in record.content:
                l += " <LB>"
            record.content = " ".join(record.content)
            # record.content = self.clean_str()
            content_len.append(len(record.content))
        logging.info("longest content: " + str(max(content_len)))
        return data_list

    @staticmethod
    def xy_formatter(data_list, author_list):
        """attach lines to tokenized x, convert author names to one hot labels"""
        author_code_map = {}
        code = 0
        # map author name (author key) to a number (author code)
        for key in author_list:
            author_code_map[key] = code
            code += 1
        x = []
        y = np.zeros((len(data_list), len(author_list)))
        global_index = 0
        # attach string together then split to tokens, also generates one hot label
        for record in data_list:
            doc = " <LB> ".join(record.content)
            doc = DataHelper.clean_str(doc)
            x.append(doc)
            y[global_index, author_code_map[record.author]] = 1
            global_index += 1
        return x, y

    @staticmethod
    def chain(data_splits):
        for data in data_splits:
            for doc in data.raw:
                if doc is None:
                    print("here")
                for sent in doc:
                    for word in sent:
                        yield word

    @staticmethod
    def build_vocab(data, vocabulary_size):
        """
        Builds a vocabulary mapping from word to index based on the sentences.
        Returns vocabulary mapping and inverse vocabulary mapping.
        """
        # Build vocabulary
        word_counts = Counter(DataHelper.chain(data))
        # Mapping from index to word
        # vocabulary_inv = [x[0] for x in word_counts.most_common()]
        word_counts = sorted(word_counts.items(), key=lambda t: t[::-1], reverse=True)
        vocabulary_inv = [item[0] for item in word_counts]
        vocabulary_inv.insert(0, "<PAD>")
        vocabulary_inv.insert(1, "<UNK>")

        logging.info("size of vocabulary: " + str(len(vocabulary_inv)))
        # vocabulary_inv = list(sorted(vocabulary_inv))
        vocabulary_inv = list(vocabulary_inv[:vocabulary_size])  # limit vocab size

        # Mapping from word to index
        vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
        return [vocabulary, vocabulary_inv]

    @staticmethod
    def batch_iter(data, batch_size, num_epochs, shuffle=True):
        """
        Generates a batch iterator for a dataset.
        """
        data = np.array(data)
        data_size = len(data)
        num_batches_per_epoch = int(len(data) / batch_size)
        if len(data) % batch_size > 0:
            num_batches_per_epoch += 1
        logging.info("number of batches per epoch: " + str(num_batches_per_epoch))
        for epoch in range(num_epochs):
            # Shuffle the data at each epoch
            if shuffle:
                shuffle_indices = np.random.permutation(np.arange(data_size))
                shuffled_data = data[shuffle_indices]
            else:
                shuffled_data = data
            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, data_size)
                yield shuffled_data[start_index:end_index]

    @staticmethod
    def get_vocab_path(file_name, embed_type, embed_dim):
        current_path = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(current_path, file_name + "_" + embed_type + "_" + str(embed_dim) + ".pickle")

    @staticmethod
    def split_by_fold(num_of_fold, test_fold_index, data_size, data_to_split):
        fold_size = int(math.floor(data_size / num_of_fold))
        last_fold_size = data_size - (fold_size * (num_of_fold - 1))

        logging.info("number of fold: " + str(num_of_fold))
        logging.info("[testing fold index]: " + str(test_fold_index))
        logging.info("fold size: " + str(fold_size))
        logging.info("last fold size: " + str(last_fold_size))

        training_data = []
        testing_data = None
        for i in range(num_of_fold):
            if i == test_fold_index:
                if i == num_of_fold - 1:
                    testing_data = data_to_split[-last_fold_size:]
                else:
                    testing_data = data_to_split[i * fold_size:(i + 1) * fold_size]
            else:
                if i == num_of_fold - 1:
                    training_data.extend(data_to_split[-last_fold_size:])
                else:
                    training_data.extend(data_to_split[i * fold_size:(i + 1) * fold_size])

        return training_data, testing_data

    @staticmethod
    def split_by_fold_2(num_of_fold, test_fold_index, data):
        fold_size = int(math.ceil(data.size / num_of_fold))
        test_items = np.arange(fold_size) * num_of_fold + test_fold_index
        if test_items[-1] > data.size:
            test_items.pop(-1)

        train_data = AAData(size=data.size - len(test_items))
        train_data.init_empty_list()
        test_data = AAData(size=len(test_items))
        test_data.init_empty_list()
        for i in range(data.size):
            if i in test_items:
                test_data.file_id.append(data.file_id[i])
                test_data.raw.append(data.raw[i])
                test_data.value.append(data.value[i])
                test_data.label.append(data.label[i])
                test_data.doc_size.append(data.doc_size[i])
            else:
                train_data.file_id.append(data.file_id[i])
                train_data.raw.append(data.raw[i])
                train_data.value.append(data.value[i])
                train_data.label.append(data.label[i])
                train_data.doc_size.append(data.doc_size[i])

        return [train_data, test_data]

    @staticmethod
    def load_csv(csv_file_path):
        file_id_list = []
        label_matrix = []

        truth_file_content = open(csv_file_path, "r").readlines()
        author_list = truth_file_content[0].split(",")[1:]
        for line in truth_file_content[1:]:
            line = line.split(",")
            file_id_list.append(line[0])
            label_vector = list(map(int, line[1:]))
            label_matrix.append(np.array(label_vector))
        label_matrix = np.array(label_matrix)

        return author_list, file_id_list, label_matrix