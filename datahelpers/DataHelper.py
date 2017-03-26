import numpy as np
import re
from collections import Counter
import itertools
import gensim


class DataHelper(object):

    def __init__(self):
        pass

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

    @staticmethod
    def read_one_file(file_path):
        # if "tom_mitchell_3.txt" in file_path:
        #     print "huh"

        file_content = open(file_path, "r").readlines()
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

    @staticmethod
    def load_glove_vector(glove_path):
        glove_lines = list(open(glove_path, "r").readlines())
        glove_lines = [s.split(" ", 1) for s in glove_lines if (len(s) > 0 and s != "\n")]
        glove_words = [s[0] for s in glove_lines]
        vector_list = [s[1] for s in glove_lines]
        glove_vectors = np.array([np.fromstring(line, dtype=float, sep=' ') for line in vector_list])
        return [glove_words, glove_vectors]

    @staticmethod
    def load_w2v_vector():
        word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(
            '../datahelpers/w2v/GoogleNews-vectors-negative300.bin',
            binary=True)
        return word2vec_model

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
        print("longest content: " + str(max(content_len)))
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
            doc = doc.split()
            x.append(doc)
            y[global_index, author_code_map[record.author]] = 1
            global_index += 1
        return x, y

    @staticmethod
    def build_vocab(data, vocabulary_size):
        """
        Builds a vocabulary mapping from word to index based on the sentences.
        Returns vocabulary mapping and inverse vocabulary mapping.
        """
        # Build vocabulary
        word_counts = Counter(itertools.chain(*data))
        # Mapping from index to word
        vocabulary_inv = [x[0] for x in word_counts.most_common()]
        vocabulary_inv.insert(0, "<PAD>")
        vocabulary_inv.insert(1, "<UNK>")

        print "size of vocabulary: " + str(len(vocabulary_inv))
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
