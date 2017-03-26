import numpy as np
import re
import itertools
import pickle
import io
from unidecode import unidecode
from collections import Counter
from datahelpers.DataHelper import DataHelper

# THIS FILE LOADS PAN12 DATA AND CONVERT EACH TRAINING DOCUMENT INTO A LONG VECTOR
# PRETENDING EACH DOCUMENT TO BE AN SENTENCE (DOCUMENT AS SENTENCE)
# IMPORT AND USE load_data() AND load_test_data()


class DataHelperPan12(DataHelper):
    author_codes_A = ["A", "B", "C"]
    #                  0    1    2
    author_codes_C = ["A", "B", "C", "D", "E", "F", "G", "H"]
    #                  0    1    2    3    4    5    6    7
    author_codes_I = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N"]
    #                  0    1    2    3    4    5    6    7    8     9   10   11   12   13

    author_codes = None

    file_name_A = ["12Atrain", ".txt"]
    file_name_C = ["12Ctrain", ".txt"]
    file_name_I = ["12Itrain", ".TXT"]

    file_name = None

    test_file_index_A = ["01", "02", "03", "04", "05", "06"]  # TODO
    test_author_index_A = [1, 0, 0, 2, 2, 1]  # TODO

    test_file_index_C = ["01", "02", "03", "04", "05", "06", "07", "08"]
    test_author_index_C = [2, 4, 0, 5, 7, 1, 6, 3]

    test_file_index_I = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12", "13", "14"]
    test_author_index_I = [4, 3, 6, 9, 8, 10, 0, 7, 5, 2, 1, 12, 13, 11]

    test_file_index = None
    test_author_index = None

    test_file_name_A = ["12Atest", ".txt"]
    test_file_name_C = ["12Ctest", ".txt"]
    test_file_name_I = ["12Itest", ".txt"]

    test_file_name = None

    test_sentence_cut = [26, 344, 1024]
    sentence_cut = None

    test_doc_cut = [7357, 15409, 500]
    doc_cut = None

    encodings = ["cp1252", "utf-8", "utf-8"]
    encode = None

    num_of_classes = 3

    embedding_dim = 300

    vocabulary_size = 25000

    problem_name = None

    def set_problem(self, target_ac, embedding_dim):
        if target_ac == self.author_codes_A or target_ac == self.author_codes_C or target_ac == self.author_codes_I:
            self.author_codes = target_ac
        else:
            print "wrong parameter"
        self.num_of_classes = len(self.author_codes)
        if self.author_codes == self.author_codes_A:
            self.file_name = self.file_name_A
            self.test_file_index = self.test_file_index_A
            self.test_author_index = self.test_author_index_A
            self.test_file_name = self.test_file_name_A
            self.sentence_cut = self.test_sentence_cut[0]
            self.problem_name = "A"
            self.encode = self.encodings[0]
            self.doc_cut = self.test_doc_cut[0]
        elif self.author_codes == self.author_codes_C:
            self.file_name = self.file_name_C
            self.test_file_index = self.test_file_index_C
            self.test_author_index = self.test_author_index_C
            self.test_file_name = self.test_file_name_C
            self.sentence_cut = self.test_sentence_cut[1]
            self.problem_name = "C"
            self.encode = self.encodings[1]
            self.doc_cut = self.test_doc_cut[1]
        elif self.author_codes == self.author_codes_I:
            self.file_name = self.file_name_I
            self.test_file_index = self.test_file_index_I
            self.test_author_index = self.test_author_index_I
            self.test_file_name = self.test_file_name_I
            self.sentence_cut = self.test_sentence_cut[2]
            self.problem_name = "I"
            self.encode = self.encodings[2]
            self.doc_cut = self.test_doc_cut[2]
        self.embedding_dim = embedding_dim

    @staticmethod
    def clean_str(string):
        # TODO: need more thought
        """
        Tokenization/string cleaning for all datasets except for SST.
        Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
        """
        # string = re.sub(r"[^A-Za-z0-9(),!?\'`\"]", " ", string)
        if isinstance(string, unicode):
            string = re.sub(u"\u2018", u"'", string)
            string = re.sub(u"\u2019", u"'", string)
            string = re.sub(u"\u201c", u"\"", string)
            string = re.sub(u"\u201d", u"\"", string)
            string = re.sub(u"\u2014", " <EMDASH> ", string)
            string = re.sub(u"\u2026", " <ELLIPSIS> ", string)

            string = unidecode(string)

        string = re.sub("\"([\w])", "\" \\1", string)
        string = re.sub("([\w])\"", "\\1 \"", string)

        string = re.sub("-([\w])", "- \\1", string)
        string = re.sub("([\w])-", "\\1 -", string)

        string = re.sub("([\w])/([\w])", "\\1 / \\2", string)

        string = re.sub(r"\'s", " \'s", string)
        string = re.sub(r"\'ve", " \'ve", string)
        string = re.sub(r"n\'t", " n\'t", string)
        string = re.sub(r"\'re", " \'re", string)
        string = re.sub(r"\'d", " \'d", string)
        string = re.sub(r"\'ll", " \'ll", string)
        string = re.sub(r",", " , ", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"\.", " . ", string)
        string = re.sub(r"\(", " ( ", string)
        string = re.sub(r"\)", " ) ", string)
        string = re.sub(r"\?", " ? ", string)
        string = re.sub(r"\s{2,}", " ", string)

        string = string + " <LB>"

        # print string

        # string = string.encode("ascii")
        return string.strip().lower()

    def load_data_and_labels(self):
        # Load data from files
        y_identity = np.identity(self.num_of_classes)

        x = []
        y = None

        total_instance = 0
        doc_sentence = []
        author_sentence = [0] * self.num_of_classes
        for author_index in range(self.num_of_classes):
            author_code = self.author_codes[author_index]
            for work_index in ["1", "2"]:
                file_full_path = "./data/pan12-train/" +\
                                 self.file_name[0] + author_code + work_index + self.file_name[1]
                train_content = list(io.open(file_full_path, mode="r", encoding=self.encode).readlines())
                train_content = [s.strip() for s in train_content]
                train_content = [s for s in train_content if len(s) > 0]
                print file_full_path + "\t\t" + str(len(train_content))
                doc_sentence.append(len(train_content))
                author_sentence[author_index] += len(train_content)
                total_instance += len(train_content)

                # Split by words
                x_text = [self.clean_str(sent) for sent in train_content]

                for train_line_index in range(len(x_text)):
                    tokens = x_text[train_line_index].split()

                    if len(tokens) > self.sentence_cut:
                        print str(len(tokens)) + "\t" + x_text[train_line_index]
                        tokens = tokens[:self.sentence_cut]
                        print "\t### Force Cut"
                        # print "\t" + str(len(tokens)) + "\t" + x_text[train_line_index]
                    x.append(tokens)
                    if y is None:
                        y = np.expand_dims(y_identity[author_index, :], axis=0)
                    else:
                        y = np.concatenate([y, np.expand_dims(y_identity[author_index, :], axis=0)], axis=0)

        print "TOTAL: " + str(total_instance)
        print "AUTHOR SENTENCE: " + str(author_sentence)
        return [x, y, doc_sentence, author_sentence]

    def load_test_data_and_labels(self):
        """
        Loads MR polarity data from files, splits the data into words and generates labels.
        Returns split sentences and labels.
        """
        # Load data from files
        y_identity = np.identity(self.num_of_classes)

        x = []
        y = None
        file_sizes = []
        for file_index in range(len(self.test_file_index)):
            file_code = self.test_file_index[file_index]
            file_full_path = "./data/pan12-test/" + self.test_file_name[0] + file_code + self.test_file_name[1]
            train_content = list(io.open(file_full_path, mode="r", encoding=self.encode).readlines())
            train_content = [s.strip() for s in train_content]
            train_content = [s for s in train_content if len(s) > 0]
            print file_full_path + "\t\t" + str(len(train_content))
            file_sizes.append(len(train_content))

            # Split by words
            x_text = [self.clean_str(sent) for sent in train_content]

            for train_line_index in range(len(x_text)):
                tokens = x_text[train_line_index].split()

                if len(tokens) > self.sentence_cut:
                    print str(len(tokens)) + "\t" + x_text[train_line_index]
                    tokens = tokens[:self.sentence_cut]
                    print "\t### Force Cut"
                    # print "\t" + str(len(tokens)) + "\t" + x_text[train_line_index]
                x.append(tokens)
                if y is None:
                    y = np.expand_dims(y_identity[self.test_author_index[file_index], :], axis=0)
                else:
                    y = np.concatenate([y, np.expand_dims(y_identity[self.test_author_index[file_index], :], axis=0)], axis=0)

        return [x, y, file_sizes]

    def pad_sentences(self, sentences, padding_word="<PAD>", target_length=-1):
        """
        Pads all sentences to the same length. The length is defined by the longest sentence.
        Returns padded sentences.
        """
        if target_length > 0:
            max_length = target_length
        else:
            review_lengths = [len(x) for x in sentences]
            max_length = max(review_lengths)

        padded_sentences = []
        for i in range(len(sentences)):
            rev = sentences[i]
            if len(rev) <= max_length:
                num_padding = max_length - len(rev)
                new_sentence = np.concatenate([rev, np.zeros(num_padding, dtype=np.int)])
            else:
                new_sentence = rev[:max_length]
            padded_sentences.append(new_sentence)
        return np.array(padded_sentences)

    def build_input_data(self, reviews, labels, vocabulary):
        """
        Maps sentencs and labels to vectors based on a vocabulary.
        """
        x = np.array([[vocabulary.get(word, vocabulary["<UNK>"]) for word in rev] for rev in reviews])
        y = np.array(labels)
        return [x, y]

    def load_glove_vector(self):
        glove_lines = list(open("./glove.6B."+str(self.embedding_dim)+"d.txt", "r").readlines())
        glove_lines = [s.split(" ", 1) for s in glove_lines if (len(s) > 0 and s != "\n")]
        glove_words = [s[0] for s in glove_lines]
        vector_list = [s[1] for s in glove_lines]
        glove_vectors = np.array([np.fromstring(line, dtype=float, sep=' ') for line in vector_list])
        return [glove_words, glove_vectors]

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

    def balance_data(self, x, y, author_sent_counts):
        max_sent_count = max(author_sent_counts)
        np.random.seed(10)

        global_index = 0
        for author_index in range(len(author_sent_counts)):
            author_count = author_sent_counts[author_index]
            dif = max_sent_count - author_count
            indexes = np.random.choice(np.arange(global_index, global_index+author_count-1), dif)
            new_inst_x = x[indexes]
            new_inst_y = y[indexes]
            x = np.concatenate([x, new_inst_x], axis=0)
            y = np.concatenate([y, new_inst_y], axis=0)
            global_index += author_count

        return x, y

    def concat_doc(self, x, y, document_sentence, chunk_size=1000):
        doc = []
        doc_y = []
        s = 0
        chunk_per_doc = []
        for i in document_sentence:
            d = np.concatenate(x[s:s + i], axis=0)
            d_len = d.size
            chk_count = 0
            if chunk_size > 0:
                chk_index = 0
                while chk_index < d_len:
                    doc.append(d[chk_index:chk_index+chunk_size])
                    chk_index += chunk_size
                    chk_count += 1
                    doc_y.append(y[s])
            else:
                doc.append(d)
                doc_y.append(y[s])
                chk_count = 1
            chunk_per_doc.append(chk_count)
            s += i
            print "doc size: " + str(d.size) + "\tchunk size: " + str(chk_count)
        return doc, doc_y, chunk_per_doc

    def load_data(self):
        """
        Loads and preprocessed data for the MR dataset.
        Returns input vectors, labels, vocabulary, and inverse vocabulary.
        """
        # Load and preprocess data
        sentences, labels, doc_sentence, author_sentence = self.load_data_and_labels()
        vocabulary, vocabulary_inv = self.build_vocab(sentences, self.vocabulary_size)
        pickle.dump([vocabulary, vocabulary_inv], open(self.file_name[0] + "_vocabulary.pickle", "wb"))

        [glove_words, glove_vectors] = self.load_glove_vector()
        embed_matrix = self.build_embedding(vocabulary_inv, glove_words, glove_vectors)
        # pickle.dump([embed_matrix], open("embed_matrix.pickle", "wb"))

        # sentences_padded, labels = self.pad_sentences(sentences, labels)  # TODO
        x, y = self.build_input_data(sentences, labels, vocabulary)

        x, y, chunk_per_doc = self.concat_doc(x, y, doc_sentence)

        y = np.array(y).astype(int)

        x = self.pad_sentences(x)

        # x, y = self.balance_data(x, y, author_sentence)

        return [x, y, vocabulary, vocabulary_inv, embed_matrix]

    def load_test_data(self):
        """
        Loads and preprocessed data for the MR dataset.
        Returns input vectors, labels, vocabulary, and inverse vocabulary.
        """
        # Load and preprocess data
        reviews, labels, file_sizes = self.load_test_data_and_labels()

        vocabulary, vocabulary_inv = pickle.load(open(self.file_name[0] + "_vocabulary.pickle", "rb"))

        x, y = self.build_input_data(reviews, labels, vocabulary)

        x, y, chunk_per_doc = self.concat_doc(x, y, file_sizes)

        y = np.array(y).astype(int)

        # x = self.pad_sentences(x, target_length=self.doc_cut)
        x = self.pad_sentences(x)
        return [x, y, vocabulary, vocabulary_inv, chunk_per_doc]

if __name__ == "__main__":
    dh = DataHelperPan12()
    dh.set_problem(DataHelperPan12.author_codes_C, 100)
    x, y, vocabulary, vocabulary_inv, embed_matrix = dh.load_data()
    x, y, vocabulary, vocabulary_inv, file_sizes = dh.load_test_data()
    print "a"
