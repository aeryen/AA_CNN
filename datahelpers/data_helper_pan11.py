import collections
import logging
import pickle
import re
import itertools
import pkg_resources
import numpy as np

from collections import Counter

from datahelpers.DataHelper import DataHelper
from datahelpers.Data import AAData
from datahelpers.Data import LoadMethod


# THIS FILE LOADS PAN11 DATA
# CODE 0 IS SMALL TRAINING AND TESTING, CODE 1 IS LARGE TRAINING AND TESTING


class DataHelperPan11(DataHelper):
    Record = collections.namedtuple('Record', ['file', 'author', 'content'])
    Small_Author_Order = ['965226', '543719', 'x16198468117121', '997109', '464635',
                          '940826', '698837', 'x41117151611103260461997140', '636010',
                          'x159911161646101978', '580177', 'x659461599419513114', '255029',
                          '797711', 'x759461997140', 'x1297161459914695915', 'x6114104622172211485',
                          'x79721469971010', 'x64646297149114', 'x1546461997140', '900986', 'x8464695915',
                          'x12971784621399897149811', 'x10114697001411515', 'x0971411046297149114', '339173']

    Large_Author_Order = ['x9971451464197140', '904579', 'x711851046421971616', 'x1946461945161', '446543',
                          'x7599811482146199716151110', '676708', '601217', 'x99464635141110', 'x959941881469997154',
                          '280939', '995484', 'x169710974661110115', '965194', '428896', '265281', '640334', '363505',
                          '655031', 'x9710021462251212114', 'x611410469714101180', '454005', 'x8510021460111011411',
                          '894867', 'x1541881214699111499710', '664011', 'x1597882146981997',
                          'x611109716497104699979721', '273750', 'x79794671515114', 'x71185104612141151611',
                          'x811175151467516994110', '949679', '194257', 'x9464612141151611', '183141',
                          'x01716994461317538121', '761806', 'x64646719710', 'x31149780461019199', 'x95714631453159821',
                          'x1614979921463197999911101', '140914', '956947', '956112', '315875',
                          'x41180110461597851598171421', 'x997161641194681104971416', '730970',
                          'x05971097461599411816115', '100228', 'x997141651046991758897', '693864',
                          'x14110464972115811616', 'x16464681799995', 'x1519710469914971009788', 'x9746469971416510',
                          '425445', 'x151611811046719710', 'x897181114971611', 'x9111057974699971715411885',
                          'x1641199715469971416510', 'x61141046897181114971611', '339173', 'x097141411104635141110',
                          '559588', '648564', '769031', '658916', '935669', 'x10114697001411515', '580177']

    author_order = None
    training_dir = pkg_resources.resource_filename('datahelpers', 'data/pan11-training/')
    testing_dir = pkg_resources.resource_filename('datahelpers', 'data/pan11-test/')
    training_options = [training_dir + "SmallTrain.xml",
                        training_dir + "LargeTrain.xml"]
    testing_options = [testing_dir + "SmallTest.xml",
                       testing_dir + "LargeTest.xml"]
    truth_options = [testing_dir + "GroundTruthSmallTest.xml",
                     testing_dir + "GroundTruthLargeTest.xml"]

    problem_name_options = ["PAN11small", "PAN11large"]
    problem_name = None

    prob_code = None
    record_list = []

    def __init__(self, embed_type, embed_dim, target_sent_len, prob_code=1):

        logging.info("Data Helper: " + __file__ + " initiated.")
        super(DataHelperPan11, self).__init__(doc_level=LoadMethod.SENT, embed_type=embed_type, embed_dim=embed_dim,
                                              target_doc_len=None, target_sent_len=target_sent_len,
                                              total_fold=None, t_fold_index=None)

        self.prob_code = prob_code
        if prob_code == 0:
            self.author_order = self.Small_Author_Order
        elif prob_code == 1:
            self.author_order = self.Large_Author_Order
        else:
            print("code ERROR")

        self.problem_name = self.problem_name_options[self.prob_code]
        self.train_file = self.training_options[self.prob_code]
        self.test_file = self.testing_options[self.prob_code]
        self.test_truth_file = self.truth_options[self.prob_code]

        self.embed_matrix_glv = None
        self.embed_matrix_w2v = None

        self.load_train_data()
        self.load_test_data()

    @staticmethod
    def build_vocab(data, vocabulary_size):
        """
        Builds a vocabulary mapping from word to index based on the sentences.
        Returns vocabulary mapping and inverse vocabulary mapping.
        """
        # Build vocabulary
        word_counts = Counter(itertools.chain(*data.raw))
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
    def longest_sentence(data, print_content=True):
        sent_lengths = [len(x) for x in data.raw]
        result_index = sorted(list(range(len(sent_lengths))), key=lambda i: sent_lengths[i])[-30:]
        print("max: " + str(np.max(sent_lengths)))
        print("mean: " + str(np.mean(sent_lengths)))
        print("median: " + str(np.median(sent_lengths)))
        print("80 percentile: " + str(np.percentile(sent_lengths, q=80)))
        for i in result_index:
            s = data.raw[i]
            print(len(s))
            if print_content:
                print(s)

    def __load_train_data(self):
        data_list = []

        re_text_open = re.compile("\s*<text file=\"([\w/\.]*)\">\s*")
        re_text_clos = re.compile("\s*</text>\s*")
        re_auth = re.compile("\s*<author id=\"(\w*)\"/>\s*")
        re_body_open = re.compile("\s*<body>\s*")
        re_body_clos = re.compile("\s*</body>\s*")

        stages = ["text", "author", "body", "content", "body_clos", "text_clos"]
        expect_stage = "text"

        file_name = None
        author_name = None
        content = []

        file_content = open(self.train_file, "r").readlines()
        line_count = 0
        for l in file_content:
            line_count += 1
            l = l.strip()
            if l == "" or l == "<training>":
                pass
            elif l == "</training>":
                print("File Loading Ended: @ " + str(line_count))
                break
            elif expect_stage == "text":
                m = re_text_open.match(l)
                if m:
                    file_name = m.group(1)
                    expect_stage = "author"
                else:
                    print("ERROR")
            elif expect_stage == "author":
                m = re_auth.match(l)
                if m:
                    author_name = m.group(1)
                    expect_stage = "body"
                else:
                    print("ERROR")
            elif expect_stage == "body":
                m = re_body_open.match(l)
                if m:
                    expect_stage = "content"
                else:
                    print("ERROR")
            elif expect_stage == "content":
                m = re_body_clos.match(l)
                if m:
                    expect_stage = "text_clos"
                    r = self.Record(file=file_name, author=author_name, content=content)
                    data_list.append(r)
                    file_name = None
                    author_name = None
                    content = []
                else:
                    # l = self.clean_str(l)  # remove if no need
                    content.append(l)
            elif expect_stage == "text_clos":
                m = re_text_clos.match(l)
                if m:
                    expect_stage = "text"
                else:
                    print("ERROR")

        # for thing in data_list:
        #     print thing
        return data_list

    def __load_test_data(self):
        data_list = []

        re_text_open = re.compile("\s*<text file=\"([\w/\.]*)\">\s*")
        re_text_clos = re.compile("\s*</text>\s*")
        re_auth = re.compile("\s*<author id=\"(\w*)\"/>\s*")
        re_body_open = re.compile("\s*<body>\s*")
        re_body_clos = re.compile("\s*</body>\s*")

        stages_content = ["text", "body", "content", "body_clos", "text_clos"]
        stages_truth = ["text", "author", "text_clos"]
        expect_stage_content = "text"
        expect_stage_truth = "text"

        file_name = None
        author_name = None
        content = []

        file_content = open(self.test_file, "r").readlines()
        file_truth = open(self.test_truth_file, "r").readlines()

        content_line_index = 0
        truth_line_index = 0
        while content_line_index < len(file_content):
            l = file_content[content_line_index]
            content_line_index += 1
            l = l.strip()
            if l == "" or l == "<testing>":
                pass
            elif l == "</testing>":
                print("File Loading Ended: @ " + str(content_line_index))
                break
            elif expect_stage_content == "text":
                m = re_text_open.match(l)
                if m:
                    file_name = m.group(1)
                    expect_stage_content = "author"
                else:
                    print("ERROR")
            elif expect_stage_content == "author":
                content_line_index -= 1  # hold current line
                truth_loaded = False
                for truth_l in file_truth[truth_line_index:]:
                    truth_l = truth_l.strip()
                    truth_line_index += 1
                    if truth_l == "" or truth_l == "<testing>":
                        pass
                    elif expect_stage_truth == "text":
                        m = re_text_open.match(truth_l)
                        if m and m.group(1) == file_name:
                            expect_stage_truth = "author"
                        else:
                            print("ERROR: truth file mismatch")
                    elif expect_stage_truth == "author":
                        m = re_auth.match(truth_l)
                        if m:
                            author_name = m.group(1)
                            expect_stage_truth = "text_clos"
                            truth_loaded = True
                        else:
                            print("ERROR")
                    elif expect_stage_truth == "text_clos":
                        m = re_text_clos.match(truth_l)
                        if m:
                            expect_stage_truth = "text"
                            expect_stage_content = "body"
                            break
                        else:
                            print("ERROR")
                if not truth_loaded:
                    print("ERROR")
            elif expect_stage_content == "body":
                m = re_body_open.match(l)
                if m:
                    expect_stage_content = "content"
                else:
                    print("ERROR")
            elif expect_stage_content == "content":
                m = re_body_clos.match(l)
                if m:
                    expect_stage_content = "text_clos"
                    r = self.Record(file=file_name, author=author_name, content=content)
                    data_list.append(r)
                    file_name = None
                    author_name = None
                    content = []
                else:
                    # l = self.clean_str(l)  # remove if no need
                    content.append(l)
            elif expect_stage_content == "text_clos":
                m = re_text_clos.match(l)
                if m:
                    expect_stage_content = "text"
                else:
                    print("ERROR")

        # for thing in data_list:
        #     print thing
        return data_list

    def author_label(self, data_list):
        author_list = self.author_order
        author_count = {}
        for r in data_list:
            author_str = r.author
            if author_str not in author_list:
                print("WTF")
                author_list.append(author_str)  # ???
                author_count[author_str] = 1
                raise ValueError("AUTHOR NOT FOUND")
            else:
                if author_str in author_count:
                    author_count[author_str] += 1
                else:
                    author_count[author_str] = 1

        self.num_of_classes = len(author_list)

        return author_list, author_count

    def build_content_vector(self, data):
        """this method override global method because data is only sentence level"""
        unk = self.vocab["<UNK>"]
        data.value = np.array([[self.vocab.get(word, unk) for word in doc] for doc in data.raw])
        return data

    def pad_sentences(self, data):
        if self.target_sent_len > 0:
            max_length = self.target_sent_len
        else:
            sent_lengths = [len(sent) for sent in data.value]
            max_length = max(sent_lengths)
            print("longest doc: " + str(max_length))

        padded_docs = []
        trim_len = []
        for sent_i in range(len(data.value)):
            sent = data.value[sent_i]
            if len(sent) <= max_length:
                num_padding = max_length - len(sent)
                new_sentence = np.concatenate([sent, np.zeros(num_padding, dtype=np.int)])
                trim_len.append(data.doc_size[sent_i])

            else:
                new_sentence = sent[:max_length]
                trim_len.append(max_length)
            padded_docs.append(new_sentence)
        data.value = np.array(padded_docs)
        data.doc_size_trim = np.array(trim_len)
        return data

    def load_train_data(self):
        data_list = self.__load_train_data()
        author_list, author_count = self.author_label(data_list)
        x, y = self.xy_formatter(data_list, author_list)

        self.train_data = AAData("PAN11", len(data_list))
        self.train_data.raw = x
        self.train_data.label_doc = y
        self.train_data.label_instance = y
        self.train_data.doc_size = [len(doc) for doc in x]

        self.vocab, self.vocab_inv = self.build_vocab(self.train_data, self.vocabulary_size)
        pickle.dump([self.vocab, self.vocab_inv], open("pan11_vocabulary_" + str(self.prob_code) + ".pickle", "wb"))

        if self.embed_type == "glove" or self.embed_type == "both":
            self.embed_matrix_glv = self.build_glove_embedding(self.vocab_inv)
        if self.embed_type == "w2v" or self.embed_type == "both":
            self.embed_matrix_w2v = self.build_w2v_embedding(self.vocab_inv)

        self.train_data = self.build_content_vector(self.train_data)
        self.train_data = self.pad_sentences(self.train_data)

        self.train_data.embed_matrix = self.embed_matrix_glv
        self.train_data.embed_matrix_w2v = self.embed_matrix_w2v
        self.train_data.vocab = self.vocab
        self.train_data.vocab_inv = self.vocab_inv

    def load_test_data(self):
        data_list = self.__load_test_data()
        author_list, author_count = self.author_label(data_list)
        x, y = self.xy_formatter(data_list, author_list)

        if self.vocab is None and self.vocab_inv is None:
            self.vocab, self.vocab_inv = pickle.load(open("pan11_vocabulary_" + str(self.prob_code) + ".pickle", "rb"))

        self.test_data = AAData("PAN11", len(data_list))
        self.test_data.raw = x
        self.test_data.label_doc = y
        self.test_data.label_instance = y
        self.test_data.doc_size = [len(doc) for doc in x]

        self.test_data = self.build_content_vector(self.test_data)
        self.test_data = self.pad_sentences(self.test_data)

        self.test_data.embed_matrix = self.embed_matrix_glv
        self.test_data.embed_matrix_w2v = self.embed_matrix_w2v
        self.test_data.vocab = self.vocab
        self.test_data.vocab_inv = self.vocab_inv


if __name__ == "__main__":
    o = DataHelperPan11(embed_type="glove", embed_dim=300, target_sent_len=100, prob_code=1)
    data = o.get_train_data()
    DataHelperPan11.longest_sentence(data, True)
    # o.load_test_data()
    print("o")
