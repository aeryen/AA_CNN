import pkg_resources
import os
import numpy as np
import logging

from datahelpers.DataHelper import DataHelper
from datahelpers.Data import AAData
from datahelpers.Data import LoadMethod
from utils import featuremaker
import errno


class DataHelperML(DataHelper):
    def __init__(self, doc_level, embed_type, embed_dim, target_doc_len, target_sent_len, train_csv_file,
                 data_dir="ml_mulmol"):
        super(DataHelperML, self).__init__(doc_level=doc_level, embed_type=embed_type, embed_dim=embed_dim,
                                           target_doc_len=target_doc_len, target_sent_len=target_sent_len)

        self.training_data_dir = pkg_resources.resource_filename('datahelpers', 'data/' + data_dir + '/')
        self.train_label_file_path = self.training_data_dir + "_new_label/" + train_csv_file
        self.val_label_file_path = self.training_data_dir + "_new_label/val.csv"
        self.test_label_file_path = self.training_data_dir + "_new_label/test.csv"

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

    @staticmethod
    def load_raw_file(data_dir, author_name, file_name):
        if not os.path.exists(os.path.dirname(data_dir + author_name + "/")):
            logging.error("error: " + author_name + " does not exit")
            return
        file_content = open(data_dir + author_name + "/txt/txt-preprocessed/" + file_name, "r").readlines()
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

    def load_raw_dir(self, csv_file):
        authors, file_ids, label_matrix = DataHelperML.load_csv(csv_file_path=csv_file)
        self.num_of_classes = len(authors)

        data = AAData(size=len(file_ids))
        data.file_id = file_ids

        origin_list = [None] * data.size
        doc_size = [None] * data.size

        folder_list = os.listdir(self.training_data_dir)
        for author in folder_list:
            f = self.training_data_dir + author + "/txt/txt-preprocessed/"
            if os.path.isdir(f):
                sub_file_list = os.listdir(f)
                for file_name in sub_file_list:
                    if file_name in data.file_id:
                        index = data.file_id.index(file_name)
                        file_content = DataHelperML.load_raw_file(data_dir=self.training_data_dir,
                                                                  author_name=author, file_name=file_name)
                        self.temp_write_channel_file(author, file_name, file_content)
                        origin_list[index] = file_content
                        doc_size[index] = len(file_content)

        doc_size = np.array(doc_size)

        data.raw = origin_list
        data.label_doc = label_matrix
        data.doc_size = doc_size

        return data

    @staticmethod
    def load_proced_file(data_dir, author_code, file_name):
        if not os.path.exists(os.path.dirname(data_dir + author_code + "/")):
            logging.error("error: " + author_code + " does not exit")
            return
        file_content = open(data_dir + author_code + "/" + file_name, "r").readlines()
        file_content = [line.split() for line in file_content]
        return file_content

    def load_proced_dir(self, csv_file):
        authors, file_ids, label_matrix = DataHelper.load_csv(csv_file_path=csv_file)
        self.num_of_classes = len(authors)

        data = AAData(size=len(file_ids))
        data.file_id = file_ids

        origin_list = [None] * data.size
        doc_size = [None] * data.size

        folder_list = os.listdir(self.training_data_dir)
        for author in folder_list:
            f = self.training_data_dir + author
            if os.path.isdir(f):
                sub_file_list = os.listdir(f)
                for file_name in sub_file_list:
                    if file_name in data.file_id:
                        index = data.file_id.index(file_name)
                        file_content = DataHelperML.load_proced_file(data_dir=self.training_data_dir,
                                                                     author_code=author, file_name=file_name)
                        origin_list[index] = file_content
                        doc_size[index] = len(file_content)

        doc_size = np.array(doc_size)

        data.raw = origin_list
        data.label_doc = label_matrix
        data.doc_size = doc_size

        return data

    def pad_sentences(self, data):
        if self.target_sent_len > 0:
            max_length = self.target_sent_len
        else:
            sent_lengths = [[len(sent) for sent in doc] for doc in data.value]
            max_length = max(sent_lengths)
            print("longest doc: " + str(max_length))

        padded_docs = []
        for doc in data.value:
            padded_doc = []
            for sent_i in range(len(doc)):
                sent = doc[sent_i]
                if len(sent) <= max_length:
                    num_padding = max_length - len(sent)
                    new_sentence = np.concatenate([sent, np.zeros(num_padding, dtype=np.int)])
                else:
                    new_sentence = sent[:max_length]
                padded_doc.append(new_sentence)
            padded_docs.append(np.array(padded_doc))
            data.value = np.array(padded_docs)
        return data

    def pad_document(self, docs, padding_word="<PAD>", target_length=-1):
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

    @staticmethod
    def flatten_doc_to_sent(data):
        expand_raw = []
        expand_vector = []
        expand_y = []

        for x_doc in data.raw:
            expand_raw.extend(x_doc)
        for x_doc in data.value:
            expand_vector.extend(x_doc)
        for i in range(len(data.label_doc)):
            expand_y.extend(np.tile(data.label_doc[i], [len(data.raw[i]), 1]))

        data.raw = expand_raw
        data.value = np.array(expand_vector)
        data.label_instance = np.array(expand_y)
        return data

    def build_content_vector(self, data):
        unk = self.vocab["<UNK>"]
        # if self.doc_level_data == LoadMethod.DOC or self.doc_level_data == LoadMethod.COMB:
        content_vector = np.array([[[self.vocab.get(word, unk) for word in sent] for sent in doc] for doc in data.raw])
        data.value = content_vector
        # else:
        #     x = np.array([[self.vocab.get(word, unk) for word in doc] for doc in data])
        return data
