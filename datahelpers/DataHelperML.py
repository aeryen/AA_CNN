import pkg_resources
import os
import numpy as np
import logging

from datahelpers.DataHelper import DataHelper
from datahelpers.Data import AAData


class DataHelperML(DataHelper):

    def __init__(self, doc_level, embed_type, embed_dim, target_doc_len, target_sent_len, train_csv_file,
                 data_dir="ml_mulmol"):
        super(DataHelperML, self).__init__(doc_level=doc_level, embed_type=embed_type, embed_dim=embed_dim,
                                           target_doc_len=target_doc_len, target_sent_len=target_sent_len)

        self.training_data_dir = pkg_resources.resource_filename('datahelpers', 'data/'+data_dir+'/')
        self.train_label_file_path = self.training_data_dir + "_new_label/" + train_csv_file
        self.val_label_file_path = self.training_data_dir + "_new_label/val.csv"
        self.test_label_file_path = self.training_data_dir + "_new_label/test.csv"

    @staticmethod
    def load_raw_file(data_dir, author_code, file_name):
        if not os.path.exists(os.path.dirname(data_dir + author_code + "/")):
            logging.error("error: " + author_code + " does not exit")
            return
        file_content = open(data_dir + author_code + "/" + file_name, "r").readlines()
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
    def load_proced_file(data_dir, author_code, file_name):
        if not os.path.exists(os.path.dirname(data_dir + author_code + "/")):
            logging.error("error: " + author_code + " does not exit")
            return
        file_content = open(data_dir + author_code + "/" + file_name, "r").readlines()
        file_content = [line.split() for line in file_content]
        return file_content

    def load_origin_dir(self, csv_file, load_raw=False):
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
                        if load_raw:
                            file_content = DataHelperML.load_raw_file(author_code=author, file_name=file_name)
                        else:
                            file_content = DataHelperML.load_proced_file(author_code=author, file_name=file_name)
                        origin_list[index] = file_content
                        doc_size[index] = len(file_content)

        doc_size = np.array(doc_size)

        data.raw = origin_list
        data.label = label_matrix
        data.doc_size = doc_size

        return data


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

    @staticmethod
    def flatten_doc_to_sent(data):
        expand_x = []
        expand_y = []

        for x_doc in data.raw:
            expand_x.extend(x_doc)
        for i in range(len(data.label)):
            expand_y.extend(np.tile(data.label[i], [len(data.raw[i]), 1]))

        data.raw = expand_x
        data.label = expand_y
        return data

    @staticmethod
    def build_input_data(docs, vocabulary, doc_level):
        unk = vocabulary["<UNK>"]
        if doc_level:
            x = np.array([[[vocabulary.get(word, unk) for word in sent] for sent in doc] for doc in docs])
        else:
            x = np.array([[vocabulary.get(word, unk) for word in doc] for doc in docs])
        return x