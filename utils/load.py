import gensim
import os
import re
from nltk.tokenize import RegexpTokenizer
# from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim.models.doc2vec import TaggedDocument
import numpy as np


def get_fnames(dataset_dir, csv):
    corpus = []
    with open(dataset_dir + '/' + csv, "r") as fin:
        authors = []
        y = []
        firstLine = True
        for line in fin:
            if firstLine == True:
                authors = line.strip().split(",")[1:]
                firstLine = False
            else:
                currentAuthor = None
                tokens = line.strip().split(",")
                for author in authors:
                    if author in tokens[0]:
                        currentAuthor = author
                        break
                corpus.append(dataset_dir + '/' + currentAuthor + '/txt/txt-preprocessed/' + tokens[0])
                labels = list(map(int, tokens[1:]))
                y.append(labels)

    return corpus, np.array(y)


def get_doc_list(folder_name, csv):
    doc_list = []
    file_list, y = get_fnames(folder_name, csv)
    for file in file_list:
        st = open(file, 'r').read()
        doc_list.append(st)
    return doc_list, y


def get_doc(folder_name, csv):
    doc_list, y = get_doc_list(folder_name, csv)
    tokenizer = RegexpTokenizer(r'\w+')
    # en_stop = get_stop_words('en')
    p_stemmer = PorterStemmer()

    taggeddoc = []

    texts = []
    for index, i in enumerate(doc_list):
        # for tagged doc
        wordslist = []
        tagslist = []

        # clean and tokenize document string
        # raw = i.lower()
        tokens = tokenizer.tokenize(i)

        # remove stop words from tokens
        # stopped_tokens = [i for i in tokens if not i in en_stop]

        # remove numbers
        # number_tokens = [re.sub(r'[\d]', ' ', i) for i in stopped_tokens]
        # number_tokens = ' '.join(number_tokens).split()

        # stem tokens
        # stemmed_tokens = [p_stemmer.stem(i) for i in number_tokens]
        # remove empty
        # length_tokens = [i for i in stemmed_tokens if len(i) => 1]
        # add tokens to list
        # texts.append(length_tokens)
        try:
            td = TaggedDocument(gensim.utils.to_unicode(str.encode(' '.join(tokens))).split(), ["SENT_" + str(index)])
            taggeddoc.append(td)
        except UnicodeDecodeError:
            print(index)
    assert (len(taggeddoc) == len(y))
    return taggeddoc, y
