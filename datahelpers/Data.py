from enum import Enum


class AAData:
    def __init__(self, name, size):
        self.name = name
        self.size = size
        self.file_id = None
        self.raw = None
        self.value = None
        self.label_doc = None
        self.label_instance = None  # this is for sentences or comb or paragraph
        self.doc_size = None
        self.doc_size_trim = None

        self.vocab = None
        self.vocab_inv = None
        self.embed_matrix = None
        self.embed_matrix_w2v = None

    def init_empty_list(self):
        self.file_id = []
        self.raw = []
        self.value = []
        self.label_doc = []
        self.label_instance = []
        self.doc_size = []
        self.doc_size_trim = []


class LoadMethod(Enum):
    DOC = 1
    COMB = 2
    SENT = 3

