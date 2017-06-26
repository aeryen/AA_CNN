from enum import Enum


class AAData:
    def __init__(self, size):
        self.size = size
        self.file_id = None
        self.raw = None
        self.value = None
        self.label_doc = None
        self.label_instance = None  # this is for sentences or comb or paragraph
        self.doc_size = None
        self.doc_size_trim = None

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

