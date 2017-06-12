from enum import Enum


class AAData:
    def __init__(self, size):
        self.size = size
        self.file_id = None
        self.raw = None
        self.value = None
        self.label = None
        self.doc_size = None

    def init_empty_list(self):
        self.file_id = []
        self.raw = []
        self.label = []
        self.doc_size = []


class LoadMethod(Enum):
    DOC = 1
    COMB = 2
    SENT = 3

