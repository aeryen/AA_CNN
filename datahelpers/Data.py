from enum import Enum


class AAData:
    def __init__(self, size):
        self.size = size
        self.file_id = None
        self.raw = None
        self.label = None
        self.doc_size = None


class LoadMethod(Enum):
    DOC = 1
    COMB = 2
    SENT = 3

