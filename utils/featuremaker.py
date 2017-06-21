import nltk
import numpy as np
import readability

from nltk.tag import StanfordPOSTagger
from nltk.parse.stanford import StanfordParser
from unidecode import unidecode


class FeatureMaker:

    _sentence_data = None
    _split_data = None
    _stf_pos_tagger = None
    _stf_parser = None

    _pos_list = []
    _neg_list = []

    def __init__(self, data):
        self._split_data = data
        self._sentence_data = [" ".join(line) for line in self._split_data]

    def _pos_tag_sent(self, sent):
        # text = word_tokenize("And now for something completely different")
        return nltk.pos_tag(sent)

    def _sf_pos_tag_sent(self, sent):
        return self._stf_pos_tagger.tag(sent)

    def prefix_suffix(self):
        prefix_2 = []
        prefix_3 = []
        suffix_2 = []
        suffix_3 = []
        for line in self._split_data:
            prefix_2.append([w[:2] for w in line])
            prefix_3.append([w[:3] for w in line])
            suffix_2.append([w[-2:] for w in line])
            suffix_3.append([w[-3:] for w in line])

        return [prefix_2, prefix_3, suffix_2, suffix_3]

    def fast_pos_tag(self):
        tag_result = [[token[1] for token in self._pos_tag_sent(line)] for line in self._split_data]
        return tag_result

    def pos_tag(self):
        if self._stf_pos_tagger is None:
            self._stf_pos_tagger = StanfordPOSTagger('english-bidirectional-distsim.tagger')
        index = 0
        tag_result = []
        while index < len(self._split_data):
            temp = self._stf_pos_tagger.tag_sents(self._split_data[index:index+1000])
            tag_result.extend(temp)
            index += 1000
            print(("pos:" + str(index)), end=' ')
        # tag_result = self._stf_pos_tagger.tag_sents(self._split_data)
        tag_result = [[unidecode(p[1]) for p in line] for line in tag_result]

        # for line in tag_result:
        #     print str(line)
        return tag_result

    def parser(self):
        if self._stf_parser is None:
            self._stf_parser = StanfordParser(model_path="edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz")
        result = self._stf_parser.parse_sents(self._split_data)
        result = sum([[parse for parse in dep_graphs] for dep_graphs in result], [])
        for i in result:
            print(i)

    def per_word_length(self):
        wl_result = [[len(w) for w in line] for line in self._split_data]
        return wl_result

    def sentence_avg_word_length(self):
        wl_result = self.per_word_length()
        wl_result = [np.mean(line) for line in wl_result]
        return wl_result

    def sentence_length(self):
        sl_result = [len(line) for line in self._split_data]
        return sl_result

    def sentence_length_mean_sd(self):
        return np.mean(self.sentence_length()), np.std(self.sentence_length())

    def load_sentiment_list(self):
        if not self._pos_list:
            with open("./../pos_neg/positive-words.txt", mode='r') as f:
                file_content = f.readlines()
                for line in file_content:
                    line = line.strip()
                    if not line.startswith(";") and line:
                        self._pos_list.append(line)
        if not self._neg_list:
            with open("./../pos_neg/negative-words.txt", mode='r') as f:
                file_content = f.readlines()
                for line in file_content:
                    line = line.strip()
                    if not line.startswith(";") and line:
                        self._neg_list.append(line)
        return [self._pos_list, self._neg_list]

    def sentiment_sequence(self):
        sentiment_data = []
        for line in self._split_data:
            sentiment_line = []
            for word in line:
                if word in self._pos_list:
                    sentiment_line.append("POS")
                elif word in self._neg_list:
                    sentiment_line.append("NEG")
                else:
                    sentiment_line.append("NON")
            sentiment_data.append(sentiment_line)
        return sentiment_data

    def get_read_measure(self):
        value_list = []
        for cat, data in list(readability.getmeasures(self._sentence_data, lang='en').items()):
            print(('%s:' % cat))
            for key, val in list(data.items()):
                print((('    %-20s %12.2f' % (key + ':', val)).rstrip('0 ').rstrip('.')))

            value_list.append(val)
        return val

if __name__ == "__main__":
    sent_data = ["I am looking for the person who paid for my wine last Friday evening .",
                 "I have not heard anything from American Airlines .",
                 "Could you please forward to Bev ?",
                 "That place is so very beautiful .",
                 "I hate the coffee there ."]

    sent_data = [line.split() for line in sent_data]

    FM = FeatureMaker(sent_data)
    [prefix_2, prefix_3, suffix_2, suffix_3] = FM.prefix_suffix()

    FM.pos_tag()
    FM.parser()
    print((FM.per_word_length()))
    print((FM.sentence_avg_word_length()))
    print((FM.sentence_length()))
    FM.load_sentiment_list()
    FM.sentiment_sequence()

    FM.get_read_measure()

