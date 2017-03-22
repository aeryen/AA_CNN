from input_components.OneChannel import OneChannel
from input_components.SixChannel import SixChannel
from input_components.OneChannel_DocLevel import OneChannel_DocLevel
from middle_components.yifan_conv import YifanConv
from middle_components.parallel_conv import NParallelConvOnePoolNFC
from middle_components.parallel_size_joined_conv import NCrossSizeParallelConvNFC
from middle_components.parallel_joined_conv import ParallelJoinedConv
from middle_components.parallel_conv_DocLevel import NConvDocConvNFC
from output_components.pan_output import PANOutput
from output_components.ml_output import MLOutput


class TextCNN:
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    Works for both single label (PAN) and multilabel (ML) datasets
    """

    def __init__(
            self, sequence_length, num_classes, word_vocab_size,
            embedding_size, filter_sizes, num_filters, input_component="OneChannel", middle_component = 'OneCMiddle',
            dataset="ML", l2_reg_lambda=0.0, pref2_vocab_size=None, pref3_vocab_size=None, suff2_vocab_size=None,
            suff3_vocab_size=None, pos_vocab_size=None,init_embedding=None, dropout=False, batch_normalize = False,
            elu = False, n_conv=1, fc=[], document_length=-1):

        # input component
        if input_component == "OneChannel":
            self.input_comp = OneChannel(sequence_length, num_classes, word_vocab_size, embedding_size, init_embedding)
        if input_component == "OneChannel_DocLevel":
            self.input_comp = OneChannel_DocLevel(document_length, sequence_length, num_classes, word_vocab_size, embedding_size, init_embedding)
        elif input_component == "SixChannel":
            self.input_comp = SixChannel(sequence_length, num_classes, word_vocab_size, embedding_size,
                  pref2_vocab_size, pref3_vocab_size, suff2_vocab_size, suff3_vocab_size, pos_vocab_size,
                  init_embedding)
            self.input_pref2 = self.input_comp.input_pref2
            self.input_pref3 = self.input_comp.input_pref3
            self.input_suff2 = self.input_comp.input_suff2
            self.input_suff3 = self.input_comp.input_suff3
            self.input_pos = self.input_comp.input_pos
        else:
            raise NotImplementedError

        self.input_x = self.input_comp.input_x
        self.input_y = self.input_comp.input_y

        self.dropout_keep_prob = self.input_comp.dropout_keep_prob

        # middle component
        if middle_component == 'NParallelConvOnePoolNFC':
            self.middle_comp = NParallelConvOnePoolNFC(sequence_length, embedding_size, filter_sizes, num_filters,
                                                       previous_component=self.input_comp, dropout=dropout,
                                                       batch_normalize=batch_normalize, elu=elu, n_conv = n_conv, fc=fc)
        elif middle_component == 'ParallelJoinedConv':
            self.middle_comp = ParallelJoinedConv(sequence_length, embedding_size, filter_sizes, num_filters,
                                                  previous_component=self.input_comp, dropout=dropout,
                                                  batch_normalize=batch_normalize, elu=elu, n_conv=n_conv,
                                                  fc=fc)
        elif middle_component == 'NCrossSizeParallelConvNFC':
            self.middle_comp = NCrossSizeParallelConvNFC(sequence_length, embedding_size, filter_sizes, num_filters,
                                                         previous_component=self.input_comp, dropout=dropout,
                                                         batch_normalize=batch_normalize, elu=elu, n_conv=n_conv,
                                                         fc=fc)
        elif middle_component == "NConvDocConvNFC":
            self.middle_comp = NConvDocConvNFC(document_length, sequence_length, embedding_size, filter_sizes,
                                               num_filters,
                                               previous_component=self.input_comp, dropout=dropout,
                                               batch_normalize=batch_normalize, elu=elu, n_conv=n_conv,
                                               fc=fc)
        else:
            raise NotImplementedError

        self.is_training = self.middle_comp.is_training

        prev_layer, num_nodes = self.middle_comp.get_last_layer_info()
        # output component
        if dataset == "ML":
            output = MLOutput(self.input_comp.input_y, prev_layer, num_nodes, num_classes, l2_reg_lambda)
        elif dataset == "PAN":
            output = PANOutput(self.input_comp.input_y, prev_layer, num_nodes, num_classes, l2_reg_lambda)
            self.rate_percentage = output.rate_percentage
        else:
            raise NotImplementedError

        self.loss = output.loss
        self.scores = output.scores
        self.predictions = output.predictions
        self.accuracy = output.accuracy
