import tensorflow as tf
from middle_components.cnn_origin import BaseTextCNN
from output_components.ml_output import MLOutput
from output_components.pan_output import PANOutput
from input_components.OneChannel import OneChannel


class TextCNN:
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    Works for both single label (PAN) and multilabel (ML) datasets
    """

    def __init__(
            self, sequence_length, num_classes, word_vocab_size,
            embedding_size, filter_sizes, num_filters, dataset="ML", l2_reg_lambda=0.0,
            init_embedding=None):

        # input layer
        self.input_comp = OneChannel(sequence_length, num_classes, word_vocab_size, embedding_size, init_embedding)
        self.input_x = self.input_comp.input_x
        self.input_y = self.input_comp.input_y
        self.dropout_keep_prob = self.input_comp.dropout_keep_prob

        # middle layer
        self.middle_comp = BaseTextCNN(sequence_length, embedding_size, filter_sizes, num_filters,
                                       previous_component=self.input_comp)
        prev_layer, num_nodes = self.middle_comp.get_last_layer_info()

        # output layer
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
