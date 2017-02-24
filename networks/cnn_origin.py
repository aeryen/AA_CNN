import tensorflow as tf
from network_base.cnn_origin import BaseTextCNN
from output_layers.ml_output import MLOutput
from output_layers.pan_output import PANOutput

class TextCNN(BaseTextCNN):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    Works for both single label (PAN) and multilabel (ML) datasets
    """

    def __init__(
            self, sequence_length, num_classes, word_vocab_size,
            embedding_size, filter_sizes, num_filters, dataset="ML", l2_reg_lambda=0.0,
            init_embedding=None):

        super(TextCNN, self).__init__(sequence_length, num_classes, word_vocab_size,
            embedding_size, filter_sizes, num_filters, l2_reg_lambda,
            init_embedding)

        prev_layer, num_nodes = super(TextCNN, self).get_last_layer_info()

        if dataset == "ML":
            output = MLOutput(self.input_y, prev_layer, num_nodes, num_classes, l2_reg_lambda)
        elif dataset == "PAN":
            output = PANOutput(self.input_y, prev_layer, num_nodes, num_classes, l2_reg_lambda)
            self.rate_percentage = output.rate_percentage
        else:
            raise NotImplementedError

        self.loss = output.loss
        self.scores = output.scores
        self.predictions = output.predictions
        self.accuracy = output.accuracy
