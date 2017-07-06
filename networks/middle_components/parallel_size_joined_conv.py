import tensorflow as tf
import logging
import tensorflow.contrib.rnn as rnn


class NCrossSizeParallelConvNFC(object):
    """
    CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling layers.
    Lacks an output layer.
    """

    def __init__(
            self, sequence_length, embedding_size, filter_size_lists, num_filters, previous_component,
            batch_normalize=False, dropout=False, elu=False, fc=[], l2_reg_lambda=0.0):

        self.is_training = tf.placeholder(tf.bool, name='is_training')
        self.dropout = dropout
        self.batch_normalize = batch_normalize
        self.elu = elu
        self.last_layer = None
        self.num_filters_total = None
        self.l2_reg_lambda = l2_reg_lambda
        self.l2_sum = tf.constant(0.0)

        # Create a convolution + + nonlinearity + maxpool layer for each filter size
        for n in range(len(filter_size_lists)):
            if not isinstance(filter_size_lists[n], list):
                raise ValueError("filter_sizes must be list of lists, for ex.[[3,4,5]] or [[3,4,5],[3,4,5],[5]]")

            all_filter_size_output = []
            if n == 0:
                self.last_layer = previous_component.embedded_expanded
                n_input_channels = previous_component.embedded_expanded.get_shape()[3].value
                cols = embedding_size
            else:
                self.last_layer = tf.transpose(self.last_layer, perm=[0, 1, 3, 2])
                if self.dropout == True:
                    self.last_layer = tf.nn.dropout(self.last_layer, 0.8, name="dropout-inter-conv")
                cols = self.num_filters_total
                n_input_channels = 1

            for filter_size in filter_size_lists[n]:
                with tf.variable_scope("conv-%s-%s" % (str(n + 1), filter_size)):
                    filter_shape = [filter_size, cols, n_input_channels, num_filters]

                    # Convolution Layer
                    W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")

                    conv = tf.nn.depthwise_conv2d_native(
                        self.last_layer,
                        W,
                        strides=[1, 1, 1, 1],
                        padding="VALID",
                        name="conv")
                    logging.warning("DEEP WISE CNN")

                    top_pad = int((filter_size - 1) / 2.0)
                    bottom_pad = filter_size - 1 - top_pad
                    conv = tf.pad(conv, [[0, 0], [top_pad, bottom_pad], [0, 0], [0, 0]], mode='CONSTANT',
                                  name="conv_word_pad")

                    # conv ==> [batch_size, sequence_length, 1, num_filters]
                    if batch_normalize == True:
                        conv = tf.contrib.layers.batch_norm(conv,
                                                            center=True, scale=True, fused=True,
                                                            is_training=self.is_training)
                    # Add bias; Apply non-linearity
                    b = tf.Variable(tf.constant(0.1, shape=[num_filters * n_input_channels]), name="b")
                    if elu == False:
                        h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                    else:
                        h = tf.nn.elu(tf.nn.bias_add(conv, b), name="elu")

                    all_filter_size_output.append(h)

                    self.num_filters_total = num_filters * len(filter_size_lists[n]) * n_input_channels

            self.last_layer = tf.concat(all_filter_size_output, 3)  # [?, 100, 1, 800]
            # self.last_layer = tf.reshape(self.last_layer, [-1, sequence_length, self.num_filters_total, 1])

        with tf.variable_scope("maxpool-all"):
            # Maxpooling over the outputs
            pooled_all = tf.nn.max_pool(
                self.last_layer,
                ksize=[1, sequence_length, 1, 1],
                strides=[1, 1, 1, 1],
                padding='VALID',
                name="pool")

        # Combine all the pooled features
        self.h_pool_flat = tf.reshape(pooled_all, [-1, self.num_filters_total])
        self.last_layer = self.h_pool_flat
        self.n_nodes_last_layer = self.num_filters_total
        # Add dropout
        if self.dropout == True:
            with tf.variable_scope("dropout-keep"):
                self.last_layer = tf.nn.dropout(self.last_layer, previous_component.dropout_keep_prob)

        for i, n_nodes in enumerate(fc):
            self._fc_layer(i + 1, n_nodes)


    def _fc_layer(self, tag, n_nodes):
        with tf.variable_scope('fc-%s' % str(tag)):
            n_nodes = n_nodes
            W = tf.get_variable(
                "W",
                shape=[self.n_nodes_last_layer, n_nodes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[n_nodes]), name="b")
            x = tf.matmul(self.last_layer, W) + b

            if self.l2_reg_lambda > 0:
                self.l2_sum += tf.nn.l2_loss(W)

            if not self.elu:
                relu = tf.nn.relu(x, name='relu')
            else:
                relu = tf.nn.elu(x, name='elu')

            if self.batch_normalize:
                self.last_layer = tf.contrib.layers.batch_norm(relu, center=True, scale=True, fused=False,
                                                               is_training=self.is_training)
            else:
                self.last_layer = relu

            self.n_nodes_last_layer = n_nodes

    def get_last_layer_info(self):
        return self.last_layer, self.n_nodes_last_layer
