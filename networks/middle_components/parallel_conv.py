import tensorflow as tf

class NParallelConvOnePoolNFC(object):
    """
    CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling layers.
    Lacks an output layer.
    """

    def __init__(
            self, sequence_length, embedding_size, filter_sizes, num_filters, previous_component, batch_normalize=False,
            dropout = False, elu = False, n_conv=1, n_fc=1):
        self.is_training = tf.placeholder(tf.bool, name='is_training')
        self.dropout = dropout
        self.batch_normalize = batch_normalize
        self.elu = elu
        self.n_conv = n_conv
        self.last_layer = None
        # Create a convolution + + nonlinearity + maxpool layer for each filter size
        pooled_outputs = []
        for filter_size in filter_sizes:
            for n in range(n_conv):
                with tf.variable_scope("conv-%s-%s" % (str(n+1), filter_size)):
                    if n == 0:
                        self.last_layer = previous_component.embedded_expanded
                        shape = previous_component.embedded_expanded.get_shape()
                        n_input_channels = shape[3].value
                    else:
                        n_input_channels = num_filters
                        embedding_size = self.last_layer.get_shape()[2].value

                    filter_shape = [filter_size, embedding_size, n_input_channels, num_filters]

                    # Convolution Layer
                    W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                    conv = tf.nn.conv2d(
                        self.last_layer,
                        W,
                        strides=[1, 1, 1, 1],
                        padding="VALID",
                        name="conv")
                    top_pad = int((filter_size - 1) / 2.0)
                    bottom_pad = filter_size - 1 - top_pad
                    conv = tf.pad(conv, [[0, 0], [top_pad, bottom_pad], [0, 0], [0, 0]], mode='CONSTANT',
                                  name="conv_word_pad")
                    # conv ==> [1, sequence_length - filter_size + 1, 1, 1]
                    if batch_normalize == True:
                        bn = tf.contrib.layers.batch_norm(conv,
                                                     center=True, scale=True, fused=True,
                                                     is_training=self.is_training)
                        # Apply nonlinearity
                        b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                        h = tf.nn.relu(tf.nn.bias_add(bn, b), name="relu")
                    else:
                        # Apply nonlinearity
                        b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                        if elu == False:
                            h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                        else:
                            h = tf.nn.elu(tf.nn.bias_add(conv, b), name="elu")

                    self.last_layer = h

            with tf.variable_scope("maxpool-%s" % filter_size):
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    self.last_layer,
                    ksize=[1, sequence_length, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        self.num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(3, pooled_outputs)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, self.num_filters_total])
        self.last_layer = self.h_pool_flat
        self.n_nodes_last_layer = self.num_filters_total
        # Add dropout
        if self.dropout == True:
            with tf.variable_scope("dropout-keep"):
                h_drop = tf.nn.dropout(self.last_layer, previous_component.dropout_keep_prob)
                self.last_layer = h_drop

        for i in range(n_fc):
            self._fc_layer(i + 1, 384)


    def _fc_layer(self, tag, n_nodes):
        with tf.variable_scope('fc-%s' % str(tag)):
            n_nodes = n_nodes
            W = tf.get_variable(
                "W",
                shape=[self.n_nodes_last_layer, n_nodes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[n_nodes]), name="b")
            x = tf.matmul(self.last_layer, W) + b

            if self.batch_normalize == True and self.dropout == False:
                bn = tf.contrib.layers.batch_norm(x, center=True, scale=True, fused=False,
                                                  is_training=self.is_training)
                self.last_layer = tf.nn.relu(bn, name='relu')
            elif self.batch_normalize == True and self.dropout == True:
                relu = tf.nn.relu(x, name='relu')
                self.last_layer = tf.contrib.layers.batch_norm(relu, center=True, scale=True, fused=False,
                                             is_training=self.is_training)
            else:
                if self.elu == False:
                    h = tf.nn.relu(x, name='relu')
                else:
                    h = tf.nn.elu(x, name='elu')
                self.last_layer = h

            self.n_nodes_last_layer = n_nodes

    def get_last_layer_info(self):
        return self.last_layer, self.n_nodes_last_layer
