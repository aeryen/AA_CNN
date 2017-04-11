import tensorflow as tf

class NConvDocConvNFC(object):
    """
    CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling layers.
    Lacks an output layer.
    """

    def __init__(
            self, document_length, sequence_length, embedding_size, filter_size_lists, num_filters, previous_component,
            batch_normalize=False, dropout = False, elu = False, n_conv=1, fc=[], l2_reg_lambda=0.0):
        self.is_training = tf.placeholder(tf.bool, name='is_training')
        self.dropout = dropout
        self.batch_normalize = batch_normalize
        self.elu = elu
        self.n_conv = n_conv
        self.last_layer = None
        self.num_filters_total = None
        # Create a convolution + + nonlinearity + maxpool layer for each filter size
        for n in range(n_conv):
            all_filter_size_output = []
            self.num_filters_total = num_filters * len(filter_size_lists[n])
            for filter_size in filter_size_lists[n]:
                with tf.variable_scope("conv-%s-%s" % (str(n+1), filter_size)):
                    if n == 0:
                        self.last_layer = previous_component.last_layer
                        # last layer: [?, document size 128, sentence size 128, embedding size 100]
                    else:
                        if self.dropout == True:
                            self.last_layer = tf.nn.dropout(self.last_layer, 0.8, name="dropout-inter-conv")
                        embedding_size = self.last_layer.get_shape()[2].value

                    filter_shape = [1, filter_size, embedding_size, num_filters]

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
                    conv = tf.pad(conv, [[0, 0], [0, 0], [top_pad, bottom_pad], [0, 0]], mode='CONSTANT',
                                  name="conv_word_pad")
                    # conv: [?, document size 128, sentence size 128, filter num 100]

                    if batch_normalize == True:
                        conv = tf.contrib.layers.batch_norm(conv,
                                                            center=True, scale=True, fused=True,
                                                            is_training=self.is_training)

                    # Apply nonlinearity
                    b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                    if elu == False:
                        h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                    else:
                        h = tf.nn.elu(tf.nn.bias_add(conv, b), name="elu")

                    all_filter_size_output.append(h)

            self.last_layer = tf.concat(3, all_filter_size_output)
            # last_layer: [?, doc size 128, sentence size 128, all filters 300]

        with tf.variable_scope("maxpool-sentence"):
            # Maxpooling over the outputs
            self.sent_pooled = tf.nn.max_pool(
                self.last_layer,
                ksize=[1, 1, sequence_length, 1],
                strides=[1, 1, 1, 1],
                padding='VALID',
                name="pool")
            # pooled: [?, doc size 128, sent size 1, all filters 300]

        if self.dropout == True:
            with tf.variable_scope("dropout-keep"):
                self.sent_drop = tf.nn.dropout(self.sent_pooled, previous_component.dropout_keep_prob)

        document_filter_size = [2, 3, 4]
        all_filter_size_output = []
        self.num_filters_total = num_filters * len(document_filter_size)
        for filter_size in document_filter_size:
            with tf.variable_scope("doc-conv-%s" % filter_size):
                filter_shape = [filter_size, 1, self.sent_drop.get_shape()[3].value, num_filters]
                # Convolution Layer
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                conv = tf.nn.conv2d(
                    self.sent_drop,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # conv: [?, 126, 1, 100]
                if batch_normalize == True:
                    conv = tf.contrib.layers.batch_norm(conv,
                                                        center=True, scale=True, fused=True,
                                                        is_training=self.is_training)
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                if elu == False:
                    h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                else:
                    h = tf.nn.elu(tf.nn.bias_add(conv, b), name="elu")

                with tf.variable_scope("maxpool-sentence"):
                    # Maxpooling over the outputs
                    doc_pooled = tf.nn.max_pool(
                        h,
                        ksize=[1, document_length - filter_size + 1, 1, 1],
                        strides=[1, 1, 1, 1],
                        padding='VALID',
                        name="pool")
                # doc_pooled: [?, 1, 1, 100]
                all_filter_size_output.append(doc_pooled)

        self.doc_features = tf.concat(3, all_filter_size_output)
        # last_layer: [?, 1, 1, 300]
        self.h_pool_flat = tf.reshape(self.doc_features, [-1, self.num_filters_total])
        self.last_layer = self.h_pool_flat
        self.n_nodes_last_layer = self.num_filters_total

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
