import tensorflow as tf

class OneCMiddle(object):
    """
    CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling layers.
    Lacks an output layer.
    """

    def __init__(
            self, sequence_length, embedding_size, filter_sizes, num_filters, previous_component, batch_normalize=False,
            dropout = False):
        self.is_training = tf.placeholder(tf.bool, name='is_training')
        self.dropout = dropout
        # Create a convolution + + nonlinearity + maxpool layer for each filter size
        pooled_outputs = []
        for filter_size in filter_sizes:
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                # Convolution Layer
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                conv = tf.nn.conv2d(
                    previous_component.embedded_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
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
                    h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        self.num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(3, pooled_outputs)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, self.num_filters_total])

        # Add dropout
        if self.dropout == True:
            with tf.name_scope("dropout-keep"):
                self.h_drop = tf.nn.dropout(self.h_pool_flat, previous_component.dropout_keep_prob)

    def get_last_layer_info(self):
        if self.dropout == True:
            return self.h_drop, self.num_filters_total
        return self.h_pool_flat, self.num_filters_total
