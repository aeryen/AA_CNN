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
            self, sequence_length, embedding_size, filter_size_lists, num_filters, previous_component, batch_normalize=False,
            dropout = False, elu = False, n_conv=1, fc=[], l2_reg_lambda=0.0):

        self.is_training = tf.placeholder(tf.bool, name='is_training')
        self.dropout = dropout
        self.batch_normalize = batch_normalize
        self.elu = elu
        self.n_conv = n_conv
        self.last_layer = None
        self.num_filters_total = None
        self.l2_reg_lambda = l2_reg_lambda
        self.l2_sum = tf.constant(0.0)

        # Create a convolution + + nonlinearity + maxpool layer for each filter size
        for n in range(n_conv):
            if not isinstance(filter_size_lists[n], list):
                raise ValueError("filter_sizes must be list of lists, for ex.[[3,4,5]] or [[3,4,5],[3,4,5],[5]]")
            all_filter_size_output = []
            self.num_filters_total = num_filters * len(filter_size_lists[n])
            for filter_size in filter_size_lists[n]:
                with tf.variable_scope("conv-%s-%s" % (str(n+1), filter_size)):
                    if n == 0:
                        self.last_layer = previous_component.embedded_expanded
                        n_input_channels = previous_component.embedded_expanded.get_shape()[3].value
                        cols = embedding_size
                    else:
                        if self.dropout == True:
                            self.last_layer = tf.nn.dropout(self.last_layer, 0.8, name="dropout-inter-conv")
                        cols = total_output
                        n_input_channels = 1

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

                    if self.l2_reg_lambda > 0:
                        self.l2_sum += tf.nn.l2_loss(W)
                    all_filter_size_output.append(h)

            self.last_layer = tf.concat(all_filter_size_output, 3)
            total_output = num_filters * len(filter_size_lists[n]) * n_input_channels
            self.last_layer = tf.reshape(self.last_layer, [-1, sequence_length, total_output, 1])

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

        self._gru(500, 1, True, self.n_nodes_last_layer, self.n_nodes_last_layer, self.n_nodes_last_layer, self.n_nodes_last_layer)

        for i, n_nodes in enumerate(fc):
            self.last_layer = self._fc_layer(i + 1, n_nodes)


    def _gru(self, n_nodes, num_layers, bidirectional,sequence_length,attn_length, attn_size, attn_vec_size):
        """
        Args:
          num_layers: The number of layers of the rnn model.
          bidirectional: boolean, Whether this is a bidirectional rnn.
          sequence_length: If sequence_length is provided, dynamic calculation is
            performed. This saves computational time when unrolling past max sequence
            length. Required for bidirectional RNNs.
          initial_state: An initial state for the RNN. This must be a tensor of
            appropriate type and shape [batch_size x cell.state_size].
          attn_length: integer, the size of attention vector attached to rnn cells.
          attn_size: integer, the size of an attention window attached to rnn cells.
          attn_vec_size: integer, the number of convolutional features calculated on
            attention state and the size of the hidden layer built from base cell
            state.

        """
        x = self.last_layer

        if bidirectional:
            # forward direction cell
            fw_cell = rnn.GRUCell(n_nodes)
            bw_cell = rnn.GRUCell(n_nodes)
            # attach attention cells if specified
            if attn_length is not None:
                fw_cell = rnn.AttentionCellWrapper(
                    fw_cell, attn_length=attn_length, attn_size=attn_size,
                    attn_vec_size=attn_vec_size, state_is_tuple=False)
                bw_cell = rnn.AttentionCellWrapper(
                    bw_cell, attn_length=attn_length, attn_size=attn_size,
                    attn_vec_size=attn_vec_size, state_is_tuple=False)
            rnn_fw_cell = rnn.MultiRNNCell([fw_cell] * num_layers,
                                                   state_is_tuple=False)
            # backward direction cell
            rnn_bw_cell = rnn.MultiRNNCell([bw_cell] * num_layers,
                                                   state_is_tuple=False)
            outputs, output_state_fw, output_state_bw = rnn.stack_bidirectional_dynamic_rnn(rnn_fw_cell,
                                            rnn_bw_cell,
                                            x,
                                            dtype=tf.dtypes.float32,
                                            sequence_length=sequence_length)
            self.last_layer = outputs

            return outputs, output_state_fw, output_state_bw
        else:
            rnn_cell = rnn.GRUCell(n_nodes)
            if attn_length is not None:
                rnn_cell = rnn.AttentionCellWrapper(
                    rnn_cell, attn_length=attn_length, attn_size=attn_size,
                    attn_vec_size=attn_vec_size, state_is_tuple=False)
            cell = rnn.MultiRNNCell([rnn_cell] * num_layers,
                                            state_is_tuple=False)
            outputs, state = rnn.static_rnn(cell,
                                 x,
                                 dtype=tf.dtypes.float32,
                                 sequence_length=sequence_length)
            self.last_layer = outputs
            return outputs, state




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
