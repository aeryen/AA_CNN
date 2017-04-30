import tensorflow as tf
import logging
import tensorflow.contrib.rnn as rnn

class PureRNN(object):

    def __init__(self, sequence_length, embedding_size, previous_component,
                 num_layers, bidirectional, attn_length, attn_size, attn_vec_size):
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
        x = previous_component.embedded_expanded
        n_nodes = embedding_size

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