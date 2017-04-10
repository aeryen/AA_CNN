import tensorflow as tf


class SixChannel:

    def __init__( self, sequence_length, num_classes, word_vocab_size, embedding_size,
                  pref2_vocab_size, pref3_vocab_size, suff2_vocab_size, suff3_vocab_size, pos_vocab_size,
                  init_embedding=None):
        # Placeholders for input, output and dropout, First None is batch size.
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.input_pref2 = tf.placeholder(tf.int32, [None, sequence_length], name="input_pref2")
        self.input_pref3 = tf.placeholder(tf.int32, [None, sequence_length], name="input_pref3")
        self.input_suff2 = tf.placeholder(tf.int32, [None, sequence_length], name="input_suff2")
        self.input_suff3 = tf.placeholder(tf.int32, [None, sequence_length], name="input_suff3")
        self.input_pos = tf.placeholder(tf.int32, [None, sequence_length], name="input_pos")

        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Embedding layer
        with tf.device('/cpu:0'), tf.variable_scope("embedding"):
            if init_embedding is None:
                W = tf.Variable(
                    tf.random_uniform([word_vocab_size, embedding_size], -1.0, 1.0),
                    name="W")
            else:
                W = tf.Variable(init_embedding, name="W", dtype="float32")
            self.embedded = tf.nn.embedding_lookup(W, self.input_x)
            self.embedded_expanded = tf.expand_dims(self.embedded, -1)

            W_pref2 = tf.Variable(tf.random_uniform([pref2_vocab_size, embedding_size], -1.0, 1.0), name="W_pref2")
            self.embedded_chars_pref2 = tf.nn.embedding_lookup(W_pref2, self.input_pref2)
            self.embedded_chars_expanded_pref2 = tf.expand_dims(self.embedded_chars_pref2, -1)

            W_pref3 = tf.Variable(tf.random_uniform([pref3_vocab_size, embedding_size], -1.0, 1.0), name="W_pref3")
            self.embedded_chars_pref3 = tf.nn.embedding_lookup(W_pref3, self.input_pref3)
            self.embedded_chars_expanded_pref3 = tf.expand_dims(self.embedded_chars_pref3, -1)

            W_suff2 = tf.Variable(tf.random_uniform([suff2_vocab_size, embedding_size], -1.0, 1.0), name="W_suff2")
            self.embedded_chars_suff2 = tf.nn.embedding_lookup(W_suff2, self.input_suff2)
            self.embedded_chars_expanded_suff2 = tf.expand_dims(self.embedded_chars_suff2, -1)

            W_suff3 = tf.Variable(tf.random_uniform([suff3_vocab_size, embedding_size], -1.0, 1.0), name="W_suff3")
            self.embedded_chars_suff3 = tf.nn.embedding_lookup(W_suff3, self.input_suff3)
            self.embedded_chars_expanded_suff3 = tf.expand_dims(self.embedded_chars_suff3, -1)

            W_pos = tf.Variable(tf.random_uniform([pos_vocab_size, embedding_size], -1.0, 1.0), name="W_pos")
            self.embedded_chars_pos = tf.nn.embedding_lookup(W_pos, self.input_pos)
            self.embedded_chars_expanded_pos = tf.expand_dims(self.embedded_chars_pos, -1)

            self.embedded_expanded = tf.concat(values=[self.embedded_expanded,
                                                       self.embedded_chars_expanded_pref2,
                                                       self.embedded_chars_expanded_pref3,
                                                       self.embedded_chars_expanded_suff2,
                                                       self.embedded_chars_expanded_suff3,
                                                       self.embedded_chars_expanded_pos],
                                               axis=3)
