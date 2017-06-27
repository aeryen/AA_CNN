import tensorflow as tf


class TwoEmbChannel:
    def __init__(self, sequence_length, num_classes, word_vocab_size, embedding_size,
                 init_embedding_glv=None, init_embedding_w2v=None):
        # Placeholders for input, output and dropout, First None is batch size.
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Embedding layer
        with tf.variable_scope("embedding"):
            if init_embedding_glv is not None and init_embedding_w2v is not None:
                W_glove = tf.Variable(init_embedding_glv, name="W-glove", dtype="float32")
                self.embedded_glove = tf.nn.embedding_lookup(W_glove, self.input_x)

                W_w2v = tf.Variable(init_embedding_w2v, name="W-w2v", dtype="float32")
                self.embedded_w2v = tf.nn.embedding_lookup(W_w2v, self.input_x)

                self.embedded_glove = tf.expand_dims(self.embedded_glove, -1)
                self.embedded_w2v = tf.expand_dims(self.embedded_w2v, -1)

                self.embedded_expanded = tf.concat(values=[self.embedded_glove, self.embedded_w2v],
                                                   axis=3)
            else:
                raise ValueError("un supported.")
