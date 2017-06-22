import tensorflow as tf


class DocCNN(object):
    def __init__(
            self, doc_length, sent_length, num_classes, word_vocab_size,
            embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0,
            init_embedding=None, init_filter_w=None, init_filter_b=None):
        # Placeholders for input, output and dropout, First None is batch size.
        self.input_x = tf.placeholder(tf.int32, [None, doc_length, sent_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.name_scope("embedding"):
            self.embedding_matrix = tf.Variable(initial_value=init_embedding, name="W", dtype="float32")
            self.embedded_words = tf.nn.embedding_lookup(self.embedding_matrix,
                                                         self.input_x)  # [batch, doc, sent, embed]

        # Create a convolution + maxpool layer for each filter size
        first_pooled_outputs = []
        word_cnn_w_list = []
        word_cnn_b_list = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-1-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                init_filter_w[i] = tf.transpose(init_filter_w[i], [2, 0, 1, 3])
                word_cnn_w = tf.Variable(initial_value=init_filter_w[i], name="W")  # to be init
                word_cnn_w_list.append(word_cnn_w)
                word_cnn_b = tf.Variable(initial_value=init_filter_b[i], name="b")  # to be init
                word_cnn_b_list.append(word_cnn_b)
                conv = tf.nn.conv2d(
                    self.embedded_words,
                    word_cnn_w,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, word_cnn_b), name="relu")
                # [batch, document 400, sent 48, out_filter 100]

                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, 1, sent_length - filter_size + 1, 1],  # batch,
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                # [batch, doc 400, sent 1, filters 100]
                first_pooled_outputs.append(pooled)

        num_filters_total = num_filters * len(filter_sizes)
        first_pooled_outputs = tf.concat(values=first_pooled_outputs, axis=3)  # [64, 400, 1, 300]
        self.h_pool_flat = tf.reshape(first_pooled_outputs, [-1, doc_length, 1, num_filters_total])



        # Add dropout
        with tf.name_scope("dropout-keep"):
            self.h_drop = tf.nn.dropout(self.doc_vector, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            # W = tf.Variable(tf.truncated_normal([num_filters_total, num_classes], stddev=0.1), name="W")
            word_cnn_w = tf.get_variable(
                "W",
                shape=[2 * 300, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(word_cnn_w)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, word_cnn_w, b, name="scores")
            self.predictions_sigmoid = tf.sigmoid(self.scores, name="predictions_sigmoid")
            self.predictions_max = tf.argmax(self.scores, 1, name="predictions_max")  # 3333333333333333333333333
            print("Prediction shape: " + str(self.predictions_sigmoid.get_shape()))

        # CalculateMean cross-entropy loss
        with tf.name_scope("loss-lbd" + str(l2_reg_lambda)):
            # input_y_prob = self.input_y / tf.reduce_sum(self.input_y, axis=1, keep_dims=True)
            # losses = tf.nn.softmax_cross_entropy_with_logits(labels=input_y_prob, logits=self.scores)  # TODO
            losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.input_y, logits=self.scores)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            # all correct
            correct_predictions = tf.equal(tf.greater_equal(self.predictions_sigmoid, 0.5), tf.equal(self.input_y, 1))
            correct_predictions = tf.reduce_all(correct_predictions, axis=1)
            self.accuracy_sigmoid = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy_sigmoid")

            correct_max = tf.equal(tf.one_hot(indices=self.predictions_max, depth=num_classes),
                                   self.input_y)
            correct_max = tf.cast(tf.reduce_all(correct_max, axis=1), "float")
            self.accuracy_max = tf.reduce_mean(correct_max, name="accuracy_max")
