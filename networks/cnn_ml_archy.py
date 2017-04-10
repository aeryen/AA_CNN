import tensorflow as tf
import numpy as np


class TextCNN(object):
    """
    Hierarchical attempt
    ideally have a sentence then document hierarchy, but i forgot how well this goes
    """

    def __init__(
            self, doc_sent_len, sent_len, num_classes, vocab_size,
            embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0,
            init_embedding=None):
        # Placeholders for input, output and dropout, First None is batch size.
        self.input_x = tf.placeholder(tf.int32, [None, doc_sent_len, sent_len], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            if init_embedding is None:
                W = tf.Variable(
                    tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                    name="embed_W")
            else:
                W = tf.Variable(init_embedding, name="embed_W", dtype="float32")
            self.embedded_chars = tf.nn.embedding_lookup(W, self.input_x)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        # Create a convolution + maxpool layer for each filter size
        num_filters_per_sent = num_filters * len(filter_sizes)  # = 300~
        doc_pool_flat_list = []

        W_list = []
        b_list = []
        for i, filter_size in enumerate(filter_sizes):
            filter_shape = [filter_size, embedding_size, 1, num_filters]
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W"+str(filter_size))
            W_list.append(W)
            b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b"+str(filter_size))
            b_list.append(b)

        for sent_i in range(doc_sent_len):
            pooled_outputs = []
            for i, filter_size in enumerate(filter_sizes):
                with tf.name_scope("conv-maxpool-%s" % filter_size):
                    # Convolution Layer
                    conv = tf.nn.conv2d(
                        self.embedded_chars_expanded[:, sent_i, :, :, :],
                        W_list[i],
                        strides=[1, 1, 1, 1],
                        padding="VALID",
                        name="conv")
                    # conv ==> [batch, sequence_length - filter_size + 1, 1, 100]
                    # Apply nonlinearity
                    h = tf.nn.relu(tf.nn.bias_add(conv, b_list[i]), name="relu")
                    # Maxpooling over the outputs
                    pooled = tf.nn.max_pool(  # (batch, 1, 1, 100)
                        h,
                        ksize=[1, sent_len - filter_size + 1, 1, 1],
                        strides=[1, 1, 1, 1],
                        padding='VALID',
                        name="pool")
                    pooled_outputs.append(pooled)

            # Combine all the pooled features
            sent_h_pool = tf.concat(pooled_outputs, 3)
            sent_h_pool_flat = tf.reshape(sent_h_pool, [-1, 1, num_filters_per_sent])
            doc_pool_flat_list.append(sent_h_pool_flat)

        # doc_pool_flat_list is 100 (doc_sent_len) of 64 * 1 * 300
        self.doc_pool_flat = tf.concat(doc_pool_flat_list, 1)  # batch * doc_sent * sent_feat
        self.doc_pool_flat = tf.expand_dims(self.doc_pool_flat, -1) # 64 * 300 * 300 * 1
        doc_pool_output_list = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool_doc-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, num_filters_per_sent, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="doc_conv_W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="doc_conv_b")
                conv = tf.nn.conv2d(
                    self.doc_pool_flat,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # conv ==> [1, doc_sent_len, 1, 1]
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="doc_conv_relu")  # (20. 298. 1. 100)
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, doc_sent_len - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="doc_conv_pool")
                doc_pool_output_list.append(pooled)

        # Combine all the pooled features
        self.doc_h_pool = tf.concat(doc_pool_output_list, 3)  # (20, 1, 1, 300)
        self.doc_h_pool_flat = tf.reshape(self.doc_h_pool, [-1, num_filters_per_sent])

        # Add dropout
        with tf.name_scope("dropout-keep"):
            self.h_drop = tf.nn.dropout(self.doc_h_pool_flat, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            # W = tf.Variable(tf.truncated_normal([num_filters_per_sent, num_classes], stddev=0.1), name="W")
            W = tf.get_variable(
                "W",
                shape=[num_filters_per_sent, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            # l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.predictions = tf.sigmoid(self.scores, name="predictions")

        self.rate_percentage = [0.0] * num_classes
        with tf.name_scope("prediction-ratio"):
            for i in range(num_classes):
                rate1_logistic = tf.equal(self.predictions, i)
                self.rate_percentage[i] = tf.reduce_mean(tf.cast(rate1_logistic, "float"),
                                                         name="rate-" + str(i) + "/percentage")

        # CalculateMean cross-entropy loss
        with tf.name_scope("loss-lbd" + str(l2_reg_lambda)):
            losses = tf.nn.sigmoid_cross_entropy_with_logits(self.scores, self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            self.multi_pred = tf.greater_equal(self.scores, 0.0)
            compare_result = tf.equal(self.multi_pred, tf.equal(self.input_y, 1))
            allright = tf.equal( tf.reduce_sum(tf.cast(compare_result, "float"), reduction_indices=1), 20 )
            # correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(allright, "float"), name="allright_accuracy")
