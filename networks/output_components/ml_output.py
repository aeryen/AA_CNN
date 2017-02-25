import tensorflow as tf

class MLOutput(object):
    def __init__(self, input_y, prev_layer, num_nodes_prev_layer, num_classes, l2_reg_lambda):
        l2_loss = tf.constant(0.0)
        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_nodes_prev_layer, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            # l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(prev_layer, W, b, name="scores")
            self.predictions = tf.sigmoid(self.scores, name="predictions")
            print "Prediction shape: " + str(self.predictions.get_shape())

        with tf.name_scope("loss-lbd" + str(l2_reg_lambda)):
            # losses = tf.nn.softmax_cross_entropy_with_logits(self.scores, self.input_y)  # TODO
            losses = tf.nn.sigmoid_cross_entropy_with_logits(self.scores, input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            # all correct
            correct_predictions = tf.equal(tf.greater_equal(self.predictions, 0.5), tf.equal(input_y, 1))
            correct_predictions = tf.reduce_all(correct_predictions, axis=1)
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

