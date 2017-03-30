import tensorflow as tf
import logging

class MLOutput(object):
    def __init__(self, input_y, prev_layer, num_nodes_prev_layer, num_classes, l2_reg_lambda):
        if prev_layer.l2_sum is not None:
            self.l2_sum = prev_layer.l2_sum
            logging.warning("OPTIMIZING PROPER L2")
        else:
            self.l2_sum = tf.constant(0.0)
        # Final (unnormalized) scores and predictions
        with tf.variable_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_nodes_prev_layer, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            if l2_reg_lambda > 0:
                self.l2_sum += tf.nn.l2_loss(W)
            # l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(prev_layer, W, b, name="scores")
            self.predictions = tf.sigmoid(self.scores, name="predictions")
            #print "Prediction shape: " + str(self.predictions.get_shape())

        with tf.variable_scope("loss-lbd" + str(l2_reg_lambda)):
            # losses = tf.nn.softmax_cross_entropy_with_logits(self.scores, self.input_y)  # TODO
            losses = tf.nn.sigmoid_cross_entropy_with_logits(self.scores, input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * self.l2_sum

        # Accuracy
        with tf.variable_scope("accuracy"):
            # all correct
            correct_predictions = tf.equal(tf.greater_equal(self.predictions, 0.5), tf.equal(input_y, 1))
            correct_predictions = tf.reduce_all(correct_predictions, axis=1)
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

