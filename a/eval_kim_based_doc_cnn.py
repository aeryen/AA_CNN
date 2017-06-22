#! /usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import hamming_loss
from sklearn.metrics import accuracy_score

import sys
import logging
import os.path

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
from datahelpers.DataHelper import DataHelper
from datahelpers.data_helper_ml_normal import DataHelperMLNormal
import utils.ArchiveManager as AM
from datahelpers.Data import LoadMethod
import math


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


class Evaluator:
    def __init__(self, dater):
        self.dater = dater
        self.test_data = None
        self.vocab = None
        self.vocab_inv = None
        self.eval_log = None
        print("Loading data...")
        self.test_data, self.vocab, self.vocab_inv = self.dater.get_test_data()

    def test(self, experiment_dir, checkpoint_step, doc_acc=True, do_is_training=True):
        if checkpoint_step is not None:
            checkpoint_file = experiment_dir + "/checkpoints/" + "model-" + str(checkpoint_step)
        else:
            checkpoint_file = tf.train.latest_checkpoint(experiment_dir + "/checkpoints/", latest_filename=None)
        file_name = os.path.basename(checkpoint_file)
        self.eval_log = open(os.path.join(experiment_dir, file_name + "_eval.log"), mode="w+")

        logging.info("Evaluating: " + __file__)
        self.eval_log.write("Evaluating: " + __file__ + "\n")
        logging.info("Test for prob: " + self.dater.problem_name)
        self.eval_log.write("Test for prob: " + self.dater.problem_name + "\n")
        logging.info(checkpoint_file)
        self.eval_log.write(checkpoint_file + "\n")
        logging.info(AM.get_time())
        self.eval_log.write(AM.get_time() + "\n")
        logging.info("Total number of test examples: {}".format(len(self.test_data.label_instance)))
        self.eval_log.write("Total number of test examples: {}\n".format(len(self.test_data.label_instance)))

        graph = tf.Graph()
        with graph.as_default():
            session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
            sess = tf.Session(config=session_conf)
            with sess.as_default():
                # Load the saved meta graph and restore variables
                saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
                saver.restore(sess, checkpoint_file)

                # Get the placeholders from the graph by name
                input_x = graph.get_operation_by_name("input_x").outputs[0]
                input_y = graph.get_operation_by_name("input_y").outputs[0]
                dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
                if do_is_training:
                    is_training = graph.get_operation_by_name("is_training").outputs[0]
                else:
                    is_training = None

                # Tensors we want to evaluate
                scores = graph.get_operation_by_name("output/scores").outputs[0]
                predictions_sigmoid = graph.get_operation_by_name("output/predictions_sigmoid").outputs[0]
                predictions_max = graph.get_operation_by_name("output/predictions_max").outputs[0]

                # Generate batches for one epoch
                x_batches = DataHelper.batch_iter(self.test_data.value, 10, 1, shuffle=False)
                y_batches = DataHelper.batch_iter(self.test_data.label_instance, 10, 1, shuffle=False)

                # Collect the predictions here
                all_score = None
                pred_max = None
                pred_sigmoid = None
                for [x_test_batch, y_test_batch] in zip(x_batches, y_batches):
                    if do_is_training:
                        batch_scores, batch_pred_sigmoid, batch_pred_max = sess.run(
                            [scores, predictions_sigmoid, predictions_max],
                            {input_x: x_test_batch, dropout_keep_prob: 1.0,
                             is_training: 0})
                    else:
                        batch_scores, batch_pred_sigmoid, batch_pred_max = sess.run(
                            [scores, predictions_sigmoid, predictions_max],
                            {input_x: x_test_batch, dropout_keep_prob: 1.0})

                    batch_pred_max = tf.one_hot(indices=batch_pred_max,
                                                depth=self.dater.num_of_classes).eval() == 1  # TODO temp

                    if all_score is None:
                        all_score = batch_scores
                        pred_max = batch_pred_max
                        pred_sigmoid = batch_pred_sigmoid
                    else:
                        all_score = np.concatenate([all_score, batch_scores], axis=0)
                        pred_max = np.concatenate([pred_max, batch_pred_max], axis=0)
                        pred_sigmoid = np.concatenate([pred_sigmoid, batch_pred_sigmoid], axis=0)

            self.sent_accuracy(pred_sigmoid)

            self.eval_log.write("\n")
            self.eval_log.write("\n")

    def sent_accuracy(self, all_predictions_sigmoid):
        np.set_printoptions(precision=2, linewidth=160, suppress=True)

        all_predictions_bool = all_predictions_sigmoid > 0.5
        pred_num = all_predictions_bool.astype(int)

        test_label_bool = np.array(self.test_data.label_instance) == 1
        sentence_result_label_matrix = all_predictions_bool == test_label_bool
        sentence_result = np.logical_and.reduce(sentence_result_label_matrix, axis=1)
        correct_predictions = np.sum(a=sentence_result.astype(float))
        average_accuracy = correct_predictions / float(sentence_result.shape[0])

        for i in range(len(self.test_data.doc_size)):
            logging.info(str(i) + " : " + str(all_predictions_sigmoid[i]) + "\n" +
                         "pred: " + str(pred_num[i]) + "\n" +
                         "true: " + str(self.test_data.label_doc[i]))

            self.eval_log.write(str(i) + " : " + str(all_predictions_sigmoid[i]) + "\n" +
                                "pred: " + str(pred_num[i]) + "\n" +
                                "true: " + str(self.test_data.label_doc[i]) + "\n")

        logging.info("Sent ACC\t" + str(average_accuracy) + "\t\t(cor: " + str(correct_predictions) + ")")
        self.eval_log.write("Sent ACC\t" + str(average_accuracy) + "\t\t(cor: " + str(correct_predictions) + ")\n")


if __name__ == "__main__":
    step = None
    dater = DataHelperMLNormal(doc_level=LoadMethod.DOC, embed_type="glove",
                               embed_dim=300, target_sent_len=50, target_doc_len=400, train_csv_file="labels.csv")

    e = Evaluator(dater)
    path = sys.argv[1]
    if len(sys.argv) == 2:
        e.test(path, step, doc_acc=True, do_is_training=False)
    elif len(sys.argv) > 2:
        steps = list(map(int, sys.argv[2].split("/")))
        for step in steps:
            e.test(path, step, doc_acc=True, do_is_training=False)
