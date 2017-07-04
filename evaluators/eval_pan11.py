#! /usr/bin/env python

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
import re

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
from datahelpers.DataHelper import DataHelper
from datahelpers.data_helper_pan11 import DataHelperPan11
import utils.ArchiveManager as AM
from datahelpers.Data import LoadMethod



class Evaluator:
    def __init__(self):
        self.dater = None
        self.eval_log = None
        self.test_data = None
        self.y_test_scalar = None

    def print_a_csv(self, exp_dir, file_name, method_name, prob, pred, true):
        csv_file = open(os.path.join(exp_dir, file_name + "_ " + method_name + "_out.csv"), mode="w+")
        for i in range(len(pred)):
            # csv_file.write(self.test_data.file_id[i] + "\n")
            csv_file.write("prob:," + re.sub(r'[\[\]\s]+', ',', str(prob[i])) + "\n")
            csv_file.write("pred:," + re.sub(r'[\[\]\s]+', ',', str(pred[i])) + "\n")
            csv_file.write("true:," + re.sub(r'[\[\]\s]+', ',', str(true[i])) + "\n")

    def load(self, dater):
        self.dater = dater
        print("Loading data...")
        self.test_data = self.dater.get_test_data()
        self.y_test_scalar = np.argmax(self.test_data.label_instance, axis=1)

    def evaluate(self, experiment_dir, checkpoint_step, doc_acc=False, do_is_training=True):
        if checkpoint_step is not None:
            checkpoint_file = experiment_dir + "/checkpoints/" + "model-" + str(checkpoint_step)
        else:
            checkpoint_file = tf.train.latest_checkpoint(experiment_dir + "/checkpoints/", latest_filename=None)
        file_name = os.path.basename(checkpoint_file)
        self.eval_log = open(os.path.join(experiment_dir, file_name + "_eval.log"), mode="w+")
        console = logging.StreamHandler()
        logging.getLogger('').addHandler(console)

        self.eval_log.write("Evaluating: " + __file__ + "\n")
        self.eval_log.write("Test for prob: " + self.dater.problem_name + "\n")
        self.eval_log.write(checkpoint_file + "\n")
        self.eval_log.write(AM.get_time() + "\n")
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
                predictions = graph.get_operation_by_name("output/predictions").outputs[0]

                # Generate batches for one epoch
                x_batches = DataHelper.batch_iter(self.test_data.value, 64, 1, shuffle=False)
                y_batches = DataHelper.batch_iter(self.test_data.label_instance, 64, 1, shuffle=False)

                # Collect the predictions here
                all_score = None
                pred = None
                for [x_test_batch, y_test_batch] in zip(x_batches, y_batches):
                    if do_is_training:
                        batch_scores, batch_pred_max = sess.run(
                            [scores, predictions],
                            {input_x: x_test_batch, dropout_keep_prob: 1.0,
                             is_training: 0})
                    else:
                        batch_scores, batch_pred_max = sess.run(
                            [scores, predictions],
                            {input_x: x_test_batch, dropout_keep_prob: 1.0})

                    batch_scores = tf.nn.softmax(batch_scores).eval()

                    if all_score is None:
                        all_score = batch_scores
                        pred = batch_pred_max
                    else:
                        all_score = np.concatenate([all_score, batch_scores], axis=0)
                        pred = np.concatenate([pred, batch_pred_max], axis=0)

        mi_prec = precision_score(y_true=self.y_test_scalar, y_pred=pred, average="micro")
        self.eval_log.write("micro prec:\t" + str(mi_prec) + "\n")

        mi_recall = recall_score(y_true=self.y_test_scalar, y_pred=pred, average="micro")
        self.eval_log.write("micro recall:\t" + str(mi_recall) + "\n")

        mi_f1 = f1_score(y_true=self.y_test_scalar, y_pred=pred, average="micro")
        self.eval_log.write("micro f1:\t" + str(mi_f1) + "\n")

        ma_prec = precision_score(y_true=self.y_test_scalar, y_pred=pred, average='macro')
        self.eval_log.write("macro prec:\t" + str(ma_prec) + "\n")

        ma_recall = recall_score(y_true=self.y_test_scalar, y_pred=pred, average='macro')
        self.eval_log.write("macro recall:\t" + str(ma_recall) + "\n")

        ma_f1 = f1_score(y_true=self.y_test_scalar, y_pred=pred, average='macro')
        self.eval_log.write("macro f1:\t" + str(ma_f1) + "\n")

        jaccard = jaccard_similarity_score(y_true=self.y_test_scalar, y_pred=pred)
        self.eval_log.write("jaccard:\t" + str(jaccard) + "\n")

        hamming = hamming_loss(y_true=self.y_test_scalar, y_pred=pred)
        self.eval_log.write("hamming:\t" + str(hamming) + "\n")

        acc = accuracy_score(y_true=self.y_test_scalar, y_pred=pred)
        self.eval_log.write("acc:\t" + str(acc) + "\n")

        self.eval_log.write("\n")
        self.eval_log.write("\n")

        self.print_a_csv(exp_dir=experiment_dir, file_name=file_name, method_name="NORM",
                         prob=all_score, pred=pred, true=self.y_test_scalar)

if __name__ == "__main__":
    step = None
    dater = DataHelperPan11(embed_type="glove", embed_dim=300, target_sent_len=100, prob_code=1)
    e = Evaluator()
    e.load(dater)
    path = sys.argv[1]
    if len(sys.argv) == 2:
        e.evaluate(path, step, do_is_training=True)
    elif len(sys.argv) > 2:
        steps = list(map(int, sys.argv[2].split("/")))
        for step in steps:
            e.evaluate(path, step, do_is_training=True)

