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
from datahelpers.data_helper_ml_2chan import DataHelperML2CH
from datahelpers.data_helper_pan11 import DataHelperPan11
import utils.ArchiveManager as AM
from datahelpers.Data import LoadMethod
import math
import re


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


class Evaluator:
    def __init__(self):
        self.dater = None
        self.test_data = None
        self.vocab = None
        self.vocab_inv = None
        self.eval_log = None

    def print_a_csv(self, exp_dir, file_name, method_name, prob, pred, true):
        csv_file = open(os.path.join(exp_dir, file_name + "_ " + method_name + "_out.csv"), mode="w+")
        for i in range(len(pred)):
            csv_file.write(self.test_data.file_id[i] + "\n")
            csv_file.write("prob:," + re.sub(r'[\[\]\s]+', ',', str(prob[i])) + "\n")
            csv_file.write("pred:," + re.sub(r'[\[\]\s]+', ',', str(pred[i])) + "\n")
            csv_file.write("true:," + re.sub(r'[\[\]\s]+', ',', str(true[i])) + "\n")

    def load(self, dater):
        self.dater = dater
        print("Loading data...")
        self.test_data = self.dater.get_test_data()

    def evaluate(self, experiment_dir, checkpoint_step, doc_acc=True, do_is_training=True):
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
                predictions = graph.get_operation_by_name("output/predictions").outputs[0]

                # Generate batches for one epoch
                x_batches = DataHelper.batch_iter(self.test_data.value, 64, 1, shuffle=False)
                y_batches = DataHelper.batch_iter(self.test_data.label_instance, 64, 1, shuffle=False)

                # Collect the predictions here
                all_score = None
                pred_sigmoid = None
                for [x_test_batch, y_test_batch] in zip(x_batches, y_batches):
                    if do_is_training:
                        batch_scores, batch_pred_sigmoid = sess.run(
                            [scores, predictions],
                            {input_x: x_test_batch, dropout_keep_prob: 1.0,
                             is_training: 0})
                    else:
                        batch_scores, batch_pred_sigmoid = sess.run(
                            [scores, predictions],
                            {input_x: x_test_batch, dropout_keep_prob: 1.0})

                    if all_score is None:
                        all_score = batch_scores
                        pred_sigmoid = batch_pred_sigmoid
                    else:
                        all_score = np.concatenate([all_score, batch_scores], axis=0)
                        pred_sigmoid = np.concatenate([pred_sigmoid, batch_pred_sigmoid], axis=0)

            self.sent_accuracy_sigmoid(pred_sigmoid > 0.5)
            if doc_acc:
                print("========== WITH CUMU-SIGMOID ==========")
                self.doc_accuracy_sigmoid_cumulation(pred_sigmoid)
                # print("========== WITH CUMU-SCORE ==========")
                # self.doc_accuracy_score_cumulation(all_score)
                print("========== WITH SIGMOID ==========")
                self.doc_accuracy(pred_sigmoid > 0.5, exp_dir=experiment_dir, file_name=file_name)

            self.eval_log.write("\n")
            self.eval_log.write("\n")

    def sent_accuracy_sigmoid(self, all_predictions_bool):
        # Print prediction into file
        # np.savetxt('temp.out', all_predictions_bool.astype(int), fmt='%1.0f')

        test_label_bool = np.array(self.test_data.label_instance) == 1
        sentence_result_label_matrix = all_predictions_bool == test_label_bool
        sentence_result = np.logical_and.reduce(sentence_result_label_matrix, axis=1)
        correct_predictions = np.sum(a=sentence_result.astype(float))
        average_accuracy = correct_predictions / float(sentence_result.shape[0])

        logging.info("Sent ACC\t" + str(average_accuracy) + "\t\t(cor: " + str(correct_predictions) + ")")
        self.eval_log.write("Sent ACC\t" + str(average_accuracy) + "\t\t(cor: " + str(correct_predictions) + ")\n")

    def doc_accuracy_sigmoid_cumulation(self, all_sigmoids):
        np.set_printoptions(precision=2, linewidth=160)

        logging.info("EVALUATING USING doc_accuracy_sigmoid_cumulation")
        self.eval_log.write("EVALUATING USING doc_accuracy_sigmoid_cumulation\n")
        doc_prediction = []
        sum_to = 0

        for i in range(len(self.test_data.doc_size)):
            f_size = self.test_data.doc_size[i]
            p = all_sigmoids[sum_to:sum_to + f_size]
            sum_to = sum_to + f_size  # increment to next file
            p = np.sum(p, axis=0).astype(float)
            p = p / f_size
            pred_class = p >= 0.5
            pred_class = pred_class.astype(int)
            if 1 not in pred_class:
                print(p)
                pred_class = np.zeros([self.dater.num_of_classes], dtype=np.int)
                pred_class[np.argmax(p)] = 1
            doc_prediction.append(pred_class)
            print("pred: " + str(pred_class) + "\n" + "true: " + str(self.test_data.label_doc[i]))
            self.eval_log.write("File:" + self.test_data.file_id[i] + "\n")
            self.eval_log.write("pred: " + str(pred_class) + "\n" +
                                "true: " + str(self.test_data.label_doc[i]) + "\n")

        logging.info("")
        self.eval_log.write("\n")

        logging.info("Document ACC")
        self.eval_log.write("Document ACC\n")
        total_doc = len(self.test_data.file_id)
        correct = 0.0
        for i in range(len(doc_prediction)):
            if np.array_equal(doc_prediction[i], self.test_data.label_doc[i]):
                correct += 1
        doc_acc = correct / total_doc
        print("Doc ACC: " + str(doc_acc))
        self.eval_log.write("Doc ACC: " + str(doc_acc) + "\n\n")

        y_true = np.array(self.test_data.label_doc).astype(bool)
        y_pred = np.array(doc_prediction).astype(bool)

        mi_prec = precision_score(y_true=y_true, y_pred=y_pred, average="micro")
        logging.info("micro prec:\t" + str(mi_prec))
        self.eval_log.write("micro prec:\t" + str(mi_prec) + "\n")

        mi_recall = recall_score(y_true=y_true, y_pred=y_pred, average="micro")
        logging.info("micro recall:\t" + str(mi_recall))
        self.eval_log.write("micro recall:\t" + str(mi_recall) + "\n")

        mi_f1 = f1_score(y_true=y_true, y_pred=y_pred, average="micro")
        logging.info("micro f1:\t" + str(mi_f1))
        self.eval_log.write("micro f1:\t" + str(mi_f1) + "\n")

        ma_prec = precision_score(y_true=y_true, y_pred=y_pred, average='macro')
        logging.info("macro prec:\t" + str(ma_prec))
        self.eval_log.write("macro prec:\t" + str(ma_prec) + "\n")

        ma_recall = recall_score(y_true=y_true, y_pred=y_pred, average='macro')
        logging.info("macro recall:\t" + str(ma_recall))
        self.eval_log.write("macro recall:\t" + str(ma_recall) + "\n")

        ma_f1 = f1_score(y_true=y_true, y_pred=y_pred, average='macro')
        logging.info("macro f1:\t" + str(ma_f1))
        self.eval_log.write("macro f1:\t" + str(ma_f1) + "\n")

        jaccard = jaccard_similarity_score(y_true=y_true, y_pred=y_pred)
        logging.info("jaccard:\t" + str(jaccard))
        self.eval_log.write("jaccard:\t" + str(jaccard) + "\n")

        hamming = hamming_loss(y_true=y_true, y_pred=y_pred)
        logging.info("hamming:\t" + str(hamming))
        self.eval_log.write("hamming:\t" + str(hamming) + "\n")

        acc = accuracy_score(y_true=y_true, y_pred=y_pred)
        logging.info("acc:\t" + str(acc))
        self.eval_log.write("acc:\t" + str(acc) + "\n")

    def doc_accuracy_score_cumulation(self, all_scores, exp_dir, file_name):
        logging.info("EVALUATING USING doc_accuracy_score_cumulation")
        self.eval_log.write("EVALUATING USING doc_accuracy_score_cumulation")
        doc_prediction = []
        sum_to = 0
        prob_list = []

        for i in range(len(self.test_data.doc_size)):
            f_size = self.test_data.doc_size[i]
            p = all_scores[sum_to:sum_to + f_size].astype(int)
            sum_to = sum_to + f_size  # increment to next file
            p = np.sum(p, axis=0).astype(float)
            p = p / f_size
            p = np.array([sigmoid(i) for i in p])
            prob_list.append(p)
            pred_class = p >= 0.5
            pred_class = pred_class.astype(float)
            if 1 not in pred_class:
                pred_class = np.zeros([self.dater.num_of_classes], dtype=np.int)
                pred_class[np.argmax(p)] = 1
            doc_prediction.append(pred_class)
            print("pred: " + str(pred_class) + "\n" + "true: " + str(self.test_data.label_doc[i]))
            self.eval_log.write("File:" + self.test_data.file_id[i] + "\n")
            self.eval_log.write("pred: " + str(pred_class) + "\n" +
                                "true: " + str(self.test_data.label_doc[i]) + "\n\n")

        logging.info("")
        self.eval_log.write("\n")

        logging.info("Document ACC")
        self.eval_log.write("Document ACC\n")
        total_doc = len(self.test_data.file_id)
        correct = 0.0
        for i in range(len(doc_prediction)):
            if np.array_equal(doc_prediction[i], self.test_data.label_doc[i]):
                correct += 1
        doc_acc = correct / total_doc
        print("Doc ACC: " + str(doc_acc))
        self.eval_log.write("Doc ACC: " + str(doc_acc) + "\n\n")

        y_true = np.array(self.test_data.label_doc).astype(bool)
        y_pred = np.array(doc_prediction).astype(bool)

        mi_prec = precision_score(y_true=y_true, y_pred=y_pred, average="micro")
        logging.info("micro prec:\t" + str(mi_prec))
        self.eval_log.write("micro prec:\t" + str(mi_prec) + "\n")

        mi_recall = recall_score(y_true=y_true, y_pred=y_pred, average="micro")
        logging.info("micro recall:\t" + str(mi_recall))
        self.eval_log.write("micro recall:\t" + str(mi_recall) + "\n")

        mi_f1 = f1_score(y_true=y_true, y_pred=y_pred, average="micro")
        logging.info("micro f1:\t" + str(mi_f1))
        self.eval_log.write("micro f1:\t" + str(mi_f1) + "\n")

        ma_prec = precision_score(y_true=y_true, y_pred=y_pred, average='macro')
        logging.info("macro prec:\t" + str(ma_prec))
        self.eval_log.write("macro prec:\t" + str(ma_prec) + "\n")

        ma_recall = recall_score(y_true=y_true, y_pred=y_pred, average='macro')
        logging.info("macro recall:\t" + str(ma_recall))
        self.eval_log.write("macro recall:\t" + str(ma_recall) + "\n")

        ma_f1 = f1_score(y_true=y_true, y_pred=y_pred, average='macro')
        logging.info("macro f1:\t" + str(ma_f1))
        self.eval_log.write("macro f1:\t" + str(ma_f1) + "\n")

        jaccard = jaccard_similarity_score(y_true=y_true, y_pred=y_pred)
        logging.info("jaccard:\t" + str(jaccard))
        self.eval_log.write("jaccard:\t" + str(jaccard) + "\n")

        hamming = hamming_loss(y_true=y_true, y_pred=y_pred)
        logging.info("hamming:\t" + str(hamming))
        self.eval_log.write("hamming:\t" + str(hamming) + "\n")

        acc = accuracy_score(y_true=y_true, y_pred=y_pred)
        logging.info("acc:\t" + str(acc))
        self.eval_log.write("acc:\t" + str(acc) + "\n")

        self.print_a_csv(exp_dir=exp_dir, file_name=file_name, method_name="SCR_CUMU",
                         prob=prob_list, pred=doc_prediction, true=self.test_data.label_doc)

    def doc_accuracy(self, all_predictions, exp_dir, file_name):
        self.eval_log.write(" ### Document Accuracy ### \n")

        np.set_printoptions(precision=2, linewidth=160)
        prob_list = []
        doc_prediction = []
        sum_to = 0
        for i in range(len(self.test_data.doc_size)):
            f_size = self.test_data.doc_size[i]
            p = all_predictions[sum_to:sum_to + f_size].astype(int)
            sum_to = sum_to + f_size  # increment to next file
            p = np.sum(p, axis=0).astype(float)
            p = p / f_size
            print("file " + str(i) + " : " + str(p))
            prob_list.append((p))
            pred_class = p >= 0.30
            pred_class = pred_class.astype(int)
            if 1 not in pred_class:
                pred_class = np.zeros([self.dater.num_of_classes], dtype=np.int)
                pred_class[np.argmax(p)] = 1
            doc_prediction.append(pred_class)
            print("pred: " + str(pred_class) + "   " + "true: " + str(self.test_data.label_doc[i]))
            self.eval_log.write("File:" + self.test_data.file_id[i] + "\n")
            self.eval_log.write("pred: " + str(pred_class) + "\n" +
                                "true: " + str(self.test_data.label_doc[i]) + "\n")

        logging.info("")
        self.eval_log.write("\n")

        logging.info("Document ACC")
        self.eval_log.write("Document ACC\n")
        total_doc = len(self.test_data.file_id)
        correct = 0.0
        for i in range(len(doc_prediction)):
            if np.array_equal(doc_prediction[i], self.test_data.label_doc[i]):
                correct += 1
        doc_acc = correct / total_doc
        print("Doc ACC: " + str(doc_acc))
        self.eval_log.write("Doc ACC: " + str(doc_acc) + "\n\n")

        y_true = np.array(self.test_data.label_doc).astype(bool)
        y_pred = np.array(doc_prediction).astype(bool)

        mi_prec = precision_score(y_true=y_true, y_pred=y_pred, average="micro")
        logging.info("micro prec:\t" + str(mi_prec))
        self.eval_log.write("micro prec:\t" + str(mi_prec) + "\n")

        mi_recall = recall_score(y_true=y_true, y_pred=y_pred, average="micro")
        logging.info("micro recall:\t" + str(mi_recall))
        self.eval_log.write("micro recall:\t" + str(mi_recall) + "\n")

        mi_f1 = f1_score(y_true=y_true, y_pred=y_pred, average="micro")
        logging.info("micro f1:\t" + str(mi_f1))
        self.eval_log.write("micro f1:\t" + str(mi_f1) + "\n")

        ma_prec = precision_score(y_true=y_true, y_pred=y_pred, average='macro')
        logging.info("macro prec:\t" + str(ma_prec))
        self.eval_log.write("macro prec:\t" + str(ma_prec) + "\n")

        ma_recall = recall_score(y_true=y_true, y_pred=y_pred, average='macro')
        logging.info("macro recall:\t" + str(ma_recall))
        self.eval_log.write("macro recall:\t" + str(ma_recall) + "\n")

        ma_f1 = f1_score(y_true=y_true, y_pred=y_pred, average='macro')
        logging.info("macro f1:\t" + str(ma_f1))
        self.eval_log.write("macro f1:\t" + str(ma_f1) + "\n")

        jaccard = jaccard_similarity_score(y_true=y_true, y_pred=y_pred)
        logging.info("jaccard:\t" + str(jaccard))
        self.eval_log.write("jaccard:\t" + str(jaccard) + "\n")

        hamming = hamming_loss(y_true=y_true, y_pred=y_pred)
        logging.info("hamming:\t" + str(hamming))
        self.eval_log.write("hamming:\t" + str(hamming) + "\n")

        acc = accuracy_score(y_true=y_true, y_pred=y_pred)
        logging.info("acc:\t" + str(acc))
        self.eval_log.write("acc:\t" + str(acc) + "\n")

        self.print_a_csv(exp_dir=exp_dir, file_name=file_name, method_name="THR",
                         prob=prob_list, pred=doc_prediction, true=self.test_data.label_doc)


if __name__ == "__main__":
    step = None
    dater = None
    mode = "ML_2CH"  # ML_One / ML_2CH / PAN11
    if mode == "ML_One":
        dater = DataHelperMLNormal(doc_level=LoadMethod.SENT, embed_type="glove",
                                   embed_dim=300, target_sent_len=50, target_doc_len=400, train_csv_file="labels.csv",
                                   total_fold=5, t_fold_index=0)
    elif mode == "ML_2CH":
        dater = DataHelperML2CH(doc_level=LoadMethod.SENT, embed_type="both",
                                embed_dim=300, target_sent_len=50, target_doc_len=None, train_csv_file="labels.csv",
                                total_fold=5, t_fold_index=4)

    e = Evaluator()
    e.load(dater)
    path = sys.argv[1]
    if len(sys.argv) == 2:
        e.evaluate(path, step, doc_acc=True, do_is_training=True)
    elif len(sys.argv) > 2:
        steps = list(map(int, sys.argv[2].split("/")))
        for step in steps:
            e.evaluate(path, step, doc_acc=True, do_is_training=True)
