#! /usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import sys
import logging
import os.path
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
from datahelpers import data_helper_ml_mulmol6_Read as dh
import utils.ArchiveManager as AM


# EVALUATOR FOR MULTI-MODALITY CNN

class evaler:

    def __init__(self):
        self.dater = None
        self.x_test = None
        self.y_test = None
        self.vocabulary = None
        self.vocabulary_inv = None

        self.pos_test = None
        self.wl_test = None
        self.p2_test = None
        self.p3_test = None
        self.s2_test = None
        self.s3_test = None

        self.doc_size_test = None

    def plot_confusion_matrix(self, cm, dater, title='Confusion matrix', cmap=plt.cm.Blues):
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(dater.num_of_classes)
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

    def load(self, dater):
        self.dater = dater

        print("Loading test data...")
        self.x_test, self.pos_test, self.wl_test, self.p2_test, self.p3_test, self.s2_test, self.s3_test, \
            self.y_test, self.vocabulary, self.vocabulary_inv, self.doc_size_test = \
            self.dater.load_test_data()
        print(("Vocabulary size: {:d}".format(len(self.vocabulary))))
        print(("Test set size {:d}".format(len(self.y_test))))

        return self.x_test, self.y_test, self.doc_size_test, \
            self.p2_test, self.p3_test, self.s2_test, self.s3_test, self.pos_test

    def test(self, experiment_dir, checkpoint_step, documentAcc=True):
        if checkpoint_step is not None:
            checkpoint_file = experiment_dir + "/checkpoints/" + "model-" + str(checkpoint_step)
        else:
            checkpoint_file = tf.train.latest_checkpoint(experiment_dir + "/checkpoints/", latest_filename=None)
        eval_log = open(os.path.join(experiment_dir, "eval.log"), mode="w+")

        logging.info("Evaluating: " + __file__)
        eval_log.write("Evaluating: " + __file__ + "\n")
        logging.info("Test for prob: " + self.dater.problem_name)
        eval_log.write("Test for prob: " + self.dater.problem_name + "\n")
        logging.info(checkpoint_file)
        eval_log.write(checkpoint_file + "\n")
        logging.info(AM.get_time())
        eval_log.write(AM.get_time() + "\n")
        logging.info("Total number of test examples: {}".format(len(self.y_test)))
        eval_log.write("Total number of test examples: {}\n".format(len(self.y_test)) + "n")

        graph = tf.Graph()
        with graph.as_default():
            session_conf = tf.ConfigProto(
                allow_soft_placement=True,
                log_device_placement=False)
            sess = tf.Session(config=session_conf)
            with sess.as_default():
                # Load the saved meta graph and restore variables
                saver = tf.train.import_meta_graph(checkpoint_file + ".meta")
                saver.restore(sess, checkpoint_file)

                # Get the placeholders from the graph by name
                input_x = graph.get_operation_by_name("input_x").outputs[0]
                input_pref2 = graph.get_operation_by_name("input_pref2").outputs[0]
                input_pref3 = graph.get_operation_by_name("input_pref3").outputs[0]
                input_suff2 = graph.get_operation_by_name("input_suff2").outputs[0]
                input_suff3 = graph.get_operation_by_name("input_suff3").outputs[0]
                input_pos = graph.get_operation_by_name("input_pos").outputs[0]
                is_training = graph.get_operation_by_name("is_training").outputs[0]
                # input_y = graph.get_operation_by_name("input_y").outputs[0]

                dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

                # Tensors we want to evaluate
                scores = graph.get_operation_by_name("output/scores").outputs[0]
                predictions = graph.get_operation_by_name("output/predictions").outputs[0]

                # Generate batches for one epoch
                x_batches = dh.DataHelperMulMol6.batch_iter(self.x_test, 64, 1, shuffle=False)
                y_batches = dh.DataHelperMulMol6.batch_iter(self.y_test, 64, 1, shuffle=False)
                pref2_batches = dh.DataHelperMulMol6.batch_iter(self.p2_test, 64, 1, shuffle=False)
                pref3_batches = dh.DataHelperMulMol6.batch_iter(self.p3_test, 64, 1, shuffle=False)
                suff2_batches = dh.DataHelperMulMol6.batch_iter(self.s2_test, 64, 1, shuffle=False)
                suff3_batches = dh.DataHelperMulMol6.batch_iter(self.s3_test, 64, 1, shuffle=False)
                pos_batches = dh.DataHelperMulMol6.batch_iter(self.pos_test, 64, 1, shuffle=False)


                # Collect the predictions here
                all_score = None
                all_predictions = np.zeros([0, 20])
                for [x_test_batch, y_test_batch,
                     pref2_batch, pref3_batch, suff2_batch, suff3_batch,
                     pos_batch] in zip(x_batches, y_batches,
                                       pref2_batches, pref3_batches, suff2_batches, suff3_batches,
                                       pos_batches):
                    batch_scores, batch_predictions = sess.run([scores, predictions],
                                                               {input_x: x_test_batch,
                                                                input_pref2: pref2_batch, input_pref3: pref3_batch,
                                                                input_suff2: suff2_batch, input_suff3: suff3_batch,
                                                                input_pos: pos_batch, dropout_keep_prob: 1.0,
                                                                is_training: 0})
                    # print batch_predictions
                    if all_score is None:
                        all_score = batch_scores
                    else:
                        all_score = np.concatenate([all_score, batch_scores], axis=0)
                    all_predictions = np.concatenate([all_predictions, batch_predictions], axis=0)

        # Print accuracy
        np.savetxt('temp.out', all_predictions, fmt='%1.0f')
        all_predictions = all_predictions >= 0.5
        self.y_test = np.array(self.y_test)
        sentence_result_label_matrix = all_predictions == (self.y_test == 1)
        sentence_result = np.logical_and.reduce(sentence_result_label_matrix, axis=1)
        correct_predictions = float(np.sum(sentence_result))
        average_accuracy = correct_predictions / float(all_predictions.shape[0])

        logging.info("Sent ACC\t" + str(average_accuracy) + "\t\t(cor: " + str(correct_predictions) + ")")
        eval_log.write("Sent ACC\t" + str(average_accuracy) + "\t\t(cor: " + str(correct_predictions) + ")")

        if documentAcc == True:
            doc_prediction = []
            sum_to = 0
            for i in range(len(self.doc_size_test)):
                f_size = self.doc_size_test[i]
                p = all_predictions[sum_to:sum_to + f_size - 1].astype(int)
                sum_to = sum_to + f_size  # increment to next file
                p = np.sum(p, axis=0).astype(float)
                p = p / f_size
                pred_class = p > 0.5
                pred_class = pred_class.astype(int)
                if 1 not in pred_class:
                    pred_class = np.zeros([20], dtype=np.int)
                    pred_class[np.argmax(p)] = 1
                doc_prediction.append(pred_class)
                logging.info("pred: " + str(pred_class) + "   " + "true: " + str(self.dater.doc_labels_test[i]))
                eval_log.write("File:" + self.dater.file_id_test[i] + "\n")
                eval_log.write("pred: " + str(pred_class) + "   " +
                                  "true: " + str(self.dater.doc_labels_test[i]) + "\n")

            logging.info("")
            eval_log.write("\n")

            logging.info("Document ACC")
            eval_log.write("Document ACC\n")
            total_doc = len(self.dater.file_id_test)
            correct = 0.0
            for i in range(len(doc_prediction)):
                if np.array_equal(doc_prediction[i], self.dater.doc_labels_test[i]):
                    correct += 1
            doc_acc = correct / total_doc
            logging.info("Doc ACC: " + str(doc_acc))
            eval_log.write("Doc ACC: " + str(doc_acc) + "\n")

        eval_log.write("\n")
        eval_log.write("\n")

if __name__ == "__main__":
    step1 = [250, 500, 750, 1000]
    step2 = [2000, 2250, 2500, 2750, 3000, 3250, 3500]
    step = None

    d = dh.DataHelperMulMol6(doc_level="sent", train_holdout=0.80, embed_dim=300,  target_sent_len=50, target_doc_len=400)
    d.load_data()
    e = evaler()
    e.load(d)
    path = sys.argv[1]
    if len(sys.argv) > 2:
        step = int(sys.argv[2])
    e.test(path, step)

