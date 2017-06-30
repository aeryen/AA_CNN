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
from datahelpers import data_helper_ml_normal as data_helpers
from datahelpers.data_helper_ml_2chan import DataHelperML2CH
from datahelpers.data_helper_pan11 import DataHelperPan11
import utils.ArchiveManager as AM

# THIS CLASS IS THE EVALUATOR FOR NORMAL CNN

class evaler:
    dater = None
    x_test = None
    y_test = None
    y_test_scalar = None
    vocabulary = None
    vocabulary_inv = None
    doc_size_test = None

    def plot_confusion_matrix(self, cm, dater, title='Confusion matrix', cmap=plt.cm.Blues):
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(dater.num_of_classes)
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

    def load(self, dater):
        # self.dater = dh.DataHelper()
        # self.dater.set_problem(problem, embedding_dim=embed_dim)
        self.dater = dater

        print("Loading data...")
        self.x_test, self.y_test, self.vocab, self.vocab_inv, self.doc_size_test = self.dater.get_test_data()
        self.y_test_scalar = np.argmax(self.y_test, axis=1)
        print(("Vocabulary size: {:d}".format(len(self.vocab))))
        print(("Test set size {:d}".format(len(self.y_test))))

        return self.x_test, self.y_test, self.y_test_scalar

    def test(self, experiment_dir, checkpoint_step, documentAcc=True, do_is_training=True):
        if checkpoint_step is not None:
            checkpoint_file = experiment_dir + "/checkpoints/" + "model-" + str(checkpoint_step)
        else:
            checkpoint_file = tf.train.latest_checkpoint(experiment_dir + "/checkpoints/", latest_filename=None)
        file_name = os.path.basename(checkpoint_file)
        eval_log = open(os.path.join(experiment_dir, file_name + "_eval.log"), mode="w+")

        logging.info("Evaluating: " + __file__)
        eval_log.write("Evaluating: " + __file__ + "\n")
        logging.info("Test for prob: " + self.dater.problem_name)
        eval_log.write("Test for prob: " + self.dater.problem_name + "\n")
        logging.info(checkpoint_file)
        eval_log.write(checkpoint_file + "\n")
        logging.info(AM.get_time())
        eval_log.write(AM.get_time() + "\n")
        logging.info("Total number of test examples: {}".format(len(self.y_test)))
        eval_log.write("Total number of test examples: {}\n".format(len(self.y_test)))


        graph = tf.Graph()
        with graph.as_default():
            session_conf = tf.ConfigProto(
                allow_soft_placement=True,
                log_device_placement=False)
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

                # Tensors we want to evaluate
                scores = graph.get_operation_by_name("output/scores").outputs[0]
                predictions = graph.get_operation_by_name("output/predictions").outputs[0]

                # Generate batches for one epoch
                x_batches = data_helpers.DataHelperML.batch_iter(self.x_test, 64, 1, shuffle=False)
                y_batches = data_helpers.DataHelperML.batch_iter(self.y_test, 64, 1, shuffle=False)

                # Collect the predictions here
                all_predictions = []
                for [x_test_batch, y_test_batch] in zip(x_batches, y_batches):
                    if do_is_training:
                        batch_scores, batch_predictions = sess.run([scores, predictions],
                                                                   {input_x: x_test_batch, dropout_keep_prob: 1.0,
                                                                    is_training: 0})
                    else:
                        batch_scores, batch_predictions = sess.run([scores, predictions],
                                                                   {input_x: x_test_batch, dropout_keep_prob: 1.0})
                    # print batch_predictions
                    all_predictions.append(batch_predictions)

        all_predictions = np.concatenate(all_predictions, axis=0)

        mi_prec = precision_score(y_true=self.y_test_scalar, y_pred=all_predictions, average="micro")
        logging.info("micro prec:\t" + str(mi_prec))
        eval_log.write("micro prec:\t" + str(mi_prec) + "\n")

        mi_recall = recall_score(y_true=self.y_test_scalar, y_pred=all_predictions, average="micro")
        logging.info("micro recall:\t" + str(mi_recall))
        eval_log.write("micro recall:\t" + str(mi_recall) + "\n")

        mi_f1 = f1_score(y_true=self.y_test_scalar, y_pred=all_predictions, average="micro")
        logging.info("micro f1:\t" + str(mi_f1))
        eval_log.write("micro f1:\t" + str(mi_f1)+ "\n")

        ma_prec = precision_score(y_true=self.y_test_scalar, y_pred=all_predictions, average='macro')
        logging.info("macro prec:\t" + str(ma_prec))
        eval_log.write("macro prec:\t" + str(ma_prec) + "\n")

        ma_recall = recall_score(y_true=self.y_test_scalar, y_pred=all_predictions, average='macro')
        logging.info("macro recall:\t" + str(ma_recall))
        eval_log.write("macro recall:\t" + str(ma_recall) + "\n")

        ma_f1 = f1_score(y_true=self.y_test_scalar, y_pred=all_predictions, average='macro')
        logging.info("macro f1:\t" + str(ma_f1))
        eval_log.write("macro f1:\t" + str(ma_f1) + "\n")

        jaccard = jaccard_similarity_score(y_true=self.y_test_scalar, y_pred=all_predictions)
        logging.info("jaccard:\t" + str(jaccard))
        eval_log.write("jaccard:\t" + str(jaccard) + "\n")

        hamming = hamming_loss(y_true=self.y_test_scalar, y_pred=all_predictions)
        logging.info("hamming:\t" + str(hamming))
        eval_log.write("hamming:\t" + str(hamming) + "\n")

        acc = accuracy_score(y_true=self.y_test_scalar, y_pred=all_predictions)
        logging.info("acc:\t" + str(acc))
        eval_log.write("acc:\t" + str(acc) + "\n")

        eval_log.write("\n")
        eval_log.write("\n")

if __name__ == "__main__":
    step1 = [250, 500, 750, 1000]
    step2 = [2000, 2250, 2500, 2750, 3000, 3250, 3500]
    step = None
    mode = "PAN11"  # ML_One / ML_2CH / PAN11
    if mode == "ML_One":
        dater = data_helpers.DataHelperML(doc_level="sent", embed_dim=300,
                                          target_doc_len=400, target_sent_len=50,
                                          num_fold=5, fold_index=4)
    elif mode == "ML_2CH":
        dater = DataHelperML2CH(doc_level="sent", embed_dim=300,
                                target_doc_len=400, target_sent_len=50,
                                num_fold=5, fold_index=1, truth_file="2_authors.csv")
    elif mode == "PAN11":
        dater = DataHelperPan11(0)

    dater.load_train_data()
    e = evaler()
    e.load(dater)
    path = sys.argv[1]
    if len(sys.argv) == 2:
        e.test(path, step, documentAcc=True, do_is_training=True)
    elif len(sys.argv) > 2:
        steps = list(map(int, sys.argv[2].split("/")))
        for step in steps:
            e.test(path, step, documentAcc=True, do_is_training=True)

