#! /usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import sys
import logging

import os.path
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
from datahelpers import data_helper_ml_normal as data_helpers
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
        self.x_test, self.y_test, self.vocab, self.vocab_inv, self.doc_size_test = self.dater.load_test_data()
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
                all_score = None
                all_predictions = np.zeros([0, 20])
                for [x_test_batch, y_test_batch] in zip(x_batches, y_batches):
                    if do_is_training:
                        batch_scores, batch_predictions = sess.run([scores, predictions],
                                                                   {input_x: x_test_batch, dropout_keep_prob: 1.0,
                                                                    is_training: 0})
                    else:
                        batch_scores, batch_predictions = sess.run([scores, predictions],
                                                                   {input_x: x_test_batch, dropout_keep_prob: 1.0})
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
        eval_log.write("Sent ACC\t" + str(average_accuracy) + "\t\t(cor: " + str(correct_predictions) + ")\n")

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
                print("pred: " + str(pred_class) + "   " + "true: " + str(self.dater.doc_labels_test[i]))
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
            print("Doc ACC: " + str(doc_acc))
            eval_log.write("Doc ACC: " + str(doc_acc) + "\n")

            # print "precision recall fscore support"
            # output_file.write("precision recall fscore support\n")
            # for i in range(self.dater.num_of_classes):
            #     prfs = precision_recall_fscore_support(y_true=self.dater.test_author_index, y_pred=doc_prediction,
            #                                            average="binary", pos_label=i)
            #     print "class " + str(i) + ": " + str(prfs)
            #     output_file.write("class " + str(i) + ": " + str(prfs) + "\n")
            #
            # prfs = precision_recall_fscore_support(y_true=self.dater.test_author_index,
            #                                        y_pred=doc_prediction, average="macro")
            # print "avg : " + str(prfs)
            # output_file.write("avg : " + str(prfs) + "\n")

        eval_log.write("\n")
        eval_log.write("\n")

if __name__ == "__main__":
    step1 = [250, 500, 750, 1000]
    step2 = [2000, 2250, 2500, 2750, 3000, 3250, 3500]
    step = None

    dater = data_helpers.DataHelperML(doc_level="sent", embed_dim=100,
                                      target_doc_len=400, target_sent_len=50,
                                      num_fold=5, fold_index=1)
    dater.load_data()
    e = evaler()
    e.load(dater)
    path = sys.argv[1]
    if len(sys.argv) == 2:
        e.test(path, step, documentAcc=True, do_is_training=False)
    elif len(sys.argv) > 2:
        steps = list(map(int, sys.argv[2].split("/")))
        for step in steps:
            e.test(path, step, documentAcc=True, do_is_training=False)

