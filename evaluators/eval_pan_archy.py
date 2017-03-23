#! /usr/bin/env python

import datahelpers.data_helper_ml_normal as dh
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


class evaler:
    dater = None
    x_test = None
    y_test = None
    y_test_scalar = None
    vocabulary = None
    vocabulary_inv = None
    file_sizes = None

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
        self.x_test, self.y_test, self.vocabulary, self.vocabulary_inv, self.file_sizes = self.dater.load_test_data()
        # self.y_test_scalar = np.argmax(self.y_test, axis=1)
        print("Vocabulary size: {:d}".format(len(self.vocabulary)))
        print("Test set size {:d}".format(len(self.y_test)))

        return self.x_test, self.y_test, self.y_test_scalar

    def test(self, checkpoint_dir, checkpoint_step, output_file, documentAcc=True):
        print("\nEvaluating...\n")

        checkpoint_file = checkpoint_dir + "model-" + str(checkpoint_step)

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

                print checkpoint_file
                output_file.write(checkpoint_file + "\n")

                # Get the placeholders from the graph by name
                input_x = graph.get_operation_by_name("input_x").outputs[0]
                input_y = graph.get_operation_by_name("input_y").outputs[0]
                dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

                # Tensors we want to evaluate
                scores = graph.get_operation_by_name("output/scores").outputs[0]
                predictions = graph.get_operation_by_name("output/predictions").outputs[0]

                # Generate batches for one epoch
                x_batches = dh.batch_iter(self.x_test, 12, 1, shuffle=False)
                y_batches = dh.batch_iter(self.y_test, 12, 1, shuffle=False)

                # Collect the predictions here
                all_score = None
                all_predictions = []
                for x_test_batch, y_test_batch in zip(x_batches, y_batches):
                    batch_scores, batch_predictions = sess.run([scores, predictions],
                                                               {input_x: x_test_batch, dropout_keep_prob: 1.0})
                    # print batch_predictions
                    if all_score is None:
                        all_score = batch_scores
                    else:
                        all_score = np.concatenate([all_score, batch_scores], axis=0)
                    all_predictions = np.concatenate([all_predictions, batch_predictions], axis=0)

                multi_pred = np.array(all_score >= 0).astype(int)
                for row_index in range(len(multi_pred)):
                    if np.sum(multi_pred[row_index]) == 0:
                        max_index = np.argmax(all_score[row_index])
                        multi_pred[row_index][max_index] = 1
                compare_result = np.array(multi_pred == (self.y_test == 1)).astype(int)
                allright = tf.equal(tf.reduce_sum(compare_result, reduction_indices=1), 20)
                allright_accuracy = tf.reduce_mean(tf.cast(allright, "float"), name="allright_accuracy").eval()

        # Print accuracy
        # np.savetxt('temp.out', all_predictions, fmt='%1.0f')

        output_file.write("Test for prob: " + self.dater.problem_name + "\n")
        print("Total number of test examples: {}".format(len(self.y_test)))
        output_file.write("Total number of test examples: {}\n".format(len(self.y_test)))
        # print "Doc ACC\t" + str(allright_accuracy)  # + "\t\t(cor: " + str(correct_predictions) + ")"
        # output_file.write("Doc ACC\t" + str(allright_accuracy) + "\n")

        # mse = np.mean((all_predictions - self.y_test_scalar) ** 2)
        # print "Sent MSE\t" + str(mse)
        # output_file.write("MSE\t" + str(mse) + "\n")
        #
        # cm = confusion_matrix(self.y_test_scalar, all_predictions)
        # np.set_printoptions(precision=2)
        # print('Confusion matrix')
        # print(cm)
        # output_file.write('Confusion matrix \n')
        # output_file.write(str(cm) + "\n")
        # plt.figure()
        # self.plot_confusion_matrix(cm, self.dater)
        #
        # # Normalize the confusion matrix by row (i.e by the number of samples
        # # in each class)
        # cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        # print('Normalized confusion matrix')
        # output_file.write('Normalized confusion matrix' + "\n")
        # print(cm_normalized)
        # output_file.write(str(cm_normalized) + "\n")
        # plt.figure()
        # self.plot_confusion_matrix(cm_normalized, dater=self.dater, title='Normalized confusion matrix')
        # plt.show()

        if documentAcc == True:
            doc_prediction = []
            sum_to = 0
            for i in range(len(self.file_sizes)):
                print np.array_equal(multi_pred[i], self.dater.doc_labels_test[i])
                print "pred: " + str(multi_pred[i]) + "   " + "true: " + str(self.dater.doc_labels_test[i])
                output_file.write("File:" + self.dater.file_id_test[i] + "\n")
                output_file.write("pred: " + str(multi_pred[i]) + "   " +
                                  "true: " + str(self.dater.doc_labels_test[i]) + "\n")

            print ""
            output_file.write("\n")

            print "Document ACC"
            output_file.write("Document ACC\n")
            total_doc = len(self.dater.file_id_test)
            correct = 0.0
            for i in range(len(multi_pred)):
                if np.array_equal(multi_pred[i], self.dater.doc_labels_test[i]):
                    correct += 1
            doc_acc = correct / total_doc
            print "Doc ACC: " + str(doc_acc)
            output_file.write("Doc ACC: " + str(doc_acc) + "\n")

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

        output_file.write("\n")
        output_file.write("\n")

if __name__ == "__main__":
    bold_step2 = [15000]

    dater = dh.DataHelper(doc_level=True, embed_dim=100, target_sent_len=40, target_doc_len=200)
    dater.load_data()

    e = evaler()
    e.load(dater)
    output_file = open("ml_test.txt", mode="aw")
    for step in bold_step2:
        e.test("./runs/ml_100d_doc/1476738122/checkpoints/", step, output_file, documentAcc=True)
    output_file.close()

