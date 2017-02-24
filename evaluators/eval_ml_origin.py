#! /usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from datahelpers import data_helper_ml_normal as data_helpers


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
        print("Vocabulary size: {:d}".format(len(self.vocab)))
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
                x_batches = data_helpers.DataHelper.batch_iter(self.x_test, 64, 1, shuffle=False)
                y_batches = data_helpers.DataHelper.batch_iter(self.y_test, 64, 1, shuffle=False)

                # Collect the predictions here
                all_score = None
                all_predictions = np.zeros([0, 20])
                for [x_test_batch, y_test_batch] in zip(x_batches, y_batches):
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

        output_file.write("Test for prob: " + self.dater.problem_name + "\n")
        print("Total number of test examples: {}".format(len(self.y_test)))
        output_file.write("Total number of test examples: {}\n".format(len(self.y_test)))
        print "Sent ACC\t" + str(average_accuracy)  # + "\t\t(cor: " + str(correct_predictions) + ")"
        output_file.write("ACC\t" + str(average_accuracy) + "\n")

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
                print "pred: " + str(pred_class) + "   " + "true: " + str(self.dater.doc_labels_test[i])
                output_file.write("File:" + self.dater.file_id_test[i] + "\n")
                output_file.write("pred: " + str(pred_class) + "   " +
                                  "true: " + str(self.dater.doc_labels_test[i]) + "\n")

            print ""
            output_file.write("\n")

            print "Document ACC"
            output_file.write("Document ACC\n")
            total_doc = len(self.dater.file_id_test)
            correct = 0.0
            for i in range(len(doc_prediction)):
                if np.array_equal(doc_prediction[i], self.dater.doc_labels_test[i]):
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
    step1 = [250, 500, 750, 1000]
    step2 = [2000, 2250, 2500, 2750, 3000, 3250, 3500]

    dater = data_helpers.DataHelper(doc_level="sent")
    dater.load_data()
    e = evaler()
    e.load(dater)
    output_file = open("ml_test.txt", mode="aw")
    for step in [97000]:
        e.test("../runs/ML_2c_cnn/1487894877/checkpoints/", step, output_file, documentAcc=True)
    output_file.close()

