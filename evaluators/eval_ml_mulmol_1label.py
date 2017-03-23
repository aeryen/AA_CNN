#! /usr/bin/env python

import datahelpers.data_helper_mulmol as dh
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support

# DEPRECATED, THIS IS THE CLASS DESIGNED FOR WHEN ML DATA IS 1-OUT-OF-N CLASSIFICATION
# EVALUATOR FOR MULTI-MODALITY CNN

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
        self.x_test, self.y_test, self.vocabulary, self.vocabulary_inv, self.file_sizes, \
        self.pref2, self.pref3, self.suff2, self.suff3, self.pos, \
        self.pref_2_vocab, self.pref_3_vocab, self.suff_2_vocab, self.suff_3_vocab, self.pos_vocab = \
            self.dater.load_test_data()
        self.y_test_scalar = np.argmax(self.y_test, axis=1)
        print("Vocabulary size: {:d}".format(len(self.vocabulary)))
        print("Test set size {:d}".format(len(self.y_test)))

        return self.x_test, self.y_test, self.y_test_scalar, \
               self.pref2, self.pref3, self.suff2, self.suff3, self.pos, \
               self.pref_2_vocab, self.pref_3_vocab, self.suff_2_vocab, self.suff_3_vocab, self.pos_vocab

    def test(self, checkpoint_dir, checkpoint_step, output_file, documentAcc=True):
        training_norm_I = np.array([3243, 11507, 9710, 4456, 8666, 2336, 4812, 5235, 2848, 6124, 3309, 2251, 2384, 3331])

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
                input_pref2 = graph.get_operation_by_name("input_pref2").outputs[0]
                input_pref3 = graph.get_operation_by_name("input_pref3").outputs[0]
                input_suff2 = graph.get_operation_by_name("input_suff2").outputs[0]
                input_suff3 = graph.get_operation_by_name("input_suff3").outputs[0]
                input_pos = graph.get_operation_by_name("input_pos").outputs[0]
                # input_y = graph.get_operation_by_name("input_y").outputs[0]

                dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

                # Tensors we want to evaluate
                scores = graph.get_operation_by_name("output/scores").outputs[0]
                predictions = graph.get_operation_by_name("output/predictions").outputs[0]

                # Generate batches for one epoch
                x_batches = dh.batch_iter(self.x_test, 64, 1, shuffle=False)
                y_batches = dh.batch_iter(self.y_test_scalar, 64, 1, shuffle=False)
                pref2_batches = dh.batch_iter(self.pref2, 64, 1, shuffle=False)
                pref3_batches = dh.batch_iter(self.pref3, 64, 1, shuffle=False)
                suff2_batches = dh.batch_iter(self.suff2, 64, 1, shuffle=False)
                suff3_batches = dh.batch_iter(self.suff3, 64, 1, shuffle=False)
                pos_batches = dh.batch_iter(self.pos, 64, 1, shuffle=False)


                # Collect the predictions here
                all_predictions = []
                for [x_test_batch, y_test_batch,
                     pref2_batch, pref3_batch, suff2_batch, suff3_batch,
                     pos_batch] in zip(x_batches, y_batches,
                                       pref2_batches, pref3_batches, suff2_batches, suff3_batches,
                                       pos_batches):
                    batch_scores, batch_predictions = sess.run([scores, predictions],
                                                               {input_x: x_test_batch, dropout_keep_prob: 1.0,
                                                                input_pref2: pref2_batch, input_pref3: pref3_batch,
                                                                input_suff2: suff2_batch, input_suff3: suff3_batch,
                                                                input_pos: pos_batch})
                    # print batch_predictions
                    all_predictions = np.concatenate([all_predictions, batch_predictions], axis=0)


        # Print accuracy
        np.savetxt('temp.out', all_predictions, fmt='%1.0f')
        correct_predictions = float(sum(all_predictions == self.y_test_scalar))
        # all_predictions = np.array(all_predictions)
        # average_accuracy = all_predictions.sum(axis=0) / float(all_predictions.shape[0])
        average_accuracy = correct_predictions / float(all_predictions.shape[0])
        output_file.write("Test for prob: " + self.dater.problem_name + "\n")
        print("Total number of test examples: {}".format(len(self.y_test_scalar)))
        output_file.write("Total number of test examples: {}\n".format(len(self.y_test_scalar)))
        print "Sent ACC\t" + str(average_accuracy) + "\t\t(cor: " + str(correct_predictions) + ")"
        output_file.write("ACC\t" + str(average_accuracy) + "\n")

        mse = np.mean((all_predictions - self.y_test_scalar) ** 2)
        print "Sent MSE\t" + str(mse)
        output_file.write("MSE\t" + str(mse) + "\n")

        cm = confusion_matrix(self.y_test_scalar, all_predictions)
        np.set_printoptions(precision=2)
        print('Confusion matrix')
        print(cm)
        output_file.write('Confusion matrix \n')
        output_file.write(str(cm) + "\n")
        plt.figure()
        self.plot_confusion_matrix(cm, self.dater)

        # Normalize the confusion matrix by row (i.e by the number of samples
        # in each class)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print('Normalized confusion matrix')
        output_file.write('Normalized confusion matrix' + "\n")
        print(cm_normalized)
        output_file.write(str(cm_normalized) + "\n")
        plt.figure()
        self.plot_confusion_matrix(cm_normalized, dater=self.dater, title='Normalized confusion matrix')
        # plt.show()

        if documentAcc == True:
            file_prediction = []
            sum_to = 0
            for i in range(len(self.file_sizes)):
                p = all_predictions[sum_to:sum_to+self.file_sizes[i]-1]
                sum_to = sum_to + self.file_sizes[i]
                counts = np.bincount(p.astype(int))
                pred_class = np.argmax(counts)
                file_prediction.append(pred_class)
                print counts
                print "pred: " + str(pred_class) + "   " + "true: " + str(self.dater.test_author_index[i])
                output_file.write("File:" + str(i) + "\n")
                output_file.write(str(counts) + "   " +
                                  "pred: " + str(pred_class) + "   " +
                                  "true: " + str(self.dater.test_author_index[i]) + "\n")

            print ""
            output_file.write("\n")

            print "Document ACC"
            output_file.write("Document ACC\n")
            doc_acc = accuracy_score(y_true=self.dater.test_author_index, y_pred=file_prediction, normalize=False)
            print "Doc ACC: " + str(doc_acc)
            output_file.write("Doc ACC: " + str(doc_acc) + "\n")
            doc_acc = accuracy_score(y_true=self.dater.test_author_index, y_pred=file_prediction, normalize=True)
            print "Doc ACC: " + str(doc_acc) + " (norm)"
            output_file.write("Doc ACC: " + str(doc_acc) + " (norm)\n")

            print "precision recall fscore support"
            output_file.write("precision recall fscore support\n")
            for i in range(self.dater.num_of_classes):
                prfs = precision_recall_fscore_support(y_true=self.dater.test_author_index, y_pred=file_prediction,
                                                       average="binary", pos_label=i)
                print "class " + str(i) + ": " + str(prfs)
                output_file.write("class " + str(i) + ": " + str(prfs) + "\n")

            prfs = precision_recall_fscore_support(y_true=self.dater.test_author_index,
                                                   y_pred=file_prediction, average="macro")
            print "avg : " + str(prfs)
            output_file.write("avg : " + str(prfs) + "\n")

        output_file.write("\n")
        output_file.write("\n")

if __name__ == "__main__":
    step1 = [250, 500, 750, 1000]
    step2 = [2000, 2250, 2500, 2750, 3000, 3250, 3500]

    # dater = dh.DataHelper(1)
    dater = dh.DataHelper()
    dater.set_problem(dh.DataHelper.author_codes_I, 100)

    e = evaler()
    e.load(dater)
    output_file = open("100d_multimul_pan12_probI.txt", mode="aw")
    for step in step2:
        e.test("./runs/100d_multimul_pan12_probI/1475024185/checkpoints/", step, output_file, documentAcc=True)
    output_file.close()

