import os
import datetime
import tensorflow as tf

from datahelpers.Data import LoadMethod
from datahelpers.data_helper_ml_normal import DataHelperMLNormal
from a.kim_based_doc_cnn import DocCNN
from utils.ArchiveManager import ArchiveManager

am = ArchiveManager("ML", "doc_test", truth_file=None)

doc_len = 400
sent_len = 50
num_class = 20
batch_size = 10
dropout_keep_prob = 0.8
evaluate_every = 10
checkpoint_every = 15
n_steps = 150
out_dir = am.get_exp_dir()
experiment_dir = "E:\\Research\\Paper 03\\AA_CNN_github\\runs\\ML_One_ORIGIN_NEW\\170613_1497377078_labels.csv"

data_hlp = DataHelperMLNormal(doc_level=LoadMethod.DOC, embed_type="glove",
                              embed_dim=300, target_sent_len=50, target_doc_len=400, train_csv_file="labels.csv",
                              total_fold=5, t_fold_index=0)
train_data, vocab, vocab_inv, embed_matrix = data_hlp.get_train_data()
test_data, _, _ = data_hlp.get_test_data()

checkpoint_file = tf.train.latest_checkpoint(experiment_dir + "/checkpoints/", latest_filename=None)

filter_w = []
filter_b = []

g1 = tf.Graph()
with g1.as_default() as g:
    session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        emb_matrix = g1.get_operation_by_name("embedding/W").outputs[0].eval()

        filter_w.append(g1.get_operation_by_name("conv-1-3/W").outputs[0].eval())
        filter_w.append(g1.get_operation_by_name("conv-1-4/W").outputs[0].eval())
        filter_w.append(g1.get_operation_by_name("conv-1-5/W").outputs[0].eval())

        filter_b.append(g1.get_operation_by_name("conv-1-3/b").outputs[0].eval())
        filter_b.append(g1.get_operation_by_name("conv-1-4/b").outputs[0].eval())
        filter_b.append(g1.get_operation_by_name("conv-1-5/b").outputs[0].eval())

        fc_w = g1.get_operation_by_name("W").outputs[0].eval()
        fc_b = g1.get_operation_by_name("output/b").outputs[0].eval()

tf.reset_default_graph()

graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    sess = tf.Session(config=session_conf)
    cnn = DocCNN(
        doc_length=doc_len,
        sent_length=sent_len,
        num_classes=data_hlp.num_of_classes,
        word_vocab_size=len(data_hlp.vocab),
        embedding_size=data_hlp.embedding_dim,
        filter_sizes=[3, 4, 5],
        num_filters=100,
        l2_reg_lambda=0.1,
        init_embedding=emb_matrix,
        init_filter_w=filter_w,
        init_filter_b=filter_b,
        fc_w=fc_w,
        fc_b=fc_b)
    with sess.as_default():

        global_step = tf.Variable(0, name="global_step", trainable=False)

        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Keep track of gradient values and sparsity (optional)
        with tf.name_scope('grad_summary'):
            grad_summaries = []
            for g, v in grads_and_vars:
                if g is not None:
                    grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name.replace(":", "_")), g)
                    sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name.replace(":", "_")),
                                                         tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)
            grad_summaries_merged = tf.summary.merge(grad_summaries)

        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", cnn.loss)
        acc_sig_summary = tf.summary.scalar("accuracy_sigmoid", cnn.accuracy_sigmoid)
        acc_max_summary = tf.summary.scalar("accuracy_max", cnn.accuracy_max)

        # Train Summaries
        with tf.name_scope('train_summary'):
            train_summary_op = tf.summary.merge(
                [loss_summary, acc_sig_summary, acc_max_summary, grad_summaries_merged])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        # Dev summaries
        with tf.name_scope('dev_summary'):
            dev_summary_op = tf.summary.merge([loss_summary, acc_sig_summary, acc_max_summary])
            dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=7)

        # Initialize all variables
        sess.run(tf.global_variables_initializer())


    def train_step(x_batch, y_batch, len_batch):
        """
        A single training step
        """
        feed_dict = {
            cnn.input_x: x_batch,
            cnn.input_y: y_batch,
            cnn.doc_len: len_batch,
            cnn.dropout_keep_prob: dropout_keep_prob,
        }
        _, step, summaries, loss, accuracy, acc_max = sess.run(
            [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy_sigmoid, cnn.accuracy_max],
            feed_dict)
        time_str = datetime.datetime.now().isoformat()
        print(("{}: step {}, loss {:g}, acc {:g}, acc_max {:g}".format(time_str, step, loss, accuracy, acc_max)))
        train_summary_writer.add_summary(summaries, step)


    def dev_step(x_batch, y_batch, len_batch, writer=None):
        """
        Evaluates model on a dev set
        """
        feed_dict = {
            cnn.input_x: x_batch,
            cnn.input_y: y_batch,
            cnn.doc_len: len_batch,
            cnn.dropout_keep_prob: 1,
        }
        step, summaries, loss, accuracy, acc_max = sess.run(
            [global_step, dev_summary_op, cnn.loss, cnn.accuracy_sigmoid, cnn.accuracy_max],
            feed_dict)
        time_str = datetime.datetime.now().isoformat()
        print(("{}: step {}, loss {:g}, acc {:g}, acc_max {:g}".format(time_str, step, loss, accuracy, acc_max)))
        if writer:
            writer.add_summary(summaries, step)


    # Generate batches
    batches = DataHelperMLNormal.batch_iter(list(zip(train_data.value, train_data.label_instance, train_data.doc_size)),
                                            batch_size, num_epochs=300)

    # Training loop. For each batch...
    for batch in batches:
        x_batch, y_batch, len_batch = list(zip(*batch))
        train_step(x_batch, y_batch, len_batch)

        current_step = tf.train.global_step(sess, global_step)
        if current_step % evaluate_every == 0:
            print("\nEvaluation:")
            dev_batches = DataHelperMLNormal.batch_iter(list(zip(test_data.value, test_data.label_instance, train_data.doc_size)),
                                                        batch_size, 1)
            for dev_batch in dev_batches:
                if len(dev_batch) > 0:
                    small_dev_x, small_dev_y, len_batch = list(zip(*dev_batch))
                    dev_step(small_dev_x, small_dev_y, len_batch, writer=dev_summary_writer)
                    print("")

        if current_step % checkpoint_every == 0:
            path = saver.save(sess, checkpoint_prefix, global_step=current_step)
            print(("Saved model checkpoint to {}\n".format(path)))
        if n_steps is not None and current_step >= n_steps:
            break
