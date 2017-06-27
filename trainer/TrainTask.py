import logging
import os
import datetime
import time
import tensorflow as tf
from networks.cnn import TextCNN
from datahelpers import data_helper_ml_mulmol6_Read as dh6
from datahelpers import data_helper_ml_normal as dh
import utils.ArchiveManager as AM


class TrainTask:
    """
    This is the MAIN.
    The class set corresponding parameters, log the setting in runs folder.
    the class then create a NN and initialize it.
    lastly the class data batches and feed them into NN for training.
    Currently it only- works with ML data, i'll expand this to be more flexible in the near future.
    """

    def __init__(self, data_helper, am, input_component, exp_name, batch_size,
                 evaluate_every, checkpoint_every, max_to_keep):
        self.data_hlp = data_helper
        self.exp_name = exp_name
        self.input_component = input_component
        self.am = am

        logging.warning('TrainTask instance initiated: ' + AM.get_date())
        logging.info("Logging to: " + self.am.get_exp_log_path())

        logging.info("current data is: " + self.data_hlp.problem_name)
        logging.info("current experiment is: " + self.exp_name)

        # network parameters
        self.batch_size = batch_size
        self.evaluate_every = evaluate_every
        self.checkpoint_every = checkpoint_every
        self.max_to_keep = max_to_keep

        logging.info("setting: %s is %s", "batch_size", self.batch_size)
        logging.info("setting: %s is %s", "evaluate_every", self.evaluate_every)
        logging.info("setting: %s is %s", "checkpoint_every", self.checkpoint_every)

        self.train_data = self.data_hlp.get_train_data()
        self.test_data = self.data_hlp.get_test_data()

        logging.info("Vocabulary Size: {:d}".format(len(self.train_data.vocab)))
        logging.info("Train/Dev split (DOC): {:d}/{:d}".
                     format(len(self.train_data.file_id), len(self.train_data.file_id)))
        logging.info("Train/Dev split (IST): {:d}/{:d}".
                     format(len(self.train_data.label_instance), len(self.test_data.label_instance)))

    def training(self, filter_sizes, num_filters, dropout_keep_prob, n_steps, l2_lambda,
                 dropout, batch_normalize, elu, n_conv, fc):
        logging.info("setting: %s is %s", "filter_sizes", filter_sizes)
        logging.info("setting: %s is %s", "num_filters", num_filters)
        logging.info("setting: %s is %s", "dropout_keep_prob", dropout_keep_prob)
        logging.info("setting: %s is %s", "n_steps", n_steps)
        logging.info("setting: %s is %s", "l2_lambda", l2_lambda)
        logging.info("setting: %s is %s", "dropout", dropout)
        logging.info("setting: %s is %s", "batch_normalize", batch_normalize)
        logging.info("setting: %s is %s", "elu", elu)
        logging.info("setting: %s is %s", "n_conv", n_conv)
        logging.info("setting: %s is %s", "fc", fc)

        with tf.Graph().as_default():
            session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
            sess = tf.Session(config=session_conf)
            cnn = TextCNN(
                data=self.train_data,
                document_length=self.data_hlp.target_doc_len,
                sequence_length=self.data_hlp.target_sent_len,
                num_classes=self.data_hlp.num_of_classes,  # Number of classification classes
                embedding_size=self.data_hlp.embedding_dim,
                input_component=self.input_component,
                middle_component=self.exp_name,
                filter_sizes=filter_sizes,
                num_filters=num_filters,
                l2_reg_lambda=l2_lambda,
                dropout=dropout,
                batch_normalize=batch_normalize,
                elu=elu,
                n_conv=n_conv,
                fc=fc)
            with sess.as_default():

                # Define Training procedure

                global_step = tf.Variable(0, name="global_step", trainable=False)

                if batch_normalize:
                    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                    with tf.control_dependencies(update_ops):
                        optimizer = tf.train.AdamOptimizer(1e-3)
                        grads_and_vars = optimizer.compute_gradients(cnn.loss)
                        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
                else:
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

                # Output directory for models and summaries
                out_dir = self.am.get_exp_dir()
                logging.info("Model in {}\n".format(self.am.get_exp_dir()))

                # Summaries for loss and accuracy
                loss_summary = tf.summary.scalar("loss", cnn.loss)
                acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

                # Train Summaries
                with tf.name_scope('train_summary'):
                    train_summary_op = tf.summary.merge(
                        [loss_summary, acc_summary, grad_summaries_merged])
                    train_summary_dir = os.path.join(out_dir, "summaries", "train")
                    train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

                # Dev summaries
                with tf.name_scope('dev_summary'):
                    dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
                    dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
                    dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

                # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
                checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
                checkpoint_prefix = os.path.join(checkpoint_dir, "model")
                if not os.path.exists(checkpoint_dir):
                    os.makedirs(checkpoint_dir)
                saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=self.max_to_keep)

                # Initialize all variables
                sess.run(tf.global_variables_initializer())

            if "One" in self.input_component or "2CH" in self.input_component or "PAN11" in self.input_component:
                def train_step(x_batch, y_batch):
                    """
                    A single training step
                    """
                    feed_dict = {
                        cnn.input_x: x_batch,
                        cnn.input_y: y_batch,
                        cnn.dropout_keep_prob: dropout_keep_prob,
                        cnn.is_training: 1
                    }
                    _, step, summaries, loss, accuracy = sess.run(
                        [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                        feed_dict)
                    time_str = datetime.datetime.now().isoformat()
                    print(("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy)))
                    train_summary_writer.add_summary(summaries, step)

                def dev_step(x_batch, y_batch, writer=None):
                    """
                    Evaluates model on a dev set
                    """
                    feed_dict = {
                        cnn.input_x: x_batch,
                        cnn.input_y: y_batch,
                        cnn.dropout_keep_prob: 1,
                        cnn.is_training: 0
                    }
                    step, summaries, loss, accuracy = sess.run(
                        [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
                        feed_dict)
                    time_str = datetime.datetime.now().isoformat()
                    print(("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy)))
                    if writer:
                        writer.add_summary(summaries, step)

            elif "Six" in self.input_component:
                def train_step(x_batch, y_batch, pref2_batch, pref3_batch, suff2_batch, suff3_batch, pos_batch):
                    """
                    A single training step
                    """
                    feed_dict = {
                        cnn.input_x: x_batch,
                        cnn.input_y: y_batch,

                        cnn.input_pref2: pref2_batch,
                        cnn.input_pref3: pref3_batch,
                        cnn.input_suff2: suff2_batch,
                        cnn.input_suff3: suff3_batch,
                        cnn.input_pos: pos_batch,

                        cnn.dropout_keep_prob: dropout_keep_prob,
                        cnn.is_training: 1
                    }
                    _, step, summaries, loss, accuracy = sess.run(
                        [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                        feed_dict)
                    time_str = datetime.datetime.now().isoformat()
                    print(("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy)))
                    train_summary_writer.add_summary(summaries, step)

                def dev_step(x_batch, y_batch, pref2_batch, pref3_batch, suff2_batch, suff3_batch, pos_batch,
                             writer=None):
                    """
                    Evaluates model on a dev set
                    """
                    feed_dict = {
                        cnn.input_x: x_batch,
                        cnn.input_y: y_batch,
                        cnn.input_pref2: pref2_batch,
                        cnn.input_pref3: pref3_batch,
                        cnn.input_suff2: suff2_batch,
                        cnn.input_suff3: suff3_batch,
                        cnn.input_pos: pos_batch,
                        cnn.dropout_keep_prob: 1,
                        cnn.is_training: 0
                    }
                    step, summaries, loss, accuracy = sess.run(
                        [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
                        feed_dict)
                    time_str = datetime.datetime.now().isoformat()
                    print(("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy)))
                    if writer:
                        writer.add_summary(summaries, step)
            else:
                raise NotImplementedError

            # Generate batches
            if "One" in self.input_component or "2CH" in self.input_component or "PAN11" in self.input_component:
                batches = dh.DataHelperML.batch_iter(list(zip(self.train_data.value, self.train_data.label_instance)), self.batch_size,
                                                     num_epochs=300)
            elif "Six" in self.input_component:
                batches = dh.DataHelperML.batch_iter(list(zip(self.train_data, self.y_train, self.p2_train, self.p3_train,
                                                              self.s2_train, self.s3_train, self.pos_train)),
                                                     self.batch_size, num_epochs=300)
            else:
                raise NotImplementedError
            # Training loop. For each batch...
            for batch in batches:
                if "One" in self.input_component or "2CH" in self.input_component or "PAN11" in self.input_component:
                    x_batch, y_batch = list(zip(*batch))
                    train_step(x_batch, y_batch)
                elif "Six" in self.input_component:
                    x_batch, y_batch, pref2_batch, pref3_batch, suff2_batch, suff3_batch, pos_batch = list(zip(*batch))
                    train_step(x_batch, y_batch, pref2_batch, pref3_batch, suff2_batch, suff3_batch, pos_batch)
                else:
                    raise NotImplementedError

                current_step = tf.train.global_step(sess, global_step)
                if current_step % self.evaluate_every == 0:
                    print("\nEvaluation:")
                    if "One" in self.input_component or "2CH" in self.input_component or "2CH" in self.input_component:
                        dev_batches = dh.DataHelperML.batch_iter(list(zip(self.test_data.value, self.test_data.label_instance)), self.batch_size, 1)
                        for dev_batch in dev_batches:
                            if len(dev_batch) > 0:
                                small_dev_x, small_dev_y = list(zip(*dev_batch))
                                dev_step(small_dev_x, small_dev_y, writer=dev_summary_writer)
                                print("")
                    elif "Six" in self.input_component:
                        dev_batches = dh6.DataHelperMulMol6.batch_iter(list(zip(self.test_data, self.y_dev, self.p2_test,
                                                                                self.p3_test, self.s2_test,
                                                                                self.s3_test, self.pos_test)),
                                                                       self.batch_size, 1)
                        for dev_batch in dev_batches:
                            if len(dev_batch) > 0:
                                small_dev_x, small_dev_y, small_p2_test, small_p3_test, small_s2_test, small_s3_test, \
                                small_post_test = list(zip(*dev_batch))
                                dev_step(small_dev_x, small_dev_y, small_p2_test, small_p3_test, small_s2_test,
                                         small_s3_test, small_post_test, writer=dev_summary_writer)
                                print("")
                    else:
                        raise NotImplementedError

                if current_step % self.checkpoint_every == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print(("Saved model checkpoint to {}\n".format(path)))
                if n_steps is not None and current_step >= n_steps:
                    break
