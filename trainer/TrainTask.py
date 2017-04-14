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

    def __init__(self, data_helper, am, input_component, exp_name, batch_size=64,
                 evaluate_every=1000, checkpoint_every=5000, max_to_keep=7):
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

        # Load data
        logging.debug("Loading data...")
        if "Six" in input_component:
            self.x_train, self.pos_train, _, self.p2_train, self.p3_train, self.s2_train, self.s3_train, self.y_train, \
            _, _, self.embed_matrix_glv = self.data_hlp.load_data()
            self.pref2_vocab_size = len(self.data_hlp.p2_vocab)
            self.pref3_vocab_size = len(self.data_hlp.p3_vocab)
            self.suff2_vocab_size = len(self.data_hlp.s2_vocab)
            self.suff3_vocab_size = len(self.data_hlp.s3_vocab)
            self.pos_vocab_size = len(self.data_hlp.pos_vocab)
        elif "One" in input_component:
            self.pref2_vocab_size = None
            self.pref3_vocab_size = None
            self.suff2_vocab_size = None
            self.suff3_vocab_size = None
            self.pos_vocab_size = None
            self.x_train, self.y_train, _, _, self.embed_matrix_glv = self.data_hlp.load_data()
        elif "2CH" in input_component:
            self.pref2_vocab_size = None
            self.pref3_vocab_size = None
            self.suff2_vocab_size = None
            self.suff3_vocab_size = None
            self.pos_vocab_size = None
            self.x_train, self.y_train, _, _, self.embed_matrix_glv, self.embed_matrix_w2v = self.data_hlp.load_data()
        else:
            raise NotImplementedError

        logging.debug("Vocabulary Size: {:d}".format(len(self.data_hlp.vocab)))

        if "Six" in input_component:
            self.x_dev, self.pos_test, _, self.p2_test, self.p3_test, \
                self.s2_test, self.s3_test, self.y_dev, _, _, _ = self.data_hlp.load_test_data()
        elif "One" or "2CH" in input_component:
            self.x_dev, self.y_dev, _, _, _ = self.data_hlp.load_test_data()
        else:
            raise NotImplementedError

        logging.info("Train/Dev split: {:d}/{:d}".format(len(self.y_train), len(self.y_dev)))

    def training(self, filter_sizes=[[3, 4, 5]], num_filters=100, dropout_keep_prob=1.0, n_steps=None, l2_lambda=0.0,
                 dropout=False, batch_normalize=False, elu=False, n_conv=1, fc=[]):
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

        if "DocLevel" in self.input_component:
            doc_length = self.x_train.shape[2]
        else:
            doc_length = None
            
        with tf.Graph().as_default():
            session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
            sess = tf.Session(config=session_conf)
            cnn = TextCNN(
                document_length=doc_length,
                sequence_length=self.x_train.shape[1],
                num_classes=self.data_hlp.num_of_classes,  # Number of classification classes
                word_vocab_size=len(self.data_hlp.vocab),
                embedding_size=self.data_hlp.embedding_dim,
                input_component=self.input_component,
                middle_component=self.exp_name,
                filter_sizes=filter_sizes,
                num_filters=num_filters,
                pref2_vocab_size=self.pref2_vocab_size,
                pref3_vocab_size=self.pref3_vocab_size,
                suff2_vocab_size=self.suff2_vocab_size,
                suff3_vocab_size=self.suff3_vocab_size,
                pos_vocab_size=self.pos_vocab_size,
                dataset=self.data_hlp.problem_name,
                l2_reg_lambda=l2_lambda,
                init_embedding_glv=self.embed_matrix_glv,
                init_embedding_w2v=self.embed_matrix_w2v,
                dropout=dropout,
                batch_normalize=batch_normalize,
                elu=elu,
                n_conv=n_conv,
                fc=fc)
            with sess.as_default():

                # Define Training procedure

                global_step = tf.Variable(0, name="global_step", trainable=False)

                if batch_normalize == True:
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

            if "One" in self.input_component or "2CH" in self.input_component:
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
            if "One" in self.input_component or "2CH" in self.input_component:
                batches = dh.DataHelperML.batch_iter(list(zip(self.x_train, self.y_train)), self.batch_size,
                                                     num_epochs=300)
            elif "Six" in self.input_component:
                batches = dh.DataHelperML.batch_iter(list(zip(self.x_train, self.y_train, self.p2_train, self.p3_train,
                                                              self.s2_train, self.s3_train, self.pos_train)),
                                                     self.batch_size, num_epochs=300)
            else:
                raise NotImplementedError
            # Training loop. For each batch...
            for batch in batches:
                if "One" in self.input_component or "2CH" in self.input_component:
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
                    if "One" in self.input_component or "2CH" in self.input_component:
                        dev_batches = dh.DataHelperML.batch_iter(list(zip(self.x_dev, self.y_dev)), self.batch_size, 1)
                        for dev_batch in dev_batches:
                            if len(dev_batch) > 0:
                                small_dev_x, small_dev_y = list(zip(*dev_batch))
                                dev_step(small_dev_x, small_dev_y, writer=dev_summary_writer)
                                print("")
                    elif "Six" in self.input_component:
                        dev_batches = dh6.DataHelperMulMol6.batch_iter(list(zip(self.x_dev, self.y_dev, self.p2_test,
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
