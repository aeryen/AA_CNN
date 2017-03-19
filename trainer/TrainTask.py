import logging
import os
import datetime
import time
import tensorflow as tf
from networks.cnn import TextCNN
from datahelpers import data_helper_ml_mulmol6file as dh6
from datahelpers import data_helper_ml_normal as dh


class TrainTask:
    """
    This is the MAIN.
    The class set corresponding parameters, log the setting in runs folder.
    the class then create a NN and initialize it.
    lastly the class data batches and feed them into NN for training.
    Currently it only- works with ML data, i'll expand this to be more flexible in the near future.
    """

    def __init__(self, data_helper, input_component, exp_name, do_dev_split=True, filter_sizes=[3,4,5], batch_size=64,
                 dataset="ML", evaluate_every=1000, checkpoint_every=5000):
        self.data_hlp = data_helper
        self.exp_name = exp_name
        self.input_component = input_component
        self.dataset = dataset
        # the problem tag identifies the experiment setting, currently data name + experiment name
        self.tag = self.data_hlp.problem_name+"_"+self.exp_name
        self.exp_dir = "../runs/" + self.tag + "/"
        if not os.path.exists(self.exp_dir):
            os.makedirs(self.exp_dir)
        self.log_name = self.exp_dir + "log.txt"

        # logging facility, log both into file and console
        logging.basicConfig(level=logging.DEBUG,
                            format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                            datefmt='%m-%d %H:%M',
                            filename=self.log_name,
                            filemode='aw')
        console_logger = logging.StreamHandler()
        logging.getLogger('').addHandler(console_logger)

        logging.warning('TrainTask instance initiated')
        logging.info("Logging to: " + self.log_name)

        logging.info("current data is: " + self.data_hlp.problem_name)
        logging.info("current experiment is: " + self.exp_name)

        # network parameters
        self.filter_sizes = filter_sizes
        self.batch_size = batch_size
        self.evaluate_every = evaluate_every
        self.checkpoint_every = checkpoint_every

        logging.info("setting: %s is %s", "filter_sizes", self.filter_sizes)
        logging.info("setting: %s is %s", "batch_size", self.batch_size)
        logging.info("setting: %s is %s", "evaluate_every", self.evaluate_every)
        logging.info("setting: %s is %s", "checkpoint_every", self.checkpoint_every)

        # Load data
        logging.debug("Loading data...")
        if input_component=="SixChannel":
            self.x_train, self.pos_train, _, self.p2_train, self.p3_train, self.s2_train, self.s3_train, self.y_train, \
            _, _,self.embed_matrix = self.data_hlp.load_data()
            self.pref2_vocab_size = len(self.data_hlp.p2_vocab)
            self.pref3_vocab_size = len(self.data_hlp.p3_vocab)
            self.suff2_vocab_size = len(self.data_hlp.s2_vocab)
            self.suff3_vocab_size = len(self.data_hlp.s3_vocab)
            self.pos_vocab_size = len(self.data_hlp.pos_vocab)
        elif input_component=="OneChannel":
            self.pref2_vocab_size = None
            self.pref3_vocab_size = None
            self.suff2_vocab_size = None
            self.suff3_vocab_size = None
            self.pos_vocab_size = None
            self.x_train, self.y_train, _, _, self.embed_matrix = self.data_hlp.load_data()
        else:
            raise NotImplementedError

        logging.debug("Vocabulary Size: {:d}".format(len(self.data_hlp.vocab)))

        self.do_dev_split = do_dev_split

        if self.do_dev_split:
            if input_component == "SixChannel":
                self.x_dev, self.pos_test, _, self.p2_test, self.p3_test, \
                self.s2_test, self.s3_test, self.y_dev, _, _, _ = self.data_hlp.load_test_data()
            elif input_component=="OneChannel":
                self.x_dev, self.y_dev, _, _, _ = self.data_hlp.load_test_data()
            else:
                raise NotImplementedError

            logging.info("Train/Dev split: {:d}/{:d}".format(len(self.y_train), len(self.y_dev)))
        else:
            self.x_dev = None
            self.y_dev = None
            logging.info("No Train/Dev split")



    def training(self, num_filters, dropout_keep_prob, n_steps, l2_lambda=0.0, dropout=False, batch_normalize=False,
                 elu=False, n_conv = 1, fc=[]):
        with tf.Graph().as_default():
            session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
            sess = tf.Session(config=session_conf)
            with sess.as_default():
                cnn = TextCNN(
                    sequence_length=self.x_train.shape[1],
                    num_classes=self.data_hlp.num_of_classes,  # Number of classification classes
                    word_vocab_size=len(self.data_hlp.vocab),
                    embedding_size=self.data_hlp.embedding_dim,
                    input_component=self.input_component,
                    middle_component=self.exp_name,
                    filter_sizes=self.filter_sizes,
                    num_filters=num_filters,
                    pref2_vocab_size=self.pref2_vocab_size,
                    pref3_vocab_size=self.pref3_vocab_size,
                    suff2_vocab_size=self.suff2_vocab_size,
                    suff3_vocab_size=self.suff3_vocab_size,
                    pos_vocab_size=self.pos_vocab_size,
                    dataset=self.dataset,
                    l2_reg_lambda=l2_lambda,
                    init_embedding=self.embed_matrix,
                    dropout=dropout,
                    batch_normalize=batch_normalize,
                    elu=elu,
                    n_conv = n_conv,
                    fc=fc)

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
                            grad_hist_summary = tf.histogram_summary("{}/grad/hist".format(v.name), g)
                            sparsity_summary = tf.scalar_summary("{}/grad/sparsity".format(v.name),
                                                                 tf.nn.zero_fraction(g))
                            grad_summaries.append(grad_hist_summary)
                            grad_summaries.append(sparsity_summary)
                    grad_summaries_merged = tf.merge_summary(grad_summaries)

                # Output directory for models and summaries
                timestamp = str(int(time.time()))
                out_dir = os.path.abspath(os.path.join(self.exp_dir, timestamp))
                logging.info("Model in {}\n".format(out_dir))

                # Summaries for loss and accuracy
                loss_summary = tf.scalar_summary("loss", cnn.loss)
                acc_summary = tf.scalar_summary("accuracy", cnn.accuracy)

                # Train Summaries
                with tf.name_scope('train_summary'):
                    train_summary_op = tf.merge_summary(
                        [loss_summary, acc_summary, grad_summaries_merged])
                    train_summary_dir = os.path.join(out_dir, "summaries", "train")
                    train_summary_writer = tf.train.SummaryWriter(train_summary_dir, sess.graph_def)

                # Dev summaries
                with tf.name_scope('dev_summary'):
                    dev_summary_op = tf.merge_summary([loss_summary, acc_summary])
                    dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
                    dev_summary_writer = tf.train.SummaryWriter(dev_summary_dir, sess.graph_def)

                # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
                checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
                checkpoint_prefix = os.path.join(checkpoint_dir, "model")
                if not os.path.exists(checkpoint_dir):
                    os.makedirs(checkpoint_dir)
                saver = tf.train.Saver(var_list=tf.all_variables(), max_to_keep=7)

                # Initialize all variables
                sess.run(tf.initialize_all_variables())

            last_checkpoint = 0

            if self.input_component=="OneChannel":
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
                    print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
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
                    print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                    if writer:
                        writer.add_summary(summaries, step)

            elif self.input_component=="SixChannel":
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

                        cnn.dropout_keep_prob: dropout_keep_prob
                    }
                    _, step, summaries, loss, accuracy = sess.run(
                        [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                        feed_dict)
                    time_str = datetime.datetime.now().isoformat()
                    print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                    train_summary_writer.add_summary(summaries, step)

                def dev_step(x_batch, y_batch, pref2_batch, pref3_batch, suff2_batch, suff3_batch, pos_batch, writer=None):
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
                        cnn.dropout_keep_prob: 1
                    }
                    step, summaries, loss, accuracy = sess.run(
                        [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
                        feed_dict)
                    time_str = datetime.datetime.now().isoformat()
                    print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                    if writer:
                        writer.add_summary(summaries, step)
            else:
                raise NotImplementedError

            # Generate batches
            if self.input_component == "OneChannel":
                batches = dh.DataHelper.batch_iter(list(zip(self.x_train, self.y_train)), self.batch_size,
                                                   num_epochs=300)
            elif self.input_component == "SixChannel":
                batches = dh.DataHelper.batch_iter(list(zip(self.x_train, self.y_train, self.p2_train, self.p3_train,
                                                            self.s2_train, self.s3_train, self.pos_train)),
                                                   self.batch_size, num_epochs=300)
            else:
                raise NotImplementedError
            # Training loop. For each batch...
            for batch in batches:
                if self.input_component=="OneChannel":
                    x_batch, y_batch = zip(*batch)
                    train_step(x_batch, y_batch)
                elif self.input_component=="SixChannel":
                    x_batch, y_batch, pref2_batch, pref3_batch, suff2_batch, suff3_batch, pos_batch = zip(*batch)
                    train_step(x_batch, y_batch, pref2_batch, pref3_batch, suff2_batch, suff3_batch, pos_batch)
                else:
                    raise NotImplementedError

                current_step = tf.train.global_step(sess, global_step)
                if self.do_dev_split and current_step % self.evaluate_every == 0:
                    print("\nEvaluation:")
                    if self.input_component=="OneChannel":
                        dev_batches = dh.DataHelper.batch_iter(list(zip(self.x_dev, self.y_dev)), self.batch_size, 1)
                        for dev_batch in dev_batches:
                            if len(dev_batch) > 0:
                                small_dev_x, small_dev_y = zip(*dev_batch)
                                dev_step(small_dev_x, small_dev_y, writer=dev_summary_writer)
                                print("")
                    elif self.input_component=="SixChannel":
                        dev_batches = dh6.DataHelperMulMol6.batch_iter(list(zip(self.x_dev, self.y_dev, self.p2_test,
                                                                               self.p3_test,self.s2_test,
                                                                               self.s3_test, self.pos_test)), self.batch_size, 1)
                        for dev_batch in dev_batches:
                            if len(dev_batch) > 0:
                                small_dev_x, small_dev_y, small_p2_test, small_p3_test, small_s2_test, small_s3_test,\
                                    small_post_test = zip(*dev_batch)
                                dev_step(small_dev_x, small_dev_y, small_p2_test, small_p3_test, small_s2_test,
                                         small_s3_test, small_post_test, writer=dev_summary_writer)
                                print("")
                    else:
                        raise NotImplementedError

                if current_step % self.checkpoint_every == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    last_checkpoint=current_step
                    print("Saved model checkpoint to {}\n".format(path))
                if current_step == n_steps:
                    break
        return timestamp, last_checkpoint

