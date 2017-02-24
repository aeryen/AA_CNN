# import sys
# sys.path.append('..')
import logging
import os
import datetime
import time
import tensorflow as tf

# from datahelpers.Data_Helper import Data_Helper
from datahelpers import data_helper_ml_normal as dh
from evaluators import eval_ml_mulmol_d as evaler
from networks.cnn import TextCNN


class TrainTask:

    def __init__(self, data_helper, exp_name, do_dev_split=False, filter_sizes='3,4,5', batch_size=64,
                 dataset="ML", evaluate_every=200, checkpoint_every=500):
        self.data_hlp = data_helper
        self.exp_name = exp_name
        self.dataset = dataset
        self.tag = self.data_hlp.problem_name+"_"+self.exp_name
        self.exp_dir = "../runs/" + self.tag + "/"
        if not os.path.exists(self.exp_dir):
            os.makedirs(self.exp_dir)
        self.log_name = self.exp_dir + "log.txt"

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

        self.filter_sizes = map(int, filter_sizes.split(","))
        self.batch_size = batch_size
        self.evaluate_every = evaluate_every
        self.checkpoint_every = checkpoint_every

        logging.info("setting: %s is %s", "filter_sizes", self.filter_sizes)
        logging.info("setting: %s is %s", "batch_size", self.batch_size)
        logging.info("setting: %s is %s", "evaluate_every", self.evaluate_every)
        logging.info("setting: %s is %s", "checkpoint_every", self.checkpoint_every)

        # Load data
        logging.debug("Loading data...")
        x_shuffled, y_shuffled, _, _, self.embed_matrix = self.data_hlp.load_data()
        logging.debug("Vocabulary Size: {:d}".format(len(self.data_hlp.vocab)))

        self.do_dev_split = do_dev_split
        if self.do_dev_split:
            self.x_train, self.x_dev = x_shuffled[:-500], x_shuffled[-500:]
            self.y_train, self.y_dev = y_shuffled[:-500], y_shuffled[-500:]
            logging.info("Train/Dev split: {:d}/{:d}".format(len(self.y_train), len(self.y_dev)))
        else:
            self.x_train = x_shuffled
            self.x_dev = None
            self.y_train = y_shuffled
            self.y_dev = None
            logging.info("No Train/Dev split")

    def training(self, num_filters, dropout_prob, l2_lambda):
        with tf.Graph().as_default():
            session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
            sess = tf.Session(config=session_conf)
            with sess.as_default():
                cnn = TextCNN(
                    sequence_length=self.x_train.shape[1],
                    num_classes=self.data_hlp.num_of_classes,  # Number of classification classes
                    word_vocab_size=len(self.data_hlp.vocab),
                    embedding_size=self.data_hlp.embedding_dim,
                    filter_sizes=self.filter_sizes,
                    num_filters=num_filters,
                    dataset=self.dataset,
                    l2_reg_lambda=l2_lambda,
                    init_embedding=self.embed_matrix)

                # Define Training procedure
                global_step = tf.Variable(0, name="global_step", trainable=False)
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

            def train_step(x_batch, y_batch):
                """
                A single training step
                """
                feed_dict = {
                    cnn.input_x: x_batch,
                    cnn.input_y: y_batch,

                    cnn.dropout_keep_prob: dropout_prob
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
                    cnn.dropout_keep_prob: 1
                }
                step, summaries, loss, accuracy = sess.run(
                    [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                if writer:
                    writer.add_summary(summaries, step)

            # Generate batches
            batches = dh.DataHelper.batch_iter(list(zip(self.x_train, self.y_train)), self.batch_size, num_epochs=200)

            # Training loop. For each batch...
            for batch in batches:
                x_batch, y_batch = zip(*batch)
                train_step(x_batch, y_batch)
                current_step = tf.train.global_step(sess, global_step)
                if self.do_dev_split and current_step % self.evaluate_every == 0:
                    print("\nEvaluation:")
                    dev_batches = dh.DataHelper.batch_iter(list(zip(self.x_dev, self.y_dev)), 100, 1)
                    for dev_batch in dev_batches:
                        if len(dev_batch) > 0:
                            small_dev_x, small_dev_y = zip(*dev_batch)
                            dev_step(small_dev_x, small_dev_y, writer=dev_summary_writer)
                            print("")
                if current_step % self.checkpoint_every == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))
                if current_step == 100000:  # TODO: change here to stop training early...
                    break
        return timestamp

if __name__ == "__main__":
    dater = dh.DataHelper(doc_level="sent")
    tt = TrainTask(data_helper=dater, exp_name="1c_cnn", batch_size=8, dataset="ML")
    tt.training(num_filters=100, dropout_prob=0.75, l2_lambda=0.1)
