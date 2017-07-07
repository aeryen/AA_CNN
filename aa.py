from timeit import default_timer as timer
# from datahelpers.data_helper_ml_mulmol6_OnTheFly import DataHelperMulMol6
from datahelpers.data_helper_ml_normal import DataHelperMLNormal
from datahelpers.data_helper_ml_2chan import DataHelperML2CH
from datahelpers.data_helper_ml_mulmol6_OnTheFly import DataHelperMLFly
from datahelpers.data_helper_pan11 import DataHelperPan11
from trainer import TrainTask as tr
from trainer import TrainTaskLite as ttl
from evaluators import eval_ml_mulmol_d as evaler
from evaluators import eval_ml_origin as evaler_one
from evaluators import eval_pan11 as evaler_pan
from utils.ArchiveManager import ArchiveManager
from datahelpers.Data import LoadMethod
import logging


def get_exp_logger(am):
    log_path = am.get_exp_log_path()
    # logging facility, log both into file and console
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        datefmt='%m-%d %H:%M',
                        filename=log_path,
                        filemode='w+')
    console_logger = logging.StreamHandler()
    logging.getLogger('').addHandler(console_logger)
    logging.info("log created: " + log_path)


if __name__ == "__main__":

    ###############################################
    # exp_names you can choose from at this point:
    #
    # Input Components:
    #
    # * ML_One
    # * ML_2CH
    # * ML_Six
    # * ML_One_DocLevel
    # * PAN11
    # * PAN11_2CH
    #
    # Middle Components:
    #
    # * NParallelConvOnePoolNFC
    # * NConvDocConvNFC
    # * ParallelJoinedConv
    # * NCrossSizeParallelConvNFC
    # * InceptionLike
    # * PureRNN
    ################################################

    input_component = "ML_2CH"
    middle_component = "NCrossSizeParallelConvNFC"
    truth_file = "17_papers.csv"

    am = ArchiveManager(input_component, middle_component, truth_file=truth_file)
    get_exp_logger(am)
    logging.warning('===================================================')
    logging.debug("Loading data...")

    if input_component == "ML_One":
        dater = DataHelperMLNormal(doc_level=LoadMethod.SENT, embed_type="glove",
                                   embed_dim=300, target_sent_len=50, target_doc_len=None, train_csv_file=truth_file,
                                   total_fold=5, t_fold_index=0)
        ev = evaler_one.Evaluator()
    elif input_component == "ML_FLY":
        dater = DataHelperMLFly(doc_level=LoadMethod.SENT, embed_type="glove",
                                embed_dim=300, target_sent_len=50, target_doc_len=None, train_csv_file=truth_file,
                                total_fold=5, t_fold_index=0)
        ev = evaler_one.Evaluator()
    elif input_component == "ML_2CH":
        dater = DataHelperML2CH(doc_level=LoadMethod.SENT, embed_type="both",
                                embed_dim=300, target_sent_len=50, target_doc_len=None, train_csv_file=truth_file,
                                total_fold=5, t_fold_index=0)
        ev = evaler_one.Evaluator()
    elif input_component == "ML_Six":
        dater = DataHelperMulMol6(doc_level="sent", num_fold=5, fold_index=4, embed_type="glove",
                                  embed_dim=300, target_sent_len=50, target_doc_len=400)
        ev = evaler.evaler()
    elif input_component == "ML_One_DocLevel":
        dater = DataHelperMLNormal(doc_level="doc", train_holdout=0.80, embed_type="glove",
                                   embed_dim=300, target_sent_len=128, target_doc_len=128)
        ev = evaler_one.Evaluator()
    elif input_component == "PAN11_ONE":
        dater = DataHelperPan11(embed_type="glove", embed_dim=300, target_sent_len=100, prob_code=1)
        ev = evaler_pan.Evaluator()
    elif input_component == "PAN11_2CH":
        dater = DataHelperPan11(embed_type="both", embed_dim=300, target_sent_len=100, prob_code=0)
        ev = evaler_pan.Evaluator()
    else:
        raise NotImplementedError

    if middle_component == "ORIGIN_KIM":
        tt = ttl.TrainTask(data_helper=dater, am=am, input_component=input_component, exp_name=middle_component,
                           batch_size=64, evaluate_every=100, checkpoint_every=500, max_to_keep=8)
    else:
        tt = tr.TrainTask(data_helper=dater, am=am, input_component=input_component, exp_name=middle_component,
                          batch_size=64, evaluate_every=1000, checkpoint_every=2000, max_to_keep=6,
                          restore_path=None)
    start = timer()
    # n_fc variable controls how many fc layers you got at the end, n_conv does that for conv layers

    tt.training(filter_sizes=[[1, 2, 3, 4, 5]], num_filters=80, dropout_keep_prob=0.5, n_steps=15000, l2_lambda=0,
                dropout=True, batch_normalize=True, elu=True, fc=[128])
    end = timer()
    print((end - start))

    ev.load(dater)
    ev.evaluate(am.get_exp_dir(), None, doc_acc=True, do_is_training=True)
