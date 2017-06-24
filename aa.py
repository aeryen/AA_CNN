from timeit import default_timer as timer
# from datahelpers.data_helper_ml_mulmol6_OnTheFly import DataHelperMulMol6
from datahelpers.data_helper_ml_normal import DataHelperMLNormal
from datahelpers.data_helper_ml_2chan import DataHelperML_2CH
from trainer import TrainTaskLite as tr
from evaluators import eval_ml_mulmol_d as evaler
from evaluators import eval_ml_origin as evaler_one
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

    input_component = "ML_One"
    middle_component = "ORIGIN_KIM"
    truth_file = "labels.csv"

    am = ArchiveManager(input_component, middle_component, truth_file=truth_file)
    get_exp_logger(am)
    logging.warning('===================================================')

    if input_component == "ML_One":
        dater = DataHelperMLNormal(doc_level=LoadMethod.SENT, embed_type="glove",
                                   embed_dim=300, target_sent_len=50, target_doc_len=400, train_csv_file=truth_file,
                                   total_fold=5, t_fold_index=0)
        ev = evaler_one.Evaluator()
    elif input_component == "ML_2CH":
        dater = DataHelperML_2CH(doc_level="sent", num_fold=5, fold_index=0,
                                 embed_dim=300, target_sent_len=50, target_doc_len=400, truth_file=truth_file)
        ev = evaler_one.evaler()
    elif input_component == "ML_Six":
        dater = DataHelperMulMol6(doc_level="sent", num_fold=5, fold_index=4, embed_type="glove",
                                  embed_dim=300, target_sent_len=50, target_doc_len=400)
        ev = evaler.evaler()
    elif input_component == "ML_One_DocLevel":
        dater = DataHelperMLNormal(doc_level="doc", train_holdout=0.80, embed_type="glove",
                             embed_dim=300, target_sent_len=128, target_doc_len=128)
        ev = evaler_one.evaler()
    else:
        raise NotImplementedError

    tt = tr.TrainTask(data_helper=dater, am=am, input_component=input_component, exp_name=middle_component,
                      batch_size=64, evaluate_every=1000, checkpoint_every=5000)
    start = timer()
    # n_fc variable controls how many fc layers you got at the end, n_conv does that for conv layers

    tt.training(filter_sizes=[3, 4, 5], num_filters=100, dropout_keep_prob=0.8, n_steps=30000, l2_lambda=0.1,
                dropout=True, batch_normalize=False, elu=False, n_conv=1, fc=[])
    end = timer()
    print((end - start))

    ev.load(dater)
    ev.test(am.get_exp_dir(), None, doc_acc=True, do_is_training=False)
