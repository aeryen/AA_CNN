from timeit import default_timer as timer
from datahelpers.data_helper_ml_mulmol6_Read import DataHelperMulMol6
from datahelpers.data_helper_ml_normal import DataHelperML
from trainer import TrainTask as tr
from evaluators import eval_ml_mulmol_d as evaler
from evaluators import eval_ml_origin as evaler_one
from utils.ArchiveManager import ArchiveManager
import logging


if __name__ == "__main__":
    input_component = "ML_Six"
    middle_component = "InceptionLike"

    am = ArchiveManager(input_component, middle_component)
    logging.warning('===================================================')

    if input_component == "ML_One":
        dater = DataHelperML(doc_level="sent", train_holdout=0.80, embed_type="glove", embed_dim=300)
        ev = evaler_one.evaler()
    elif input_component == "ML_Six":
        dater = DataHelperMulMol6(doc_level="sent", train_holdout=0.80, target_sent_len=50)
        ev = evaler.evaler()
    elif input_component == "ML_One_DocLevel":
        dater = DataHelperML(doc_level="doc", train_holdout=0.80, embed_type="glove", embed_dim=300,
                             target_doc_len=128, target_sent_len=128)
        ev = evaler_one.evaler()
    else:
        raise NotImplementedError

    ###############################################
    # exp_names you can choose from at this point:
    #
    # Input Components:
    #
    # * OneChannel
    # * SixChannel
    # * OneChannel_DocLevel
    #
    # Middle Components:
    #
    # * NParallelConvOnePoolNFC
    # * NConvDocConvNFC
    # * ParallelJoinedConv
    # * NCrossSizeParallelConvNFC
    # * InceptionLike
    ################################################
    tt = tr.TrainTask(data_helper=dater, input_component=input_component, exp_name=middle_component,
                      batch_size=8, evaluate_every=5, checkpoint_every=5)
    start = timer()
    # n_fc variable controls how many fc layers you got at the end, n_conv does that for conv layers

    ts = tt.training(filter_sizes=[[3, 4, 5], [3, 4, 5]], num_filters=64, dropout_keep_prob=1.0, n_steps=30000, l2_lambda=0.1,
                     dropout=False,
                     batch_normalize=False, elu=True, n_conv=2, fc=[1024])
    end = timer()
    print(end - start)

    ev.load(dater)
    with open(tt.exp_name + ".txt", mode="aw") as of:
        checkpoint_dir = tt.exp_dir + str(ts) + "/checkpoints/"
        ev.test(checkpoint_dir, None, of, documentAcc=True)
