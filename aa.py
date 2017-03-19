from timeit import default_timer as timer
from datahelpers import data_helper_ml_mulmol6file as dh6
from datahelpers import data_helper_ml_normal as dh
from trainer import TrainTask as tr

if __name__ == "__main__":

    input_component = "OneChannel"

    if input_component == "OneChannel":
        dater = dh.DataHelper(doc_level="sent", train_holdout=0.80, embed_type="w2v", embed_dim=300)
    elif input_component == "SixChannel":
        dater = dh6.DataHelperMulMol6(doc_level="sent", train_holdout=0.80)
    else:
        raise NotImplementedError

    ###############################################
    # exp_names you can choose from at this point:
    #
    # Input Components:
    #
    # *OneChannel
    # *SixChannel
    #
    # Middle Components:
    #
    # *NParallelConvOnePoolNFC
    # *YifanConv
    # *ParallelJoinedConv
    # *NCrossSizeParallelConvNFC
    ################################################
    tt = tr.TrainTask(data_helper=dater, input_component=input_component, exp_name="NCrossSizeParallelConvNFC",
                      filter_sizes=[[3, 4, 5], [3, 4, 5]], batch_size=20, dataset="ML")
    start = timer()
    # n_fc variable controls how many fc layers you got at the end, n_conv does that for conv layers
    tt.training(num_filters=100, dropout_keep_prob=0.7, n_steps=100000, l2_lambda=0.1, dropout=True,
                batch_normalize=False, elu=False, n_conv=2, fc=[])
    end = timer()
    print(end - start)
