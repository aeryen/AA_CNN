# THIS FILE LOADS THE DATA LOG PRINTED BY OTHER TRAINING SCRIPT AND PRINT ONLY IMPORTANT INFO

infile = open("probI_longTrain_300d_ba_trueS.txt", "r")
lines = infile.readlines()
sent_acc = []
doc_acc = []
for l in lines:
    if "=====" in l:
        if sent_acc and doc_acc:
            # print sent_acc
            print('\t'.join(map(str, sent_acc)))
            # print doc_acc
            print('\t'.join(map(str, doc_acc)))

        sent_acc = []
        doc_acc = []
        print(l)
    if "ACC" in l and "Doc" not in l:
        sent_acc.append(l.split()[1])
    if "Doc ACC" in l and "norm" not in l:
        doc_acc.append(l.split()[2])
print('\t'.join(map(str, sent_acc)))
print('\t'.join(map(str, doc_acc)))
