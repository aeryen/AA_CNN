import sys

import gensim
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import *
from sklearn.model_selection import KFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC

from . import load

classifiers = [OneVsRestClassifier(LinearSVC()), OneVsRestClassifier(MLPClassifier(hidden_layer_sizes = (100, 100)))]
clf_names = ["OneVsRestClassifier(LinearSVC())", "OneVsRestClassifier(MLPClassifier(hidden_layer_sizes = (100, 100)))"]
#classifiers = [OneVsRestClassifier(GradientBoostingClassifier()),OneVsRestClassifier(GaussianNB()), OneVsRestClassifier(SVC(kernel='poly')), OneVsRestClassifier(MLPClassifier(hidden_layer_sizes = (100, 100))),  OneVsRestClassifier(DecisionTreeClassifier()), OneVsRestClassifier(AdaBoostClassifier()), OneVsRestClassifier(KNeighborsClassifier()),OneVsRestClassifier(SGDClassifier(loss='hinge', penalty='l2')), OneVsRestClassifier(SGDClassifier(loss='log', penalty='l2')),OneVsRestClassifier(SGDClassifier(loss='modified_huber', penalty='l2')),OneVsRestClassifier(SGDClassifier(loss='squared_hinge', penalty='l2')),OneVsRestClassifier(SGDClassifier(loss='perceptron', penalty='l2')),OneVsRestClassifier(SGDClassifier(loss='squared_loss', penalty='l2')),OneVsRestClassifier(SGDClassifier(loss='huber', penalty='l2')),OneVsRestClassifier(SGDClassifier(loss='epsilon_insensitive', penalty='l2')),OneVsRestClassifier(SGDClassifier(loss='squared_epsilon_insensitive', penalty='l2')),OneVsRestClassifier(SGDClassifier(loss='hinge', penalty='l1')), OneVsRestClassifier(SGDClassifier(loss='log', penalty='l1')),OneVsRestClassifier(SGDClassifier(loss='modified_huber', penalty='l1')),OneVsRestClassifier(SGDClassifier(loss='squared_hinge', penalty='l1')),OneVsRestClassifier(SGDClassifier(loss='perceptron', penalty='l1')),OneVsRestClassifier(SGDClassifier(loss='squared_loss', penalty='l1')),OneVsRestClassifier(SGDClassifier(loss='huber', penalty='l1')),OneVsRestClassifier(SGDClassifier(loss='epsilon_insensitive', penalty='l1')),OneVsRestClassifier(SGDClassifier(loss='squared_epsilon_insensitive', penalty='l1')),OneVsRestClassifier(SGDClassifier(loss='hinge', penalty='elasticnet')), OneVsRestClassifier(SGDClassifier(loss='log', penalty='elasticnet')),OneVsRestClassifier(SGDClassifier(loss='modified_huber', penalty='elasticnet')),OneVsRestClassifier(SGDClassifier(loss='squared_hinge', penalty='elasticnet')),OneVsRestClassifier(SGDClassifier(loss='perceptron', penalty='elasticnet')),OneVsRestClassifier(SGDClassifier(loss='squared_loss', penalty='elasticnet')),OneVsRestClassifier(SGDClassifier(loss='huber', penalty='elasticnet')),OneVsRestClassifier(SGDClassifier(loss='epsilon_insensitive', penalty='elasticnet')),OneVsRestClassifier(SGDClassifier(loss='squared_epsilon_insensitive', penalty='elasticnet'))]
#clf_names = ["GradientBoostingClassifier", "GaussianNB()", "SVC(kernel='poly')", "MLPClassifier(hidden_layers = (100, 100)", "DecisionTreeClassifier()", "AdaBoostClassifier()", "KNeighborsClassifier()","SGDClassifier(loss='hinge', penalty='l2')", "SGDClassifier(loss='log', penalty='l2')","SGDClassifier(loss='modified_huber', penalty='l2')","SGDClassifier(loss='squared_hinge', penalty='l2')","SGDClassifier(loss='perceptron', penalty='l2')","SGDClassifier(loss='squared_loss', penalty='l2')","SGDClassifier(loss='huber', penalty='l2')","SGDClassifier(loss='epsilon_insensitive', penalty='l2')","SGDClassifier(loss='squared_epsilon_insensitive', penalty='l2')","SGDClassifier(loss='hinge', penalty='l1')", "SGDClassifier(loss='log', penalty='l1')","SGDClassifier(loss='modified_huber', penalty='l1')","SGDClassifier(loss='squared_hinge', penalty='l1')","SGDClassifier(loss='perceptron', penalty='l1')","SGDClassifier(loss='squared_loss', penalty='l1')","SGDClassifier(loss='huber', penalty='l1')","SGDClassifier(loss='epsilon_insensitive', penalty='l1')","SGDClassifier(loss='squared_epsilon_insensitive', penalty='l1')","SGDClassifier(loss='hinge', penalty='elasticnet')", "SGDClassifier(loss='log', penalty='elasticnet')","SGDClassifier(loss='modified_huber', penalty='elasticnet')","SGDClassifier(loss='squared_hinge', penalty='elasticnet')","SGDClassifier(loss='perceptron', penalty='elasticnet')","SGDClassifier(loss='squared_loss', penalty='elasticnet')","SGDClassifier(loss='huber', penalty='elasticnet')","SGDClassifier(loss='epsilon_insensitive', penalty='elasticnet')","SGDClassifier(loss='squared_epsilon_insensitive', penalty='elasticnet')"]
#clf_names =  ['OneVsRestClassifier(LinearSVC())', 'OneVsRestClassifier(SVC())', 'OneVsRestClassifier(RandomForestClassifier())']
assert(len(classifiers) == len(clf_names))

def build_model():
    documents, y = load.get_doc('papers_dataset', sys.argv[1])
    model = gensim.models.Doc2Vec(documents, min_count = 1)
    # build the model
    #model = gensim.models.Doc2Vec(documents, dm = 0, alpha=0.025, size= 20, min_alpha=0.025, min_count=0)
 
    # start training
    #for epoch in range(1):
    #    print ('Now training epoch %s'%epoch)
    #    model.train(documents)
    #    model.alpha -= 0.002  # decrease the learning rate
    #    model.min_alpha = model.alpha  # fix the learning rate, no decay
    return model.docvecs.doctag_syn0, y

def benchmark(clf):
    X, y = build_model()
    skf = KFold(n_splits=5, shuffle=True)
    jaccard = []
    accuracy = []
    precision_micro = []
    precision_macro = []
    precision_weighted = []
    precision_samples = []
    recall_micro = []
    recall_macro = []
    recall_weighted = []
    recall_samples = []
    f1_micro = []
    f1_macro = []
    f1_weighted = []
    f1_samples = []
    hamming = []

    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        jaccard.append(jaccard_similarity_score(y_test, y_pred))
        accuracy.append(accuracy_score(y_test, y_pred))
        precision_micro.append(precision_score(y_test, y_pred, average='micro'))
        precision_macro.append(precision_score(y_test, y_pred, average='macro'))
        precision_weighted.append(precision_score(y_test, y_pred, average='weighted'))
        precision_samples.append(precision_score(y_test, y_pred, average='samples'))
        recall_micro.append(recall_score(y_test, y_pred, average='micro'))
        recall_macro.append(recall_score(y_test, y_pred, average='macro'))
        recall_weighted.append(recall_score(y_test, y_pred, average='weighted'))
        recall_samples.append(recall_score(y_test, y_pred, average='samples'))
        f1_micro.append(f1_score(y_test, y_pred, average='micro'))
        f1_macro.append(f1_score(y_test, y_pred, average='macro'))
        f1_weighted.append(f1_score(y_test, y_pred, average='weighted'))
        f1_samples.append(f1_score(y_test, y_pred, average='samples'))
        hamming.append(hamming_loss(y_test, y_pred))

    #scores = cross_val_score(clf, X, y, cv=5)

    return [np.array(jaccard).mean(), np.array(accuracy).mean(), np.array(precision_micro).mean(), np.array(precision_macro).mean(), np.array(precision_weighted).mean(), np.array(precision_samples).mean(), np.array(recall_micro).mean(), np.array(recall_macro).mean(),np.array(recall_weighted).mean(),np.array(recall_samples).mean(),np.array(f1_micro).mean(), np.array(f1_macro).mean(),np.array(f1_weighted).mean(),np.array(f1_samples).mean(), np.array(hamming).mean()]

def test():
    with open(sys.argv[2], "w+") as fout:
        fout.write("classifier| vectorizer | jaccard_score | accuracy | precision_micro | precision_macro | precision_weighted | precision_samples | recall_micro | recall_macro | recall_weighted | recall_samples | f1_micro | f1_macro | f1_weighted | f1_sample | hamming_loss\n")

        j = 0
        for clf in classifiers:
            results = benchmark(clf)
            output = clf_names[j] + "|doc2vec|"
            for i in range(len(results)):
                output += str(results[i])
                if i != len(results) - 1:
                    output += "|"

            fout.write(output + "\n")
            print((clf_names[j]))
            classifiers[j] = None   
            j = j + 1

def cluster():
    with open('papers_dataset/labels.csv', 'r') as fin:
        line = fin.readline()
    tokens = line.strip().split(',')
    authors = tokens[1:]
    
    X, y = build_model()
    nrows, ncols = y.shape
    assert(len(authors) == ncols)
    text_authors = []
    for i in range(nrows):
        paper_authors = []
        for j in range(ncols):
            if y[i, j] == 1:
                paper_authors.append(authors[j])
        assert(len(paper_authors) > 0)
        text_authors.append(paper_authors)
    km = KMeans(n_clusters = 10).fit(X)
    labels = km.predict(X)
    clustered_authors = []
    for i in range(len(labels)):
        clustered_authors.append((labels[i],text_authors[i]))
    
    clusters = sorted(clustered_authors, key=lambda x: x[0])
    print(clusters)

#cluster()
test()
# shows the similar words
#print (model.most_similar('suppli'))
 
# shows the learnt embedding
#print (model['suppli'])
 
# shows the similar docs with id = 2
#print (model.docvecs.most_similar(str(2)))
