import numpy as np
from sklearn import linear_model
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier

load_path = "E:\\Research\\Paper 03\\AA_CNN_github\\runs\\ML_One_ORIGIN_NEW\\170613_1497377078_labels.csv\\"

train_dist = np.loadtxt(load_path + "train_dist.txt")
train_label = np.loadtxt(load_path + "train_label.txt")
test_dist = np.loadtxt(load_path + "test_dist.txt")
test_label = np.loadtxt(load_path + "test_label.txt")

svm_results = []
lr_results = []
knc_results = []
for i in range(20):
    clf = svm.SVC()
    clf.fit(train_dist, train_label[:, i])
    r = clf.predict(test_dist)
    svm_results.append(r)

    lr = linear_model.LogisticRegression()
    lr.fit(train_dist, train_label[:, i])
    r = lr.predict(test_dist)
    lr_results.append(r)

    knc = KNeighborsClassifier()
    knc.fit(train_dist, train_label[:, i])
    t = knc.predict(test_dist)
    knc_results.append(r)

np.concatenate(svm_results, axis=1)

