import csv
import numpy
from sklearn import svm
data_tr = numpy.loadtxt(open("SG_small_train/SG_small.train","rb"),delimiter=" ",skiprows=1).astype('float') #load training file
data_tr_size = data_tr.shape
Y_tr = data_tr[:,3].astype('int') #assuming true type is classifying attribute
X1_tr = data_tr[:, [1,2]] #rest are training attributes
X2_tr = data_tr[:, 9:]
X_tr = numpy.hstack((X1_tr,X2_tr))
lin_clf = svm.LinearSVC(penalty='l2', loss='l2', dual=True, tol=0.0001, C=630.1, multi_class='ovr', fit_intercept=True,
intercept_scaling=1, class_weight=None, verbose=0, random_state=None) #simple linear one-vs-rest SVM implementation
lin_clf.fit(X_tr, Y_tr)
data_tst = numpy.loadtxt(open("SG_small_test/SG_small.test","rb"),delimiter=" ",skiprows=1).astype('float') #load testing data
data_tst_size = data_tst.shape
Y_tst = data_tst[:,3].astype('int')
X1_tst = data_tst[:, [1,2]]
X2_tst = data_tst[:, 9:]
X_tst = numpy.hstack((X1_tst,X2_tst))
dec = lin_clf.predict(X_tst) #SVM class prediction
corr = 0
wrong = 0
for i in range(1, data_tst_size[0]):
	if Y_tst[i] == dec[i]:
		corr = corr + 1
	else:
		wrong = wrong + 1
accuracy = corr / float (corr+wrong)
print('correct predictions = ' + repr(corr))
print('correct predictions = ' + repr(wrong))
print('accuracy = ' + repr(accuracy)) # print accuracy
