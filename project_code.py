#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: sakshamjain
"""

import pandas as pd
import helper_functions as hlp
import sklearn.svm as sv

@hlp.timeit
def fitSVM(data):
    '''
        Build the SVM classifier
    '''
    # create the classifier object
    svm = sv.SVC(kernel='linear', C=20.0)

    # fit the data
    return svm.fit(data[0],data[1])


# names of file to read from
r_filenameCSV = 'winequality-red.csv'


# read the data
csv_read = pd.read_csv(r_filenameCSV,sep=';')

# print the first 10 records
print(csv_read.head(10))
# print the last 10 records
print(csv_read.tail(10))

#splitting data into test and train
train_x, train_y, test_x,  test_y, labels = hlp.split_data(csv_read, y='quality')

# train the model
classifier = fitSVM((train_x, train_y))

# classify the unseen data
predicted = classifier.predict(test_x)

# print out the results
hlp.printModelSummary(test_y, predicted)

print(classifier.support_vectors_)

