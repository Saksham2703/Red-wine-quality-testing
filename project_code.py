#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: sakshamjain
"""

import pandas as pd
import helper_functions as hlp


# names of file to read from
r_filenameCSV = 'winequality-red.csv'

# read the data
csv_read = pd.read_csv(r_filenameCSV,sep=';')

# print the first 10 records
print(csv_read.head(10))
# print the last 10 records
print(csv_read.tail(10))

#splitting data in to test and train
train_x, train_y, test_x,  test_y, labels = hlp.split_data(csv_read, y = 'quality')



