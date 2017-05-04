import sys
sys.path.append('/Users/aditi/anaconda/lib/python2.7/site-packages')

import numpy as np
import pandas as pd
import random
from collections import Counter, defaultdict
from scipy.sparse import csr_matrix
from lib2matrix import build_matrix, csr_info,csr_l2normalize, splitData, classify_using_cosine, classify_using_nbrs_after_svd_ball_tree
from sklearn.metrics import (brier_score_loss, precision_score, recall_score,f1_score)
from lsh import clsh, jlsh, generateSamples, findNeighborsBrute, recall
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD

##from sklearn.tree import DecisionTreeClassifier
##from reshape import reshape
##from sklearn.naive_bayes import GaussianNB
##import scipy.sparse as sp
##from numpy.linalg import norm
##from imblearn.over_sampling import SMOTE, ADASYN
##from sklearn.neighbors import LSHForest
##from sklearn.decomposition import SparsePCA as sparsePCA
##from sklearn.decomposition import PCA as sklearnPCA


# Import the train data set.
df = pd.read_csv(
    filepath_or_buffer='/Users/aditi/Downloads/train.dat', 
    header=None, 
    sep=',')
vals = df.ix[:,:].values
train_list = [n[0][2:] for n in vals] # train_list is a list of lists
cls = [n[0][0] for n in vals] # cls is a list of lists


# Train data set imported. Pre-processing the data by changing the 1 string column into a list of numbers
for i in range (len(train_list)):
    train_list[i]=train_list[i].rstrip()
    train_list[i]=train_list[i].split(' ')
    for j in range(len(train_list[i])):
        train_list[i][j]=int(train_list[i][j])

# Now need to take care of imbalance of data.
# Random under sampling of majority class
majority=[]
minority=[]
for i, j in enumerate(cls):
    if j=='0':
        majority.append(i)
    elif j=='1':
        minority.append(i)

num_to_select = int(len(majority)*0.5) # taking 50% of majority class
random_items_from_majority = random.sample(majority, num_to_select)
new_train_list_index = random_items_from_majority + minority
new_train_list_index.sort()
new_train_list = []
new_cls = []
for i in new_train_list_index:
    new_train_list.append(train_list[i])
    new_cls.append(cls[i])

# achieved 50% random under sampling of the majority class

# Now trying to do random over sampling of the minority class
majority=[]
minority=[]
for i, j in enumerate(new_cls):
    if j=='0':
        majority.append(i)
    elif j=='1':
        minority.append(i)
num_to_select = int(len(minority)*0.3) # taking 30% of minority class
random_items_from_minority = random.sample(minority, num_to_select)
new_train_list_index2 = majority+minority+random_items_from_minority
new_train_list_index2.sort()
new_train_list2 = []
new_cls2 = []
for i in new_train_list_index2:
    new_train_list2.append(new_train_list[i])
    new_cls2.append(new_cls[i])

final_cls = new_cls2
final_train_list = new_train_list2
# Achieved random under sampling and random over sampling to take care of data imbalance in the training data

# Import the test data set and Pre-processing the data by changing the 1 string column into a list of numbers

df = pd.read_csv(
    filepath_or_buffer='/Users/aditi/Downloads/test.dat', 
    header=None, 
    sep=',')
vals = df.ix[:,:].values
test_list = [n[0][:] for n in vals]

for i in range (len(test_list)):
    test_list[i]=test_list[i].rstrip()
    test_list[i]=test_list[i].split(' ')
    for j in range (len(test_list[i])):
        test_list[i][j]=int(test_list[i][j])


total_list = final_train_list + test_list
total_matrix = build_matrix(total_list)
csr_l2normalize(total_matrix)

train_mat = total_matrix[0:len(final_train_list)]
test_mat = total_matrix[len(final_train_list):len(total_list)]


# train, clstr, test, clste = splitData(train_mat, final_cls, fold=1, d=5)


# Apply SVD for feature reduction. Using 1000 features as feature reduction.


svd = TruncatedSVD(n_components=1000)
train_transformed = svd.fit_transform(train_mat)
print 'Percentage of variance explained by each of the selected components in total matrix is'
print (svd.explained_variance_ratio_.sum())
test_transformed = svd.transform(test_mat)


# For cross validation, splitting train data in 80% and 20%
train = train_transformed [0:(int(len(train_transformed)*0.8))]
clstr = final_cls [0:(int(len(train_transformed)*0.8))]
test = train_transformed [(int(len(train_transformed)*0.8)):(len(train_transformed))]
clste = final_cls [(int(len(train_transformed)*0.8)):(len(train_transformed))]

# Test different combinations to see what combination has higher f1 score for number of neighbors

find_best_k={}
for k in [5,7,9,11,13]: # Checking which k number of neighbors gives best f1 score to use for actual test data
    clspr = classify_using_nbrs_after_svd_ball_tree(test, train, clstr, k=k)
    print clspr
    f1=f1_score(clste,clspr,average='weighted')
    print 'f1 score for %d neighbors is %f' %(k, f1)
    find_best_k[k]=f1
    
for k, f1 in find_best_k.items():
    if f1 == max(find_best_k.values()):
        print 'best k is %d' %(k)
        k = k
        break

# Now best neighbors is selected.

# Now need to classify actual testing data
prediction = classify_using_nbrs_after_svd_ball_tree(test_transformed, train_transformed, final_cls, k=k)
fo=open('/Users/aditi/Downloads/trying_prediction_with_nbrs_and_svd_ball_tree.dat','a+')
for i in range (len(prediction)):
    fo.write(prediction[i]+'\n')
fo.close()
