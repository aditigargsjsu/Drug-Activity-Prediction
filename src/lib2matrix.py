import sys
sys.path.append('/Users/aditi/anaconda/lib/python2.7/site-packages')
import numpy as np
import pandas as pd
import scipy.sparse as sp
from numpy.linalg import norm
from collections import Counter, defaultdict
from scipy.sparse import csr_matrix
from sklearn.metrics import (brier_score_loss, precision_score, recall_score,f1_score)
from sklearn.neighbors import LSHForest
from lsh import clsh, jlsh, generateSamples, findNeighborsBrute, recall
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import TruncatedSVD

def build_matrix(docs):
    r""" Build sparse matrix from a list of documents, 
    each of which is a list of word/terms in the document.  
    """
    nrows = len(docs)
    idx = {}
    tid = 0
    nnz = 0
    for d in docs:
        nnz += len(set(d))
        for w in d:
            if w not in idx:
                idx[w] = tid
                tid += 1
    ncols = len(idx)
        
    # set up memory
    ind = np.zeros(nnz, dtype=np.int)
    val = np.zeros(nnz, dtype=np.double)
    ptr = np.zeros(nrows+1, dtype=np.int)
    i = 0  # document ID / row counter
    n = 0  # non-zero counter
    # transfer values
    for d in docs:
        cnt = Counter(d)
        keys = list(k for k,_ in cnt.most_common())
        l = len(keys)
        for j,k in enumerate(keys):
            ind[j+n] = idx[k]
            val[j+n] = cnt[k]
        ptr[i+1] = ptr[i] + l
        n += l
        i += 1
            
    mat = csr_matrix((val, ind, ptr), shape=(nrows, ncols), dtype=np.double)
    mat.sort_indices()
    
    return mat

def csr_info(mat, name="", non_empy=False):
    r""" Print out info about this CSR matrix. If non_empy, 
    report number of non-empty rows and cols as well
    """
    if non_empy:
        print("%s [nrows %d (%d non-empty), ncols %d (%d non-empty), nnz %d]" % (
                name, mat.shape[0], 
                sum(1 if mat.indptr[i+1] > mat.indptr[i] else 0 
                for i in range(mat.shape[0])), 
                mat.shape[1], len(np.unique(mat.indices)), 
                len(mat.data)))
    else:
        print( "%s [nrows %d, ncols %d, nnz %d]" % (name, 
                mat.shape[0], mat.shape[1], len(mat.data)) )

def csr_l2normalize(mat, copy=False, **kargs):
    r""" Normalize the rows of a CSR matrix by their L-2 norm. 
    If copy is True, returns a copy of the normalized matrix.
    """
    if copy is True:
        mat = mat.copy()
    nrows = mat.shape[0]
    nnz = mat.nnz
    ind, val, ptr = mat.indices, mat.data, mat.indptr
    # normalize
    for i in range(nrows):
        rsum = 0.0    
        for j in range(ptr[i], ptr[i+1]):
            rsum += val[j]**2
        if rsum == 0.0:
            continue  # do not normalize empty rows
        rsum = 1.0/np.sqrt(rsum)
        for j in range(ptr[i], ptr[i+1]):
            val[j] *= rsum
            
    if copy is True:
        return mat

def splitData(mat, cls, fold=1, d=10):
    r""" Split the matrix and class info into train and test data using d-fold hold-out
    """
    n = mat.shape[0]
    r = int(np.ceil(n*1.0/d))
    mattr = []
    clstr = []
    for f in range(d):
        if f+1 != fold:
            mattr.append( mat[f*r: min((f+1)*r, n)] )
            clstr.extend( cls[f*r: min((f+1)*r, n)] )
    train = sp.vstack(mattr, format='csr')    
    test = mat[(fold-1)*r: min(fold*r, n), :]
    clste = cls[(fold-1)*r: min(fold*r, n)]

    return train, clstr, test, clste

####def classifyNames(names, cls, c=3, k=3, d=10):
####    r""" Classify names using c-mer frequency vector representations of the names and kNN classification with 
####    cosine similarity and 10-fold cross validation
####    """
####    docs = names
####    mat = build_matrix(docs)
####    # since we're using cosine similarity, normalize the vectors
####    csr_l2normalize(mat)
####    
####    def classify(x, train, clstr):
####        r""" Classify vector x using kNN and majority vote rule given training data and associated classes
####        """
####        # find nearest neighbors for x
####        dots = x.dot(train.T)
####        sims = list(zip(dots.indices, dots.data))
####        if len(sims) == 0:
####            # could not find any neighbors
####            return '+' if np.random.rand() > 0.5 else '-'
####        sims.sort(key=lambda x: x[1], reverse=True)
####        tc = Counter(clstr[s[0]] for s in sims[:k]).most_common(2)
####        if len(tc) < 2 or tc[0][1] > tc[1][1]:
####            # majority vote
####            return tc[0][0]
####        # tie break
####        tc = defaultdict(float)
####        for s in sims[:k]:
####            tc[clstr[s[0]]] += s[1]
####        return sorted(tc.items(), key=lambda x: x[1], reverse=True)[0][0]
####        
####    macc = 0.0
####    for f in range(d):
####        # split data into training and testing
####        train, clstr, test, clste = splitData(mat, cls, f+1, d)
####        # predict the class of each test sample
####        clspr = [ classify(test[i,:], train, clstr) for i in range(test.shape[0]) ]
####        # compute the accuracy of the prediction
####        acc = 0.0
####        for i in range(len(clste)):
####            if clste[i] == clspr[i]:
####                acc += 1
####        acc /= len(clste)
####        macc += acc
####        
####    return macc/d
####

def classify_using_cosine(x, mat, cls, k):
    dots = x.dot(mat.T)
    sims = list(zip(dots.indices, dots.data))
    if len(sims) == 0:
        return '1' if np.random.rand() > 0.5 else '0'
    sims.sort(key=lambda x: x[1], reverse=True)
    tc = Counter(cls[s[0]] for s in sims[:k]).most_common(2)
    if len(tc) < 2 or tc[0][1] > tc[1][1]:
        return tc[0][0]
    tc = defaultdict(float)
    for s in sims[:k]:
        tc[cls[s[0]]] += s[1]
    return sorted(tc.items(), key=lambda x: x[1], reverse=True)[0][0]
    
def classify_using_nbrs_after_svd_ball_tree(testarray, trainarray, cls, k=3):
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(trainarray)
    indices = nbrs.kneighbors(testarray, n_neighbors = k, return_distance=False)
    classified_output=[]
    for j in range(len(indices)): 
        c = Counter()
        for i in range (len(indices[0])): 
            c[cls[indices[j][i]]] += 1
        tc = c.most_common(2)
        #print tc
        classified_output.append(tc[0][0])
    return classified_output
    

def classify_using_clsh(testarray, trainarray, cls, k=7):
    #L17_3 = clsh(trainarray, ntables=18, nfunctions=3)
    nbrsExact = findNeighborsBrute(trainarray, testarray, k=k, sim="cos")
    classified_output=[]
    for j in range(len(nbrsExact)): 
        c = Counter()
        for i in range (len(nbrsExact[0])): 
            c[cls[nbrsExact[j][i]]] += 1
        tc = c.most_common(2)
        classified_output.append(tc[0][0])
    return classified_output

def classify_using_jlsh(testarray, trainarray, cls, k=7):
    #L17_3 = clsh(trainarray, ntables=18, nfunctions=3)
    nbrsExact = findNeighborsBrute(trainarray, testarray, k=k, sim="jac")
    classified_output=[]
    for j in range(len(nbrsExact)): 
        c = Counter()
        for i in range (len(nbrsExact[0])): 
            c[cls[nbrsExact[j][i]]] += 1
        tc = c.most_common(2)
        classified_output.append(tc[0][0])
    return classified_output
