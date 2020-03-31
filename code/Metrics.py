
from scipy.special import comb
from math import log
import numpy as np

# Variation of information
# Expects list of sets
def voi(X, Y):
    n = float(sum([len(x) for x in X]))
    assert n == float(sum([len(y) for y in Y]))
    ret = 0.0
    
    for x in X:
        for y in Y:
            p = len(x)/n
            q = len(y)/n
            r = len(set(x)&set(y))/n
            if r > 0:
                ret += r*(log(r/p, 2)+log(r/q, 2))
    return -1*ret

# Rand Index
# Expects labeled 1D array

def ri(X, Y):
    tp_plus_fp = comb(np.bincount(X), 2).sum()
    tp_plus_fn = comb(np.bincount(Y), 2).sum()
    A = np.c_[(X, Y)]
    tp = sum(comb(np.bincount(A[A[:, 0] == i, 1]), 2).sum()
             for i in set(X))
    fp = tp_plus_fp - tp
    fn = tp_plus_fn - tp
    tn = comb(len(A), 2) - tp - fp - fn
    
    return (tp + tn) / (tp + fp + fn + tn)