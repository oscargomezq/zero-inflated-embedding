import numpy as np
from scipy import sparse 
import pickle

'''
Generate a random copy of data

input:
output: dict with two fields:
             trainset: dict with two fields  
                            scores: a sparse matrix, each ij entry is the rating of movie j given by person i, or the count of item j in basket i
                            atts  : a matrix, each row is a feature vector extracted from person i, or basket i
                            		it is not a sparse matrix
             testset : [same structure as trainset]

'''

def rand_data(prt=False):
    n_rows = 200
    n_columns = 50
    n_feat = 5
     
    np.random.seed(27)
    
    # allocate more rows than necessary, to make sure each row has at least 2 non-zero entries
    score_mat = np.random.rand(n_rows * 2, n_columns)
    
    # resembles a zero inflated situation with few non-zero entries
    score_mat[score_mat < 0.88] = 0
    score_mat[np.logical_and(0.96 <= score_mat, score_mat < 1)] = 3
    score_mat[np.logical_and(0.92 <= score_mat, score_mat < 0.96)] = 2
    score_mat[np.logical_and(0.88 <= score_mat, score_mat < 0.92)] = 1
    
    # remove rows with only one non-zero entry and keep only n_rows of them
    row_sum = np.sum(score_mat > 0, axis=1)
    score_mat = score_mat[row_sum >= 2, ]
    score_mat = score_mat[0 : n_rows, ]
    
    feature = np.random.rand(n_rows, n_feat)
    
    # split into two halves, one for train and one for test
    trainset = dict(scores=sparse.csr_matrix(score_mat[0:(n_rows // 2)]), atts=feature[0:(n_rows // 2)]) 
    testset = dict(scores=sparse.csr_matrix(score_mat[(n_rows // 2):]), atts=feature[(n_rows // 2):]) 

    return dict(trainset=trainset, testset=testset)


if __name__ == '__main__':

    rand_data(prt=True)
