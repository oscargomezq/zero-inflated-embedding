from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pickle
from zie import fit_emb
from zie import evaluate_emb
from random_data import rand_data


def experiment(prt=False):

    ## Step 1: load data
    print('\nGenerating a dataset ...')

    data = rand_data()

    trainset = data['trainset']
    testset = data['testset']

    """
        trainset: dict with two fields  
                      scores: a sparse matrix, each ij entry is the rating of movie j given by person i, or the count of item j in basket i
                      atts  : a matrix, each row is a feature vector extracted from person i, or basket i
        testset:  [same structure as trainset]
    """


    # one can always redefine zie.generate_batch(reviews, rind) to use other format of trainset and testset 

    print('The training set has %d rows and %d columns, and the test set has %d rows' % 
                             (trainset['scores'].shape[0], trainset['scores'].shape[1], testset['scores'].shape[0]))
    if prt: print("Train scores:", trainset['scores'].shape, "Train covariates:", trainset['atts'].shape)
    if prt: print("Test scores:", testset['scores'].shape, "Test covariates:", testset['atts'].shape)
    if prt: print()
    
    ## Step 2: configurate model 
    
    """
        K              in N^+, dimensionality of embedding
        exposure       in {True, False}
        use_covariates in {True, False}
        dist           in {poisson, binomial}. Binomial has N=3
        max_iter       in N^+, max training iteration
        ar_sigma2      in R^+, sigma^2 value for Gaussian prior of alpha and rho vectors
        w_sigma2       in R^+, sigma^2 value for Gaussian prior of covariate coefficients
    """
        
    

    # in this example of random data, we use 10k iterations. 
    # in real application, it often needs more than 100k iterations to converge. So please check the validation log-likelihood
    # the model takes out 1/10 of the training data to show validation log-likelihood. 
    # In the printout lines, the three values are training log-likelihood, model objective (neg llh + regularizer), validation log-likelihood 
    config = dict(K=16, exposure=True, use_covariates=True, dist='binomial',
                  max_iter=10000, ar_sigma2=1, w_sigma2=1)

    print('Setting model configurations: use exposure component with covariates')
    if prt: print("configuration params:", config)
    if prt: print()


    ## Step 3: Fit a Zero-Inflated Embedding model

    print('Fitting the ZIE model ...')
    emb_model, logg = fit_emb(trainset, config)
    print('Training done!')
    if prt: print()
    
    # Step 4: Test the model 
    print('Evaluating the model on test set ...')
    llh_array, pos_llh_array = evaluate_emb(testset, emb_model, config)
    print("The mean heldout log-likelihood over all entries and that over positive entries are ", 
          np.mean(llh_array), ' and ', np.mean(pos_llh_array))
    if prt: print()

    # Step 5: Dump out embedding vectors
    print('Saving embedding vectors ...') 
    pickle.dump(emb_model['alpha'], open('embedding_vectors.pkl', 'wb'))
    if prt: print()

    print('Embedding vectors')
    print(emb_model['alpha'].shape)
    if prt: print()
    
     

if __name__ == '__main__': 

    experiment(prt=True)
