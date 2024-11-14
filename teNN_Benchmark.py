import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as evm
from sklearn.base import BaseEstimator, ClassifierMixin
from ctypes import c_void_p, c_double, c_int, cdll
from numpy.ctypeslib import ndpointer
import sys, os, random, time
import argparse

import utils as ut
import teNNClassifierCEL as tenn 

PATH="./data/"

def ranged_type(value_type, min_value, max_value):
    """
    Return function handle of an argument type function for ArgumentParser checking a range:
        min_value <= arg <= max_value
    Parameters
    ----------
    value_type  - value-type to convert arg to
    min_value   - minimum acceptable argument
    max_value   - maximum acceptable argument
    Returns
    -------
    function handle of an argument type function for ArgumentParser
    Usage
    -----
        ranged_type(float, 0.0, 1.0)
    """
    def range_checker(arg: str):
        try:
            f = value_type(arg)
        except ValueError:
            raise argparse.ArgumentTypeError(f'must be a valid {value_type}')
        if f < min_value or f > max_value:
            raise argparse.ArgumentTypeError(f'must be within [{min_value}, {max_value}]')
        return f
    # Return function handle to checking function
    return range_checker
               

def teNN(Xtrain, ytrain, Xtest, ytest, args): 
    '''
    build a teNN classifier, train it on the Train data and evaluate it on the Test data
    '''
    
    clf = tenn.teNNClassifier(align_func='teNNCell', nclusters=args.nclusters, nu0=args.nu0, nuB=args.nuB, lambda_At=args.lambda_At, 
                              lambda_Ac=args.lambda_Ac, nepoch=args.nepoch, 
                              niterB=args.niterB, eta=args.eta, probe=args.probe, dataset=args.dataset) 

    #clf.fit(Xtrain, ytrain, args.nu0, args.epsilon, args.corridor_radius, args.eta, args.nepoch, args.batch_size, args.probe, args.OAT, args.OAC, args.ORF)
    clf.fit(Xtrain, ytrain, args.nu0, args.epsilon, args.corridor_radius, args.eta, args.nepoch, args.batch_size, args.probe, args.OAT, args.OAC, args.ORF,
             Xvalid=Xtest, yvalid=ytest, verbose=1)
    return clf

def relabel(labs, ly):
  '''
  relabel labels in labs using an integer in {0,1,2, .., len(np.unique(labs))}
  '''
  out = []
  for l in ly:
     out.append(int(labs.index(l)))
  return np.array(out, dtype=np.uint32)


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='BME', help='specify attention vector initialization : zeros | ones | random')
parser.add_argument('--nu0', default=1e-6, type=ranged_type(float, 0.0, 1e300), help='set initial nu values')
parser.add_argument('--nuB', default=1e-3, type=ranged_type(float, 0.0, 1e300), help='set nu parameter')
parser.add_argument('--nu1d', default=0, type=ranged_type(int, 0,1), help='set nu1d parameter')
parser.add_argument('--init_ref', default="centroids", help='initialize reference time series: medoid | centroid')
parser.add_argument('--init_cond', default="ones", help='initialize nu vector and act matrix to either to np.ones()*nu0 | np.random()*nu0 | np.zeros()')
parser.add_argument('--tmc_type', default='teNN', help='set the KAM type ("dist"|"sim")')
parser.add_argument('--epsilon', default=1e-300, type=ranged_type(float, 0.0, 100.0), help='set epsilon parameter')
parser.add_argument('--corridor_radius', default=10000,  type=ranged_type(int, 0.0, 100000), help='set corridor radius parameter. If <=0 then the corridor will be equal to the time series length.')
parser.add_argument('--eta', default=1e-1, type=ranged_type(float, 0.0, 1e3), help='set relaxation parameter')
parser.add_argument('--lambda_At', default=0, type=ranged_type(float, 0.0, 1.0), help='set lambda_nu parameter (norm L1 attention vector)')
parser.add_argument('--lambda_Ac', default=0, type=ranged_type(float, 0.0, 1e10), help='set lambda_act parameter (norm L1 activation vector)')
parser.add_argument('--normalize', default="False", help='set normalize parameter')
parser.add_argument('--nclusters', default=1, type=ranged_type(int, 0, 10000), help='define the number of centroids used to represent each class')
parser.add_argument('--nepoch', default=50000, type=ranged_type(int, 1, 100000), help='define the number of epochs')
parser.add_argument('--niterB', default=99, type=ranged_type(int, 1, 1000), help='define the number of iterations for estimation of barycenters')
parser.add_argument('--probe', default=20, type=ranged_type(int, 1, 100000), help='define the number of iteration before probing')
parser.add_argument('--batch_size', default=64, type=ranged_type(float, 0, 100000), help='size of batches in stochastic gradient descent. A value =0 means a single batch containing the full train set')

parser.add_argument('--OAT', default=1, type=ranged_type(int, 0,1), help='optimize the attention matrices if equal to 1')
parser.add_argument('--OAC', default=1, type=ranged_type(int, 0,1), help='optimize the activation matrices if equal to 1')
parser.add_argument('--ORF', default=1, type=ranged_type(int, 0,1), help='optimize the references times series if equal to 1')
parser.add_argument('--looc', action='store_true', help='activate the Leave One Out Procedure to find nu')
parser.add_argument('--loo', action='store_true', help='activate the Leave One Out Procedure to find nu')
parser.add_argument('--no_display', action='store_true', help='prohibit the displaying of figures')
parser.add_argument('--valid', default=0.0, type=ranged_type(float, 0.0, 1.0), help='use a validation subset. The value gives the ratio TRAIN/VALID, e.g .8 means 80% TRAIN, 20% VALID')
parser.add_argument('--split', default=0.0, type=ranged_type(float, 0.0, 1.0), help='use a split TRAIN/TEST subsets. The value gives the ratio TRAIN/TEST, e.g .8 means 80% TRAIN, 20% TEST')

try:
    args = parser.parse_args()
except SystemExit:
    os._exit(1)
print("############")
print(args)
   
normalize = False
if args.normalize == "True":
   normalize = True
       
Xtrain, ytrain, Xtest, ytest = ut.loadDataUCR_UEA(args.dataset, split=0, normalize=normalize, path = PATH, ext='')

if args.split>0:
    print("SPLIT TRAIN TEST RATIO :", args.split)
    X = np.concatenate((Xtrain,Xtest)) 
    y = np.concatenate((ytrain,ytest)) 
    Xtrain,ytrain,Xtest,ytest = ut.stratify(X,y,args.split,seed=0)
    print('AFTER SPLIT *** #train:', len(ytrain), '#test:',len(ytest))

print(flush=True)   
 
labs = list(np.unique(list(ytrain)+list(ytest)))
ytrain = relabel(labs,ytrain)
ytest = relabel(labs,ytest)


clf = teNN(Xtrain, ytrain, Xtest, ytest, args)
ypred_lm = clf.predict(Xtest, strategy='lastmin')
err_lm = np.sum(ypred_lm != ytest)
ypred_ml = clf.predict(Xtest, strategy='minloss')
err_ml = np.sum(ypred_ml != ytest)
ypred_c = clf.predict(Xtest, strategy='running')
err_c = np.sum(ypred_ml != ytest)
print()

print("############", args.dataset)
print("Lastmin Prediction accuracy", evm.accuracy_score(ytest, ypred_lm))
print("Minloss Prediction accuracy", evm.accuracy_score(ytest, ypred_ml))
print("Running Prediction accuracy", evm.accuracy_score(ytest, ypred_c))









