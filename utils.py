from glob import glob
import os, sys
import errno
import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn.model_selection import train_test_split

path0='/home/pfm/DATA/UCR_UEA_SKTIME/'
#path0='/home/marteau/LINUX/DATA/UCR_UEA_SKTIME/local_data/'

  
def sigmoid(x,lmbda):
    return 1.0/(1.0+np.exp(-lmbda*x))
def inv_sigmoid(y,lmbda):
    return 1/lmbda*(np.log(y)-np.log(1-y))

import inspect
def printFunc(F):
    lines = inspect.getsource(F)
    print(lines)
    
def split_on_y(X,y):
    out = dict()
    for i in range(len(y)):
       if y[i] not in out.keys():
          out[y[i]] = [X[i]]
       else:
          l = out[y[i]]
          l.append(X[i])
          out[y[i]] = l
    for k in out.keys():
       print('#class', k, len(out[k]))      
    return out

def stratifyS(splittedX,ratio_train_test,seed=42):
    r = ratio_train_test
    Xtest = []
    ytest = []
    Xtrain = []
    ytrain = []
    np.random.seed = seed
    for cat in splittedX.keys():
        Xsubset = np.array(splittedX[cat])
        I = list(range((len(Xsubset))))
        np.random.shuffle(I)
        cut = int(len(Xsubset)*ratio_train_test)
        for i in range(cut):
            Xtrain.append(Xsubset[I[i]])
            ytrain.append(cat)
        for i in range(len(Xsubset)-cut):
            Xtest.append(Xsubset[I[cut+i]])
            ytest.append(cat)
        
    return np.array(Xtrain), np.array(ytrain), np.array(Xtest), np.array(ytest)

def stratify(X,y,ratio_train_test, seed=42):
    np.random.seed(seed)
    splittedX = split_on_y(X,y)
    r = ratio_train_test
    Xtest = []
    ytest = []
    Xtrain = []
    ytrain = []
    
    for cat in splittedX.keys():
        Xsubset = np.array(splittedX[cat])
        I = list(range((len(Xsubset))))
        np.random.shuffle(I)
        cut = int(len(Xsubset)*ratio_train_test)
        for i in range(cut):
            Xtrain.append(Xsubset[I[i]])
            ytrain.append(cat)
        for i in range(len(Xsubset)-cut):
            Xtest.append(Xsubset[I[cut+i]])
            ytest.append(cat)
        
    return np.array(Xtrain), np.array(ytrain), np.array(Xtest), np.array(ytest)

def max_min_X(X):
    N,L,dim = np.shape(X)
    mx = -np.ones(dim)*1e300
    mn = np.ones(dim)*1e300
    for d in range(dim):
        for n in range(N):
            _mx = np.max(X[n,:,d])
            mx[d] = max(_mx,mx[d])
            _mn = np.min(X[n,:,d])
            mn[d] = min(_mn,mn[d])

    #return np.max(mx),np.min(mn)
    return mx,mn

def max_min_normalize_X(X,mx,mn):
    N,L,dim = np.shape(X)
    Xout = np.zeros((N,L,dim))
    for d in range(dim):
        for n in range(N):
            Xout[n,:,d] = (X[n,:,d]-mn[d])/(mx[d]-mn[d])
            #Xout[n,:,d] = (X[n,:,d]-mn)/(mx-mn)

    return Xout


def mean_std_X(X):
    N,L,dim = np.shape(X)
    mean = np.zeros(dim)
    std = np.zeros(dim)
    for d in range(dim):
        for n in range(N):
            mean[d] += np.sum(X[n,:,d])
        mean[d] /= N/L
    for d in range(dim):
        for n in range(N):
            std[d] += np.sum(np.power(X[n,:,d]-mean[d],2))
        std[d] = np.sqrt(std[d]/N/L)

    return mean,std

def z_normalize_X(X,mean,std):
    N,L,dim = np.shape(X)
    Xnz = np.zeros((N,L,dim))
    for d in range(dim):
        for n in range(N):
            Xnz[n,:,d] = (X[n,:,d]-mean[d])/std[d]

    return Xnz



def z_normalize_ts(ts):
   ts = ts - np.mean(ts, axis=0)
   M = np.std(ts, axis=0)+1e-300
   return ts/M

def normalize_ts0(ts):
   ts = ts - np.mean(ts, axis=0)
   M = np.max(np.abs(ts), axis=0)+1e-100
   #M = np.std(np.abs(ts), axis=0)+1e-100
   return ts/M
def normalize_ts(ts):
   ts = ts - np.mean(ts, axis=0)
   M = np.max(np.abs(ts), axis=0)
   m = np.min(ts, axis=0)
   #M = np.std(np.abs(ts), axis=0)+1e-100
   #return (ts-m)/(M-m+1e-300)
   return ts

def processUCRMetaData(file):
    H = {}
    # reading the data of the file and storing in the variable "data"
    try:
        with open(file) as f:
           for line in f:
               if line[0] == '@':
                   l = line[1:].split()
                   if len(l)>1:
                     H[l[0]] = l[1:]
               if not (line[0] == '#' or line[0] == '@'):
                   break
                   
    except IOError as exc:
        if exc.errno != errno.EISDIR:
            raise
    return H
               
def processUCRLine(line, length, dim):
   line = line.replace(':',',')
   line = line.replace('\n','')
   u = line.split(',')
   lu = len(u)
   lab = (u[len(u)-1])
   #(len(u)/dim, u)
   v = []
   if length == -1:
     length = int((len(u)-1)/dim)
   for i in range(dim):
     xi = []
     for j in range(length):
        xi.append(float(u[i*length+j]))
     v.append(xi)
   v = np.array(v)
   v = v.T
   return v, lab
   
def getUCRFile(file, normalize=False):
    '''
    The function would open the file passed as the argument and then read the data
    as string. It would return this data after splitting.
    '''
    X = []
    y = []
    head = processUCRMetaData(file)
    print(head)
    equalLength =  head['equalLength'][0]
    if equalLength == 'true' or equalLength == 'True':
        tsLength = int(head['seriesLength'][0])
    else:
        tsLength = -1
    if head['univariate'][0] == 'true' or head['univariate'][0] == 'True':
        dim = 1
    else:
        dim = int(head['dimensions'][0])
    print('length', tsLength, 'dimension:', dim, 'normalization:', normalize)
    # reading the data of the file and storing in the variable "data"
    try:
        with open(file) as f:
           for line in f:
              if not (line[0] == '#' or line[0] == '@' or line[0] == '\n'):
                 v, lab = processUCRLine(line, tsLength, dim)
                 if normalize:
                   v = z_normalize_ts(v)
                   #print("z_normalize")
                   #v = z_normalize_ts(v)
                 X.append(v)
                 y.append(lab)
    except IOError as exc:
        if exc.errno != errno.EISDIR:
            raise

    #return np.array(X,dtype='object'), np.array(y,dtype='object')
    return np.array(X), np.array(y)

def loadTrainTestUCR_UEAProblem(dataset, normalize=False, path=path0):
   filename=path+dataset+'/'+dataset+'_TRAIN.ts'
   Xtrain, ytrain = getUCRFile(filename, normalize=normalize)
   #split_on_y(Xtrain, ytrain)
   filename=path+dataset+'/'+dataset+'_TEST.ts'
   Xtest, ytest = getUCRFile(filename, normalize=normalize)
   labs = np.unique(ytrain)
   '''for lab in labs:
     for i in range(len(ytrain)):
        if lab == ytrain[i]:
           plt.plot(Xtrain[i])
           break
   plt.show()'''
   print('#Train:'+str(len(Xtrain)), '#Test:'+str(len(Xtest)), 'length:',len(Xtrain[0]), 'dim:',len(Xtrain[0][0]), '#labs',len(labs))

   '''if normalize:
       mx,mn = max_min_X(Xtrain)
       print("max/min normalizing with mx:",mx, "mn:",mn)
       Xtrain = max_min_normalize_X(Xtrain,mx,mn)
       Xtest = max_min_normalize_X(Xtest,mx,mn)'''
       
   #return np.array(Xtrain, dtype='object'), np.array(ytrain, dtype='object'), np.array(Xtest,dtype='object'), np.array(ytest,dtype='object')
   return np.array(Xtrain), np.array(ytrain), np.array(Xtest), np.array(ytest)
    
def test(dataset='NATOPS'):
   Xtrain, ytrain, Xtest, ytest = loadTrainTestUCR_UEAProblem(dataset)
   print('#TRAIN', len(ytrain))
   print('#TEST', len(ytest))
   labs = np.unique(ytrain)
   for lab in labs:
     for i in range(len(ytrain)):
        if lab == ytrain[i]:
           plt.plot(Xtrain[i])
           break
   plt.show()
   return Xtrain, ytrain, Xtest, ytest
  
#Xtrain, ytrain, Xtest, ytest =  test(dataset='CBF')


def loadDataUCR_UEA(dataset, split=0, normalize=False, path=path0):
    if not os.path.exists(path+'/'+dataset):
        if not os.path.exists(path+'/Multivariate_ts/'+dataset):
            print('file  '+dataset+' does not exist')
            os._exit(1)
        else:
            path = path+'/Multivariate_ts/'
    #Xtrain, ytrain = load_UCR_UEA_dataset(name=dataset, split='train', extract_path='/home/pfm/DATA/UCR_UEA_SKTIME/')
    #Xtest, ytest = load_UCR_UEA_dataset(name=dataset, split='test', extract_path='/home/pfm/DATA/UCR_UEA_SKTIME/')
    print(path)
    np.random.seed(split)
    random.seed(split)
    Xtrain, ytrain, Xtest, ytest = loadTrainTestUCR_UEAProblem(dataset, normalize=normalize, path=path)
    size_xtest = len(Xtest)
    if split > 0:
       X = np.concatenate((Xtrain,Xtest), axis=0)
       y = np.concatenate((ytrain,ytest), axis=0)
       Xtrain, Xtest, ytrain, ytest = train_test_split(X, y,  test_size=size_xtest, stratify=y)

    '''if split == 0:
        Xtrain, ytrain, Xtest, ytest = loadTrainTestUCR_UEAProblem(dataset, normalize=normalize)
        Xtrain, ytrain = load_UCR_UEA_dataset(name=dataset, split='train', extract_path=path)
        Xtest, ytest = load_UCR_UEA_dataset(name=dataset, split='test', extract_path=path)
    else:
        Xtest, ytest = load_UCR_UEA_dataset(name=dataset, split='test', extract_path=path)
        X, y = load_UCR_UEA_dataset(name=dataset, extract_path=path)
        Xtrain, Xtest, ytrain, ytest = train_test_split(X, y,  test_size=Xtest.size, stratify=y)
    
    Xtrain = convertDataFrameToNumpyArray(Xtrain, normalize=normalize)
    Xtest = convertDataFrameToNumpyArray(Xtest, normalize=normalize)'''

    #return np.array(Xtrain,dtype='object'), ytrain, np.array(Xtest,dtype='object'), ytest
    return np.array(Xtrain), ytrain, np.array(Xtest), ytest
 
