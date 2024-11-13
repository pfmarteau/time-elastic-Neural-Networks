import sys
import numpy as np
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.cluster import SpectralClustering
from ctypes import c_void_p, c_char_p, c_double, c_int, cdll

### C INTERFACE
lib = cdll.LoadLibrary("C/teNN.so")

_dtwc = lib.dtwc
_kdtw = lib.kdtw
_kdtwc = lib.kdtwc
_teNNCell = lib.teNNCell
_teNNCell_grad_re = lib.teNNCell_grad_re

_barycenter_ = lib.barycenter

_fit = lib.fit
_predict = lib.predict
_predict_proba = lib.predict_proba

_dtwc.restype = c_double
_kdtw.restype = c_double
_kdtwc.restype = c_double
_teNNCell.restype = c_double

_free = lib.freemem
lib.freemem.argtypes = c_void_p,
lib.freemem.restype = None


class teNNClassifier(ClassifierMixin, BaseEstimator):
    """ A Time Elastic Attention Kernel (TEAK) classifier which implements a first centroid neighbor algorithm with elastic alignment and dedicated metric learning.
    For more information regarding TEAK classifier, please refer to <CITE>
    Parameters
    ----------
    align_func : str, default='euclid' ('euclid', 'kdtw')
        A parameter used to define the way time series are aligned to the centroids.
    Attributes
    ----------
    X : ndarray, shape (n_samples, n_features)
        The input passed during :meth:`fit`.
    y : ndarray, shape (n_samples,)
        The labels passed during :meth:`fit`.
    classes_ : ndarray, shape (n_classes,)
        The classes seen at :meth:`fit`.
    """
    def __init__(self, align_func='teNNCell', initConditions='random', nu0=1, nuB=1, epsilon=1e-300, lambda_At=0.0, lambda_Ac=1e-4,  
                 probe=1000, nepoch=10000, npass=200, nclusters=1, niterB=50, eta =1e-3, dataset='', nchunk=1):

        self.infty = 1e200
        self.epsilon = epsilon
        self.npass = npass
        self.nclusters = nclusters
        self.nepoch = nepoch
        self.epoch = 0
        self.probe = probe
        self.nu0 = nu0
        self.nuB = nuB
        self.nu_max = 50
        self.niterB = niterB
        self.lambda_R = 1e-6
        self.lambda_At = lambda_At
        self.lambda_Ac = lambda_Ac
        self.eta = eta
        self.initConditions = initConditions
        self.dataset = dataset
        self.nchunk = nchunk
        self.align_func = align_func.lower()
        self.OR = True
        self.ONUA = True
        self.OACT = False
        self.OACT1 = False
        self.nprn = 0
        self.eps = np.ones(1)*1e-300
        self.ASP = 1e-6 #activation sparsity factor (force the sparsity of the Activation matrix)
        if align_func == 'teNNCell':
            self.dist = _teNNCell
        else:
            print('uncorrect alignment function (align_func parameter): '+align_func+'. Should be kdtwa')
            sys.exit(-1)
            
    # Partition X according to label y. 
    # Returns a dictionnary key = l, a label : value = subset of instances of X with label l.
    def split_on_y(self,X,y):
       out = dict()
       for i in range(len(y)):
          if y[i] not in out.keys():
             out[y[i]] = [X[i]]
          else:
             l = out[y[i]]
             l.append(X[i])
             out[y[i]] = l  
       return out
    
    # kdtw alignemnt kernel 
    def kdtw(self, x,y,nu,epsilon):
        return _kdtw(c_int(len(x[0])), c_void_p(x.ctypes.data), c_int(len(x)), c_void_p(y.ctypes.data), c_int(len(y)),\
                     c_double(nu), c_double(epsilon))
    # kdtw alignemnt kernel with corridor
    def kdtwc(self, x,y,nu,epsilon,corridor_radius):
        return _kdtwc(c_int(len(x[0])), c_void_p(x.ctypes.data), c_int(len(x)), c_void_p(y.ctypes.data), c_int(len(y)),\
                     c_double(nu), c_double(epsilon), c_int(corridor_radius))
              
    # teNN cell  
    def teNNCell(self,x,y,nu,act,epsilon, corridor_radius):
        return _teNNCell(c_int(len(x[0])), c_void_p(x.ctypes.data), c_int(len(x)), c_void_p(y.ctypes.data),\
                           c_int(len(y)), c_void_p(nu.ctypes.data), c_void_p(act.ctypes.data), c_double(epsilon),\
                           c_int(corridor_radius))
    
    # Evaluates the kdtw barycenter, r, of a subset of times series X   
    def barycenter(self, lab, X, r, At, Ac, eta, niter):
        _barycenter_.restype =  np.ctypeslib.ndpointer(dtype=c_double, shape=(niter), flags="C_CONTIGUOUS")
        lloss = _barycenter_(c_int(lab), c_int(len(X)), c_int(self.L), c_int(self.dim), c_void_p(X.ctypes.data),  
                     c_void_p(r.ctypes.data), c_void_p(At.ctypes.data), c_void_p(Ac.ctypes.data), c_double(self.epsilon),
                     c_int(self.corridor_radius), c_double(eta), c_int(niter))
        return lloss
    
    # fit method    
    def fit(self,X, y, nu0=1e-3, epsilon=1e-300, corridor_radius=100000, eta=1e-2, nepoch=10000, batch_size=64, probe=20, OAT=1, \
    		OAC=1, ORF=1, Xvalid=None, yvalid=None, verbose=1):
        self.X = X
        self.y = y
        lx,dim = np.shape(X[0])
        self.labs = np.unique(y)
        self.NC = len(self.labs)
        NX = len(X)
        self.NR = self.NC*self.nclusters
        self.L = lx
        self.dim = dim
        self.nu0 = nu0/dim 
        self.epsilon = epsilon
        self.eps = np.ones(self.NR)*1e-300
        self.corridor_radius = corridor_radius
        self.eta = eta
        self.batch_size = batch_size
        self.nepoch = nepoch
        self.display = False
        self.probe = probe
        self.OAT = OAT
        self.OAC = OAC
        self.ORF = ORF

        self.average(X, y, nu0=nu0, niter=self.niterB) #Evaluate centroids of classes
        if Xvalid is None:
            Xvalid = np.array([], dtype=c_double)
            yvalid = np.array([], dtype=c_int)
            NV = 0
        else:
            NV = len(Xvalid)

        _fit.restype =  np.ctypeslib.ndpointer(dtype=c_double, shape=(nepoch), flags="C_CONTIGUOUS")
        
        for k in range(self.NR):
          for i in range(lx):
            for j in range(lx):
                if (j<i-corridor_radius or j>i+corridor_radius):
                    self.Ac[k,i,j] = 0.0
	         
        RES = _fit(c_int(NX), c_int(self.L), c_int(self.dim), c_void_p(X.ctypes.data), c_void_p(y.ctypes.data), 
                 c_double(self.epsilon), c_int(self.corridor_radius), c_int(self.NC), c_void_p(self.labs.ctypes.data), 
                 c_int(self.NR), c_void_p(self.yR.ctypes.data), 
                 c_void_p(self.R.ctypes.data), c_void_p(self.At.ctypes.data), c_void_p(self.Ac.ctypes.data), 
                 c_void_p(self.R_lm.ctypes.data), c_void_p(self.At_lm.ctypes.data), c_void_p(self.Ac_lm.ctypes.data), 
                 c_void_p(self.R_ml.ctypes.data), c_void_p(self.At_ml.ctypes.data), c_void_p(self.Ac_ml.ctypes.data), 
                 c_double(self.lambda_At), c_double(self.lambda_Ac), c_double(eta), c_int(nepoch), c_double(batch_size), 
                 c_int(self.probe), c_int(self.OAT), c_int(self.OAC), c_int(self.ORF),
                 c_int(NV), c_void_p(Xvalid.ctypes.data), c_void_p(yvalid.ctypes.data), c_int(verbose))
        lloss = RES    
        self.nepoch = int(lloss[0])
        self.lloss = lloss[1:self.nepoch]

    # predict method
    def predict(self, X, strategy='lastmin'):
        if strategy == "lastmin":
            R = self.R_lm
            At = self.At_lm
            Ac = self.Ac_lm
        elif strategy == "minloss":
            R = self.R_ml
            At = self.At_ml
            Ac = self.Ac_ml
        else:
            R = self.R
            At = self.At
            Ac = self.Ac
        _predict.restype = np.ctypeslib.ndpointer(dtype=c_int, shape=(len(X)), flags="C_CONTIGUOUS")

        return _predict(c_int(len(X)), c_int(self.L), c_int(self.dim), c_void_p(X.ctypes.data), c_int(self.NR), c_void_p(self.yR.ctypes.data),
                        c_void_p(R.ctypes.data), c_void_p(At.ctypes.data), c_void_p(Ac.ctypes.data),
                        c_double(self.epsilon), c_int(self.corridor_radius), c_int(self.NC), 
                        c_void_p(self.labs.ctypes.data))

    # predict proba method
    def predict_proba(self, X, strategy='lastmin'):
        if strategy == "lastmin":
            R = self.R_lm
            At = self.At_lm
            Ac = self.Ac_lm
        elif strategy == "minloss":
            R = self.R_ml
            At = self.At_ml
            Ac = self.Ac_ml
        else:
            R = self.R
            At = self.At
            Ac = self.Ac
        _predict_proba.restype = np.ctypeslib.ndpointer(dtype=c_double, shape=(len(X),self.NC), flags="C_CONTIGUOUS")
        return _predict(c_int(len(X)), c_int(self.L), c_int(self.dim), c_void_p(X.ctypes.data), c_int(self.NR), 
                        c_void_p(R.ctypes.data), c_void_p(self.yR.ctypes.data), c_void_p(At.ctypes.data), c_void_p(Ac.ctypes.data),
                        c_double(self.epsilon), c_int(self.corridor_radius), c_int(self.NC), 
                        c_void_p(self.labs.ctypes.data))

    # evaluate kdtw-like similarity matrix useavle as an affinity matrix for spectral clustering 
    def affinityMatix(self, X, nu, epsilon, corridor_radius, align_func='cos_kdtwc'):
        N=len(X)
        A = np.zeros((N,N))
        dii = []
        for i in range(N):
            if align_func=='cos_kdtw':    
                dii.append(self.kdtw(X[i], X[i], nu, epsilon))              
            elif align_func=='cos_kdtwc':
                dii.append(self.kdtwc(X[i], X[i], nu, epsilon, corridor_radius))
    
        for i in range(N):  
            for j in range(i,N):
                if align_func=='kdtw': 
                    d =  self.kdtw(X[i], X[j], nu, epsilon)
                elif align_func=='kdtwc': 
                    d =  self.kdtwc(X[i], X[j], nu, epsilon, corridor_radius)
                elif align_func=='cos_kdtwc': 
                    d =  self.kdtwc(X[i], X[j], nu, epsilon, corridor_radius)/(np.sqrt(dii[i]*dii[j])+1e-300)
                A[i,j] = d
                A[j,i] = d
        return A           
    
    # average X, a subset of time series if nclusters == 1. If nclusters > 1 then clusterize X in nclusters, and average each cluster.
    def average(self, X, y, nu0=1, eta=1., niter=100):
        N,L,dim = np.shape(X)
        ds = self.split_on_y(X,y)
        lX = []
        ly =[]
        if self.nclusters == 1:
            for k in ds.keys():
                lX.append(ds[k])
                ly.append(k)
        else:
            for k in ds.keys():
                X0 = ds[k]
                nclust = self.nclusters
                if nclust>len(X0):
                    nclust = len(X0)
                print(k, len(X0), nclust, 'Evaluate affinity matrix...', end='', flush=True)
                A = self.affinityMatix(X0, nu0, self.epsilon, self.corridor_radius, align_func='kdtwc')
                print(' done!',flush=True)
                print('Evaluate spectral clustering...', end='', flush=True)
                sc = SpectralClustering(nclust, affinity='precomputed', n_init=100, assign_labels='discretize')
                sc.fit_predict(A)
                y0 = sc.labels_               
                print('done!',flush=True)
                dsC = self.split_on_y(X0,y0)
                for k1 in dsC.keys():
                    lX.append(dsC[k1])
                    ly.append(k)

        i = 0
        if len(ly)<self.NR:
            self.NR = len(ly)

        self.R = np.zeros((self.NR,self.L,self.dim))
        self.R_lm = np.zeros((self.NR,self.L,self.dim))
        self.R_ml = np.zeros((self.NR,self.L,self.dim))
        self.At = np.ones((self.NR,self.L,self.dim))*self.nu0
        self.At_lm = np.ones((self.NR,self.L,self.dim))*self.nu0
        self.At_ml = np.ones((self.NR,self.L,self.dim))*self.nu0
        self.Ac = np.ones((self.NR,self.L,self.L))
        self.Ac_lm = np.ones((self.NR,self.L,self.L))
        self.Ac_ml = np.ones((self.NR,self.L,self.L))

        self.yR = ly
        self.R_freqs = []
        for k in range(len(ly)):
             X0 = np.array(lX[k])
             self.R_freqs.append(len(X0)/N)
             self.R[k] = np.copy(X0[0])
             lloss = self.barycenter(self.yR[k], X0, self.R[k], self.At[k], self.Ac[k], eta, niter)
             i += 1
        
        self.yR = np.array(self.yR)
        self.R_freqs = np.array(self.R_freqs)

        self.centroids = np.copy(self.R)

        for k in range(self.NR):
             plt.figure(k)
             plt.plot(lX[k][0][:,0], label="X0[0]")
             plt.plot(self.centroids[k][:,0], label="B")
             plt.legend()
        plt.figure(1000)
        plt.plot(lloss);
        
        if self.display:
            plt.show()
        else:
            plt.close('all')

        
    # display the activation matrices
    def displayActivation(self,  _type="lm", _log=False, ext='png'):
        if _type == "lm":
          Ac = self.Ac_lm
        elif _type == "ml":
          Ac = self.Ac_ml
        else:
          Ac = self.Ac

        M = Ac
        for i in range(len(self.centroids)):
            plt.figure(i)
            M1 = M[i]
            if _log:
                M1 = np.log(M1)
            img = plt.imshow(M1, cmap='bwr', interpolation='none')
            cbar = plt.colorbar(img)
            cbar.set_label("Colorbar")
            plt.grid()
            plt.savefig('figs/'+self.dataset+'_Activation_'+str(i)+'.'+ext)
            if self.display:
                plt.show()
            plt.close('all')

    # display the attention matrices along dimension d.   
    def displayAttention_d(self, d=0, _type="lm", withR=True, ext='png'):
        yR = self.yR
        if _type == "lm":
          R = self.R_lm
          At = self.At_lm
        elif _type == "ml":
          R = self.R_ml
          At = self.At_ml
        else:
          R = self.R
          At = self.At
        for i in range(len(R)):
            plt.figure(i)
            for d in range(self.dim):
              if withR:
                  mx = np.max(R[i][:,d])
                  plt.plot(R[i][:,d], '-.', label='R '+str(yR[i]))
              mxx = np.max(At[i][:,d])
              plt.plot(At[i][:,d]/mxx*mx, label='att_'+str(d))

            plt.legend()
            plt.grid()
            plt.savefig('figs/'+self.dataset+'Attention_d'+str(d)+'_'+str(i)+'.'+ext)
        if self.display:
            plt.show()  
        plt.close('all')

    # display the attention matrices
    def displayAttention(self,  _type="lm", ext='png'):
        if _type == "lm":
          At = self.At_lm
        elif _type == "ml":
          At = self.At_ml
        else:
          At = self.At

        for i in range(len(self.centroids)):
            plt.figure(i, frameon=False)
            u = np.transpose(At[i])
            img = plt.imshow(u, label='At_'+str(i), cmap='bwr', origin='lower')
            #cbar = plt.colorbar(img)
            plt.legend()
            plt.grid()
            plt.savefig('figs/'+self.dataset+'Attention_'+str(i)+'.'+ext, bbox_inches='tight')
        if self.display:
            plt.show()  
        plt.close('all')
    
    # display reference matrices R 
    def displayR(self, d=0, _type="lm", withCentroids=True, ext='png'):
        if _type == "lm":
          R = self.R_lm
        elif _type == "ml":
          R = self.R_ml
        else:
          R = self.R
        yR = self.yR
        for i in range(len(yR)):
           plt.figure(i)
           plt.plot(R[i][:,d], label='R '+str(yR[i]))
           if withCentroids:
               plt.plot(self.centroids[i][:,d], '-.', label='C '+str(yR[i]))
           plt.legend()
           plt.grid()
           plt.savefig('figs/'+self.dataset+'R '+str(yR[i])+'.'+ext)
        if self.display:
            plt.show()  
        plt.close('all')
    
    # display the training loss as a function of the epoch
    def displayLoss(self, ext='png', logscale=False):
        nepoch = self.nepoch
        lloss = self.lloss
        if logscale:
            plt.plot(np.log(lloss+1e-300))
            plt.ylabel('log(loss)')
        else:
            plt.plot(lloss)
            plt.ylabel('loss')
        plt.grid()
        plt.xlabel('epoch') 
        plt.savefig('figs/'+self.dataset+'_loss'+'.'+ext)
        if self.display:
            plt.show()
        plt.close('all')
    
    # display some time series instances
    def displaySamples(self, d=0, n=3, withCentroids=False, ext='png'):
        yR = self.yR
        for i in range(len(yR)):
            plt.figure(i)
            j = 0
            k = 0 
            for j in range(len(self.X)):
               if self.y[j] == yR[i]:
                   plt.plot(self.X[j][:,d])
                   k += 1
               if k == n:
                   break
            if withCentroids:
               plt.plot(self.centroids[i][:,d], '-.', linewidth=3, label='C '+str(yR[i]))
            plt.title('Samples for category '+str(yR[i]))
            plt.grid()
            plt.savefig('figs/'+self.dataset+'_'+str(i)+'.'+ext)
        if self.display:
            plt.show()        
    
    # display all (calling previous display methods)
    def displayAll(self, d=0, _type='lm', show=False):
        if show==True:
            self.display=True
        self.displaySamples(d=d, n=3, withCentroids=True, ext='png')
        self.displayLoss()
        self.displayR(d)
        self.displayActivation(_type=_type)
        if self.dim>1:
            self.displayAttention(_type=_type)
        else:
            self.displayAttention_d(_type=_type)

               











