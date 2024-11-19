#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <malloc.h>
#include <limits.h>
#include <float.h>
#include <signal.h>

#define EPSILON 1e-270
#define EPSI 1e-20
#define _LAMBDA_SIGMOID .01
#define _MAX_SIGMOID 1.0
#define BINF -1e300
#define BSUP 1e300
#define MARGE 1
#define CROSS_CATEGORICAL_ENTROPY 0

static volatile int keepRunning = 1;
static double SZDBL = sizeof(double);
static double _epsi = 1e-300;

char *_DATASET=NULL;
int  _NEPOCH=-1, _VERBOSE=1;
FILE *_FLOG=NULL;

/*
 print the content of 1D array arr. type is either 'i' for int or 'd' for double.
*/
void print_1Darray(void *arr, unsigned int size, char type){
fprintf(stdout,"---------------------\n");
  if(type=='i'){
     for(int i=0; i<size; i++)
        fprintf(stdout,"%d,", ((int *)arr)[i]);
     fprintf(stdout,"\n");
     }
  else if(type=='d'){
     for(int i=0; i<size; i++)
        fprintf(stdout,"%e,", ((double *)arr)[i]);
     fprintf(stdout,"\n");
     }
fprintf(stdout,"---------------------\n");
}

/*
 reset the content of 1D int array arr with value val. arr should have been previously correctly allocated.
*/
void reset_1DIarray(int *arr, unsigned int size, int val){
for(int i=0; i<size; i++)
    arr[i] = val;
}

/*
 reset the content of 1D double array arr with value val. arr should have been previously correctly allocated.
*/
void reset_1DDarray(double *arr, unsigned int size, double val){
for(int i=0; i<size; i++)
    arr[i] = val;
}

/*
 copy the content of 1D array arr2 into AD array arr1. arr1 should have been previously correctly allocated.
*/
void cpy_1Darray(void *arr1, void *arr2, unsigned int size){
    memcpy(arr1, arr2, size);
}

/*
 used for interrupting the fit function.
*/
void intHandler(int dummy) {
    keepRunning = 0;
}

/*
 returns the max value of integers x and y
*/
int _maxi(int x, int y){
  if (x>y) return x;
  return y;
  }

/*
 returns the min value of integers x and y
*/  
int _mini(int x, int y){
  if (x>y) return y;
  return x;
  }
  
/*
 returns the max value of double values x and y
*/    
double _max(double x, double y){
  if (x>y) return x;
  return y;
  }

/*
 returns the min value of doubles values x and y
*/  
double _min(double x, double y){
  if (x>y) return y;
  return x;
  }

/*
 returns the min value of doubles values x, y and z
*/ 
double _min3(double x, double y, double z){
  if (x>y) 
     return _min(y,z);
  else 
     return _min(x,z);
  }
  
/* 
 Arrange the N elements of ARRAY in random order.
 Only effective if N is much smaller than RAND_MAX;
 if this may not be the case, use a better random
 number generator. 
*/
void shuffle(int *array, size_t n, int initialize)
{
if (initialize==1){
    for(int i=0; i<n; i++){
        array[i] = i;
    }
}
if (n > 1) {
        size_t i;
        for (i = 0; i < n - 1; i++) 
        {
          size_t j = i + rand() / (RAND_MAX / (n - i) + 1);
          int t = array[j];
          array[j] = array[i];
          array[i] = t;
        }
    }
}

/*
 get the set of unique integers that occur in tArr. The size of the return array is in varaible sz
*/
int *getUnique(const int *tArr, unsigned int *sz){
int unique_count = 0;
int i, j=0;
int *tout = (int *)calloc(*sz, sizeof(int));
for (i = 0; i < *sz; i++) {
    for (j = 0; j < unique_count; j++)
        if (tArr[i] == tout[j])
            break;
    if (j == unique_count){
        tout[unique_count] = tArr[i];
        unique_count++;
        }
}
fprintf(stdout,"realloc %d to %d\n", *sz, unique_count); fflush(stdout);
tout = (int *)realloc(tout, unique_count*sizeof(int));

*sz = unique_count;
return tout;
}

/*
get the best references ids for each teNN cells. The best reference id among the best is in best_ref variable.
*/
int *getBestRef(int NC, int *labs, int NR, const int *yref, double *LSim, int *best_ref){
int *out = (int *)malloc(NC*sizeof(int));
double mmax = -1e300;
for(int c=0; c<NC; c++){
  double max= -1e300;
  int rmax = 0;
  for(int r=0; r<NR; r++){
     if (labs[c] == yref[r]){
        if (max<LSim[r]){
           max = LSim[r];
           rmax = r;
        }
     }
  }
  out[c] = rmax;
  if(mmax<max){
    mmax = max;
    *best_ref = rmax;
  }
}
return out;
}

/*
get the best NEG reference ids among all the teNN cells. 
*/
int getBestNegRef(int lab, int NR, const int *yref, double *LSim){
double max = -1e300;
int rmax = 0;
for(int nr=0; nr<NR; nr++){
   if (lab != yref[nr]){
        if (max<LSim[nr]){
           max = LSim[nr];
           rmax = nr;
        }
     }
  }
return rmax;
}

/*
get the best NEG reference ids among all the teNN cells. 
*/
int getBestPosRef(int lab, int NR, const int *yref, double *LSim){
double max = -1e300;
int rmax = 0;
for(int nr=0; nr<NR; nr++){
   if (lab == yref[nr]){
        if (max<LSim[nr]){
           max = LSim[nr];
           rmax = nr;
        }
     }
  }
return rmax;
}

/*
free memory allocated for pointer ptr
*/
void freemem(char *ptr)
{
    free(ptr);
}

/*
Eval sparsity of 1D array arr.
*/
double eval_sparsity_1Darray(double *arr, unsigned int size, double threshold){
double sp = 0;
for(int i=0; i<size; i++)
    if (arr[i]<threshold){
      sp += 1.0;
      }
return sp/size;
}

/*
Return the maximum value of 1D array arr.
*/
double max_1DDarray(double *arr, unsigned int size){
double mx = -1e300;
for(int i=0; i<size; i++)
    if (mx<arr[i]){
      mx = arr[i];
      }
return mx;
}

/*
Return the minimum value of 1D array arr.
*/
double min_1DDarray(double *arr, unsigned int size){
double mn = 1e300;
for(int i=0; i<size; i++)
    if (mn>arr[i]){
      mn = arr[i];
      }
return mn;
}

/*
Fill the main diagonal of matrix arr whose shape is size1 x size2 represented as 1D array. The filling value is val.
*/
void filldiagonal_1Darray(double *arr, unsigned int size1, unsigned int size2, double val){
if(size1>size2)
  size1 = size2;
for(int i=0; i<size1; i++)
    arr[i*size1+i] = val;
}

/*
Return the norm L1 of 1D array arr.
*/
double norm1_1DDarray(double *arr, unsigned int size){
double nrm = 0;
for(int i=0; i<size; i++)
  if(!isnan(arr[i]))
    nrm += fabs(arr[i]);
return nrm;
}

/*
Return the norm L2 of 1D array arr.
*/
double norm2_1DDarray(double *arr, unsigned int size){
double nrm = 0, a;
for(int i=0; i<size; i++){
  a = pow(arr[i],2);
  if(!isnan(a) && isfinite(a))
    nrm += a;
  else nrm = DBL_MAX;
  }
return sqrt(nrm);
}


/*
Add element wise array arr to scalar*garr and save the result in arr.
*/
void add_1DDarray(double *arr, double *garr, unsigned int size, double scalar){
if (scalar!= 1.0)
  for(int i=0; i<size; i++)
    arr[i] += scalar*garr[i];
else
  for(int i=0; i<size; i++)
    arr[i] += garr[i];
}

/*
Add element wise random noise in array arr1 with standard deviation in sigma and save it in arr0.
*/
void addnoise_1DDarray(double *arr0, double *arr, unsigned int size, double sigma){
  for(int i=0; i<size; i++)
    arr0[i] = arr[i] + sigma*rand()/RAND_MAX;
}

/*
Replace NaN values by 0.0 in 1D array arr.
*/
void nan2zero_bounded_1DDarray(double *arr, unsigned int size){
int sign;
for(int i=0; i<size; i++){
  if (arr[i]>0)
     sign = 1;
  else
     sign = -1;
  if(isnan(arr[i]))
      arr[i] = 0.0;
  else if (!isfinite(arr[i]) || sign*arr[i]>1e30)
      arr[i] = sign*1e30;
  }
}

/*
Adapt 1D array arr given gradient vector garr, using parameters lambda, relax (\eta), and the limits binf and bsup.
*/
double adapt_1DDarray_sparsity(double *arr, unsigned int size, double relax, double lambda){
double pen = 0;

if(lambda>0){
  for(int i=0; i<size; i++){
     arr[i] -= relax*lambda;
     pen += arr[i];
  }
}
return pen*lambda;
}  

/*
 Adapt 1D array arr given gradient vector garr, using parameters lambda, relax (\eta), the limits binf and bsup and the norm (nrm) of vector garr.
 Returns the loss component corresponding to the L1 norm of arr weighted by lambda.
*/
double adapt_1DDarray(double *arr, double *garr, unsigned int size, double eta, double lambda, double binf, double bsup, double *norm){
double mx, a, pen = 0;
//nan2zero_bounded_1DDarray(garr, size);
//*norm = norm2_1DDarray(garr, size);
if(!isnan(*norm) && isfinite(*norm) && *norm>0){
  for(int i=0; i<size; i++){
    a = eta*garr[i]/(*norm);
    if(!isnan(a) && isfinite(a))
        arr[i] += a;
    }
  }
    
for(int i=0; i<size; i++){
  if (lambda>0)// && fabs(garr[i])<=(*norm)*1e-3)
       arr[i] -= eta*lambda*(arr[i]+.5);
  if (arr[i]>bsup)
       arr[i] = bsup;
  if (arr[i]<binf)
       arr[i] = binf;
  pen += arr[i]*(arr[i]+1)/2;
}  
  
return lambda*pen;
}

/*
 Print 2D array arr with shape size x dim.
*/
void print_2Darray(void **arr, unsigned int size, unsigned int dim, char type){
  if(type=='i')
     for(int i=0; i<size; i++){
        for(int j=0; j<dim; j++)
           fprintf(stdout,"%d,", ((int **)arr)[i][j]);
        fprintf(stdout,"\n");
        }
  else if(type=='d')
     for(int i=0; i<size; i++){
        for(int j=0; j<dim; j++)
           fprintf(stdout,"%e,", ((double **)arr)[i][j]);
        fprintf(stdout,"\n");
        }
}

/*
 Print string st.
*/
void out(char *st){
    fprintf(stdout, "%s", st);
    fflush(stdout);
    }

/*
 Evaluate Minkowski norm of degree 'degree' of array x whose shape is lx x dim.
*/
double Norm(double *x, unsigned int lx, unsigned int dim, unsigned int degree) {
    double S=0.0;
    for(int i=0; i<lx; i++){
      	for(int k=0; k<dim; k++){ 
      	   S += pow(fabs(x[i*dim+k]),degree);
      	}
    }
    return pow(S,1.0/degree);
}

/*
 Evaluate the Euclidean distance between time series va and vb.
*/
double sed(const unsigned dim, const double *va, const unsigned int la, const double *vb, const unsigned int lb) {
    // I:dim: dimension of multivariate time series va and vb
    // I:la: length of time series va
    // I:va: multivariate time series va
    // I:lb: length of time series vb
    // I:vb: multivariate time series vb
    // O:return value : float sed(dim,va,la, vb, lb, corridor_radius)
    
    double d,dist=0;
    unsigned int l=la;
    if (lb<la)
        l=lb;

    for(unsigned int i=0; i<l; i++){
        for (unsigned int k = 0; k < dim; k++){
           d = va[i*dim+k] - vb[i*dim+k];
           dist += d*d;
           } 
    }
    return sqrt(dist);
}

/*
 Evaluate the DTW distance between time series va and vb.
*/
double dtw(const unsigned dim, const double *va, const unsigned int la, const double *vb, const unsigned int lb) {
    // I:dim: dimension of multivariate time series va and vb
    // I:la: length of time series va
    // I:va: multivariate time series va
    // I:lb: length of time series vb
    // I:vb: multivariate time series vb
    // O:return value : float dtw(dim,va,la, vb, lb)

    unsigned int i,j,k;
    double d, dist;
 
    double **D = (double **)malloc((la+1)*sizeof(double *));
    for (i=0; i<=la; i++){
       D[i] = (double *)malloc((lb+1)*sizeof(double));
       for (j=0; j<=lb; j++){
 	  D[i][j] = 1e300;
       }
    }
 
    for(i=1; i<=la; i++){
        for (j = 1; j <= lb; j++) {
            dist = 0;
            for (k = 0; k < dim; k++){
                    d = va[(i-1)*dim+k] - vb[(j-1)*dim+k];
                    dist += d*d;
                    }
            D[i][j] = sqrt(dist);
        }
    }

    D[0][0] = 0;

    for (i = 1; i <= la; i++) {
        for (j = 1; j <= lb; j++) {
            D[i][j] = _min3(D[i - 1][j - 1], D[i - 1][j], D[i][j - 1]) + D[i][j];
        }
    }
    dist = D[la][lb];

    for (i=0; i<=la; i++){
       free(D[i]);
       }
    free(D);
    
    return dist;
}

/*
 Evaluate the DTW distance with corridor between time series va and vb.
*/
double dtwc(const unsigned dim, const double *va, const unsigned int la, const double *vb, const unsigned int lb, const int corridor_radius) {
    // I:dim: dimension of multivariate time series va and vb
    // I:la: length of time series va
    // I:va: multivariate time series va
    // I:lb: length of time series vb
    // I:vb: multivariate time series vb
    // I:corridor_radius: size of the corridor radius.
    // O:return value : float dtw(dim,va,la, vb, lb, corridor_radius)
    unsigned int i,j,k,beg,end;
    double d, dist;
 
    double **D = (double **)malloc((la+1)*sizeof(double *));
    for (i=0; i<=la; i++){
       D[i] = (double *)malloc((lb+1)*sizeof(double));
       for (j=0; j<=lb; j++){
 	  D[i][j] = 1e300;
       }
    }
 
    for(i=1; i<=la; i++){
        beg = _maxi(1,i-corridor_radius);
        end = _mini(lb, i+corridor_radius);
        for (j = beg; j <= end; j++) {
            dist = 0;
            for (k = 0; k < dim; k++){
                    d = va[(i-1)*dim+k] - vb[(j-1)*dim+k];
                    dist += d*d;
                    }
            D[i][j] = sqrt(dist);
        }
    }

    D[0][0] = 0;

    for (i = 1; i <= la; i++) {
        beg = _maxi(1,i-corridor_radius);
        end = _mini(lb, i+corridor_radius);
        for (j = beg; j <= end; j++) {
            D[i][j] = _min3(D[i - 1][j - 1], D[i - 1][j], D[i][j - 1]) + D[i][j];
        }
    }
    dist = D[la][lb];

    for (i=0; i<=la; i++){
       free(D[i]);
       }
    free(D);
    
    return dist;
}

/*
 Evaluate the KDTW kernel between time series va and vb.
*/
double kdtw(const unsigned dim, const double *va, const unsigned int la, const double *vb, const unsigned int lb, double nu, const double epsilon) {
    // I:dim: dimension of multivariate time series va and vb
    // I:la: length of time series va
    // I:va: multivariate time series va
    // I:lb: length of time series vb
    // I:vb: multivariate time series vb
    // I:nu: parameter of the local kernel (exp(-nu*delta(va(i),vb(j))+epsilon)
    // I:epsilon: parameter of the local kernel (exp(-nu*delta(va(i),vb(j))+epsilon)
    // O:return value, kdtw(va,vb,nu,epsilon)

    double dist;
    int i,j,k;
    double factor=1.0/3.0;
    double d;
    int l=la;
    if (lb>la)
        l=lb;
        
    nu /= dim;
    
    double *D3 = (double *)calloc(l+1, sizeof(double));
    double **D = (double **)calloc(la+1, sizeof(double *));
    double **D0 = (double **)calloc(la+1, sizeof(double *));
    for (i=0; i<=la; i++){
       D[i] = (double *)calloc(lb+1, sizeof(double));
       D0[i] = (double *)calloc(lb+1, sizeof(double));
       }

    for(i=1; i<=la; i++){
        for (j = 1; j <= lb; j++) {
            dist = 0;
            for (k = 0; k < dim; k++){
            	d = va[(i-1)*dim+k] - vb[(j-1)*dim+k];
                dist += d*d;
                }
            D[i][j] = factor*exp(-nu*dist)+epsilon;
            if(i==j)
                D0[i][j]=D[i][j];
        }
    }

    for(j=1; j<=l; j++) {
        if(j<=la&&j<=lb){
            dist=0;
            for(k=0; k<dim; k++){
            	d = va[(j-1)*dim+k]-vb[(j-1)*dim+k];
                dist+=d*d;
                }
            D3[j]=factor*exp(-nu*dist)+epsilon;
        }
    }

    D[0][0] = 1;
    D0[0][0] = 1;
    D3[0]=1;
    for (i = 1; i <= la; i++) {
        D[i][0] = factor*D[i-1][0]*D[i][1];
        D0[i][0] = factor*D0[i-1][0]*(D3[i]);
    }
    for (j = 1; j <= lb; j++) {
        D[0][j] = factor*D[0][j-1]*D[1][j];
        D0[0][j] = factor*D0[0][j-1]*D3[j];
    }

    double dij;
    for (i = 1; i <= la; i++) {
        for (j = 1; j <= lb; j++) {
              dij = D[i][j];
              D[i][j] = (D[i - 1][j - 1] + D[i - 1][j] + D[i][j - 1])* dij;
              D0[i][j] = (D0[i-1][j]+ D0[i][j-1])*(D3[i] + D3[j])/2.0;
              if (i == j) 
                  D0[i][j] += D0[i - 1][j - 1]*D3[i];          
              }  
        }
    
    dist = D[la][lb] + D0[la][lb] ;

    for (i=0; i<=la; i++){
       free(D[i]);
       free(D0[i]);
       }
    free(D);
    free(D0);
    free(D3);

    return dist;
}

/*
 Evaluate the KDTW kernel with corridor between time series va and vb.
*/
double kdtwc(const unsigned dim, const double *va, const unsigned int la, const double *vb, const unsigned int lb, double nu, const double epsilon, const int corridor_radius) {
    // I:dim: dimension of multivariate time series va and vb
    // I:la: length of time series va
    // I:va: multivariate time series va
    // I:lb: length of time series vb
    // I:vb: multivariate time series vb
    // I:nu: parameter of the local kernel (exp(-nu*delta(va(i),vb(j))+epsilon)
    // I:epsilon: parameter of the local kernel (exp(-nu*delta(va(i),vb(j))+epsilon)
    // I:corridor_radius: parameter defining the alignment search paths around the main diagonal
    // O:return value, kdtw(va,vb,nu,epsilon)

    double dist;
    int i,j,k,beg,end;
    double factor=1.0/3.0;
    double d;
    int l=la;
    if (lb>la)
        l=lb;
        
    nu /= dim;

    double *D3 = (double *)calloc(l+1, sizeof(double));
    double **D = (double **)calloc(la+1, sizeof(double *));
    double **D0 = (double **)calloc(la+1, sizeof(double *));
    for (i=0; i<=la; i++){
       D[i] = (double *)calloc(lb+1, sizeof(double));
       D0[i] = (double *)calloc(lb+1, sizeof(double));
       }
    for(i=1; i<=la; i++){
        beg = _maxi(1,i-corridor_radius);
        end = _mini(lb, i+corridor_radius);
        for (j = beg; j <= end; j++) {
            dist = 0;
            for (k = 0; k < dim; k++){
            	d = va[(i-1)*dim+k] - vb[(j-1)*dim+k];
                dist += d*d;
                }
            D[i][j] = factor*exp(-nu*dist) + epsilon;
            if(i==j)
                D0[i][j] = D[i][j];
        }
    }

    for(j=1; j<=l; j++){
        if(j<=la&&j<=lb){
            dist=0;
            for(k=0; k<dim; k++){
            	d = va[(j-1)*dim+k]-vb[(j-1)*dim+k];
                dist += d*d;
                }
            D3[j]=factor*exp(-nu*dist) + epsilon;
        }
    }

    D[0][0] = 1;
    D0[0][0] = 1;
    D3[0]=1;
    end = _mini(corridor_radius, la);
    for (i = 1; i <= end; i++) {
        D[i][0] = factor*D[i-1][0]*D[i][1];
        D0[i][0] = factor*D0[i-1][0]*(D3[i]);
    }
    end = _mini(corridor_radius, lb);
    for (j = 1; j <= end; j++) {
        D[0][j] = factor*D[0][j-1]*D[1][j];
        D0[0][j] = factor*D0[0][j-1]*D3[j];
    }

    double dij;
    for (i = 1; i <= la; i++) {
        beg = _maxi(1,i-corridor_radius);
        end = _mini(lb, i+corridor_radius);
        for (j = beg; j <= end; j++) {
            dij = D[i][j];
            D[i][j] = (D[i - 1][j - 1] + D[i - 1][j] + D[i][j - 1])*dij;
            D0[i][j] = (D0[i-1][j]*(D3[i] + D3[j])/2.0+ D0[i][j-1]*(D3[i] + D3[j])/2.0);
            if (i == j) 
                D0[i][j] += D0[i - 1][j - 1]*D3[i];          
        }
    }
    dist = D[la][lb] + D0[la][lb] ;

    for (i=0; i<=la; i++){
       free(D[i]);
       free(D0[i]);
       }
    free(D);
    free(D0);
    free(D3);

    return dist;
}

/*
 Evaluate the output of a teNN cell with reference vb, attention matrix At and activation matrix Ac.
*/
double teNNCell(const unsigned dim, const double *va, const unsigned int la, const double *R, const unsigned int lr, const double *At, const double *Ac, const double epsilon, const int corridor_radius) {
    // I:dim: dimension of multivariate time series va and vb
    // I:la: length of time series va
    // I:va: multivariate time series va
    // I:lr: length of reference time series R
    // I:R: multivariate time series vb
    // I:At: Attention matrix
    // I:Ac: Activation matrix
    // I:epsilon: parameter of the local kernel (exp(-At(i)*delta(va(i),vb(j))+epsilon)
    // I:corridor_radius: parameter defining the alignment search paths around the main diagonal
    // O:return value, teNNCell(va,vb,nu,epsilon)
    double dist;
    unsigned int i,j,k,beg,end,dim2=dim*dim;
    double nu, d, act;
    double factor =1.0/3.0;
    int l=la, _d=dim;
    if (lr>la)
        l=lr;
    
    double *D3 = (double *)calloc((l+1),sizeof(double));
    double **D = (double **)calloc((la+1),sizeof(double *));
    double **D0 = (double **)calloc((la+1),sizeof(double *));
    for (i=0; i<=la; i++){
       D[i] = (double *)calloc((lr+1),sizeof(double));
       D0[i] = (double *)calloc((lr+1),sizeof(double));
       }
    for(i=1; i<=la; i++){
        beg = _maxi(1,i-corridor_radius);
        end = _mini(lr, i+corridor_radius);
        for (j = beg; j <= end; j++) {
           act = Ac[(i-1)*lr+j-1];
           if (act>EPSILON){
             dist = 0;       
             for (k = 0; k < dim; k++){
               nu = At[(j-1)*dim+k];
               if (nu>0){
                    d = va[(i-1)*dim+k] - R[(j-1)*dim+k];
                    dist += (nu)*d*d;
                    }
               }
               if (dist>0)
               	  D[i][j] = factor*(exp(-dist/_d)) + epsilon;
               else
               	  D[i][j] = factor + epsilon;
             }
             	
             if(i==j)
                D0[i][j] = D[i][j];
        }
    }

    for(j=1; j<=l; j++) {
        if(j<=la&&j<=lr){
            dist=0;
            for(k=0; k<dim; k++){
                nu = At[(j-1)*dim+k];
                if (nu>0){
                    d = va[(j-1)*dim+k]-R[(j-1)*dim+k];
                    dist += nu*d*d;
                    }
                }
             if (dist>0)
             	D3[j] = factor*(exp(-dist/_d)) + epsilon;
             else
                D3[j] = factor + epsilon;
        }
    }

    D[0][0] = 1;
    D0[0][0] = 1;
    D3[0]=1;
    for (i = 1; i <= la; i++) {
        act = Ac[(i-1)*lr];
        D[i][0] = act*D[i-1][0]*D[i][1];
        D0[i][0] = act*D0[i-1][0]*(D3[i]);
    }
    for (j = 1; j <= lr; j++) {
        act = Ac[j-1];
        D[0][j] = act*D[0][j-1]*D[1][j];
        D0[0][j] = act*D0[0][j-1]*D3[j];
    }
 
    double dij, out=0;
    for (i = 1; i <= la; i++) {
        beg = _maxi(1,i-corridor_radius);
        end = _mini(lr, i+corridor_radius);
        for (j = beg; j <= end; j++) {
            act = Ac[(i-1)*lr+j-1];
            if (act>EPSILON){
              dij = D[i][j];
              D[i][j] = act*(D[i - 1][j - 1] + D[i - 1][j] + D[i][j - 1])* dij;
              D0[i][j] = act*(D0[i-1][j]+ D0[i][j-1])*(D3[i] + D3[j])/2.0;
              if (i == j) 
                  D0[i][j] += act*D0[i - 1][j - 1]*D3[i];          
              }
        }
    }

    out = D[la][lr] + D0[la][lr];
    
    for (i=0; i<=la; i++){
       free(D[i]);
       free(D0[i]);
       }
    free(D);
    free(D0);
    free(D3);
    
    return out;
}

/*
 Evaluate the alignment matrices of a teNN cell with reference vb, attention matrix At and activation matrix Ac.
*/
double ***teNNCell_mat(const unsigned int dim, const double *va, const unsigned int la, const double *R, const unsigned int lr, const double *At, const double *Ac, const double epsilon, const int corridor_radius, const int type) {
    // I:dim: dimension of multivariate time series va and vb
    // I:la: length of time series va
    // I:va: multivariate time series va
    // I:lb: length of time series vb
    // I:vb: multivariate time series vb
    // I:At: Attention matrix used to evaluate the local kernel (exp(-nu*delta(va(i),vb(j)),epsilon)
    // I:Ac: Activation matrix
    // I:epsilon: parameter of the local kernel (exp(-nu*delta(va(i),vb(j)),epsilon)
    // I:corridor_radius: parameter defining the alignment search paths around the main diagonal
    // I:type: 1:forward, -1:backward
    // O:return value, alignment matrix teNNCell_mat(va,vb,nu)
    double dist;
    unsigned int i,j,k,i1,j1, beg,end, dim2=dim*dim;
    double nu, d, act, eps;
    double deg=1.0;
    double factor =1.0/3.0;
    int l=la, _d=dim;
    if (lr>la)
        l=lr;
    int ai = 0;
    int aj = 0;
    if (type == -1){
    	ai = la+1;
    	aj = lr+1;
    	}
    double ***res = (double ***)calloc(2, sizeof(double **));
    double *D3 = (double *)calloc(l+1, sizeof(double));
    double **D = (double **)calloc(la+1, sizeof(double *));
    double **D0 = (double **)calloc(la+1, sizeof(double *));
    for (i=0; i<=la; i++){
       D[i] = (double *)calloc(lr+1, sizeof(double));
       D0[i] = (double *)calloc(lr+1, sizeof(double));
       }
    for(i=1; i<=la; i++){
        beg = _maxi(1,i-corridor_radius);
        end = _mini(lr, i+corridor_radius);
        i1 = ai + type*i;
        for (j = beg; j <= end; j++) {
            j1 = aj + type*j;
            act = Ac[(i1-1)*lr+j1-1];
            if (act>EPSILON){
              dist = 0;
              for (k = 0; k < dim; k++){
                 nu = At[(j1-1)*dim+k];
                 if (nu>0){
                   d = va[(i1-1)*dim+k] - R[(j1-1)*dim+k];
                   dist += nu*d*d;
                   }
                }
              if (dist>0)
              	D[i][j] = factor*(exp(-dist/_d)) + epsilon;
              else
              	D[i][j] = factor + epsilon;
            if(i==j)
                D0[i][j]=D[i][j];
          }
        }
    }
    
    for(j=1; j<=l; j++) {
        if(j<=la&&j<=lr){
            i1 = ai + type*j;
            j1 = aj + type*j;
            dist=0;
            for(k=0; k<dim; k++){   
                nu = At[(j1-1)*dim+k];
                if (nu>0){
                  d = va[(i1-1)*dim+k] - R[(j1-1)*dim+k];
                  dist += nu*d*d;
                  }
                }
              if (dist>0)
              	D3[j] = factor*(exp(-dist/_d)) + epsilon;
              else
              	D3[j] = factor + epsilon;
        }
    }
    D[0][0] = 1;
    D0[0][0] = 1;
    D3[0] = 1;
    for (i = 1; i <= la; i++) {
        i1 = ai + type*i;
        j1 = aj+type;
        act = Ac[(i1-1)*lr+j1-1];
        D[i][0] = act*D[i-1][0]*D[i][1];
        D0[i][0] = act*D0[i-1][0]*(D3[i]);
    }
    for (j = 1; j <= lr; j++) {
        i1 = ai+type;
        j1 = aj + type*j;
        act = Ac[(i1-1)*lr+j1-1];
        D[0][j] = act*D[0][j-1]*D[1][j];
        D0[0][j] = act*D0[0][j-1]*D3[j];
    }
 
    double dij;
    for (i = 1; i <= la; i++) {
        beg = _maxi(1,i-corridor_radius);
        end = _mini(lr, i+corridor_radius);
        i1 = ai + type*i;
        for (j = beg; j <= end; j++) {
            j1 = aj + type*j;
            act = Ac[(i1-1)*lr+j1-1];
            if (act>EPSILON){
              dij = D[i][j];
              D[i][j] = act*(D[i - 1][j - 1] + D[i - 1][j] + D[i][j - 1])* dij;
              D0[i][j] = act*(D0[i][j-1]+D0[i-1][j])*(D3[i] + D3[j])/2.0;
              if (i == j) 
                 D0[i][j] += act*D0[i - 1][j - 1] * D3[i]; 
              }          
        }
    }
    res[0] = D;
    res[1] = D0;
    
    free(D3);
    return res;
}


/*
 Evaluate the gradient vector, Gr, for the reference vector ref of a teNN cell with attention matrix At and activation matrix Ac, 
 when vector va is set as input vector.
*/
void *teNNCell_grad_re(const unsigned int dim, const double *va, const unsigned int la, double *ref, const unsigned int lr, 
	const double *At, const double *Ac, double epsilon, unsigned int corridor_radius, double *Gr) {
    // I:dim: dimension of multivariate time series va and vb
    // I:la: length of time series va
    // I:va: multivariate time series va
    // I:lr: length of reference time series ref
    // I:ref: multivariate reference time series vbref
    // I:Ac: Activation matrix
    // I:epsilon: parameter of the local kernel (exp(-nu*delta(va(i),vb(j)),epsilon)
    // I:corridor_radius: parameter defining the alignment search paths around the main diagonal
    // O:Gr: gradient vector. SHOULD BE ALLOCATED BEFORE INVOCATION. 
    // O:return value, void
	
    double *lexp3 = (double *)calloc(lr+2, sizeof(double));
    double *dk = (double *)malloc(dim*sizeof(double));
    double *d2k = (double *)malloc(dim*sizeof(double));
    double *dk3 = (double *)malloc(dim*sizeof(double));
    double *d2k3 = (double *)malloc(dim*sizeof(double));
    double factor =1.0/3.0;    
    double ***matk, ***matkr;
    double **U, **U0, **Ur, **U0r;
    int _d=dim;
      
    matk = teNNCell_mat(dim, va, la, ref, lr, At, Ac, epsilon, corridor_radius, 1);
    matkr = teNNCell_mat(dim, va, la, ref, lr, At, Ac, epsilon, corridor_radius, -1);

    U = matk[0]; U0 = matk[1];
    Ur = matkr[0]; U0r = matkr[1];

    double u, ur, u01, u10, u11, u01r, u10r, u11r, dnu, dact, dr, expo, sumk, act, Dact;
    unsigned int i,j,k, i1, j1, beg, end, dim2=dim*dim;
    
    lexp3[0]=1.0; lexp3[lr+1]=1.0;
    
    for(j=0; j<lr; j++){
        sumk = 0;
        for (k=0; k<dim; k++){
            dk[k] = va[j*dim+k]-ref[j*dim+k];
            d2k[k] = dk[k]*dk[k];
            sumk += (At[j*dim+k])*d2k[k];
	}
	lexp3[j+1] = exp(-sumk/_d); 
    }
    for(j=1; j<=lr; j++){
        j1 = lr-j+1;
        beg = _maxi(1,j-corridor_radius);
        end = _mini(la, j+corridor_radius);
        for (k=0; k<dim; k++){
            dk3[k] = va[(j-1)*dim+k]-ref[(j-1)*dim+k];
            d2k3[k] = dk3[k]*dk3[k];
            }
        for(i=beg; i<=end; i++){
            act = Ac[(i-1)*lr+j-1];
            i1 = la-i+1;
            sumk = 0;
            for (k=0; k<dim; k++){
                dk[k] = va[(i-1)*dim+k]-ref[(j-1)*dim+k];
                d2k[k] = dk[k]*dk[k];
                sumk += (At[(j-1)*dim+k])*d2k[k];
                }  
            
            expo = exp(-sumk/_d); 
            
            u = U[i-1][j] + U[i][j-1] + U[i-1][j-1];  
            ur = Ur[i1][j1-1] + Ur[i1-1][j1] + Ur[i1-1][j1-1]; 
            u = u*ur;
            
            u01 = U0[i-1][j]/2.0; u10 = U0[i][j-1]/2.0; u11=0;
            u01r = U0r[i1-1][j]/2.0; u10r = U0r[i1][j1-1]/2.0; u11r=0;
            if (i==j){ 
               u11 = U0[i-1][j-1];
               u11r = U0r[i1-1][j1-1]; 
               }
             
                  
            for(k=0; k<dim; k++){   
            	   dr = 2*factor*act*At[(j-1)*dim+k]*dk[k]/_d*expo*u;
            	   dr += 2*factor*act*At[(j-1)*dim+k]*dk3[k]/_d*(u01+u10+u11)*lexp3[j]*(u01r+u10r+u11r);
                   if (!(isnan(dr)))
                   	Gr[(j-1)*dim+k] += dr;
           } //for k
        } // for i
    }// for j


    free(dk);
    free(d2k);
    free(dk3);
    free(d2k3);
    free(lexp3);

    for (i=0; i<=la; i++){
       free(U[i]); free(U0[i]);
       free(Ur[i]); free(U0r[i]);
       }
     
    free(U); free(Ur);
    free(U0); free(U0r);
    free(matk);
    free(matkr);

}

/*
 Evaluate the gradient vectors matrices At ,Ac and Gr for the reference vector ref of a teNN cell with attention matrix At and activation matrix Ac, 
 when vector va is set as input vector.
*/
void teNNCell_grads(const unsigned int dim, const double *va, const unsigned int la, double *ref, const unsigned int lr, const double *At, const double *Ac, 
		double epsilon, int corridor_radius, double *Gr, double *Gatt, double *Gact, double scalar)
{
    // I:dim: dimension of multivariate time series va and vb
    // I:la: length of time series va
    // I:va: multivariate time series va
    // I:lr: length of reference time series ref
    // I:ref: multivariate reference time series vbref
    // I:Ac: Activation matrix
    // I:epsilon: parameter of the local kernel (exp(-nu*delta(va(i),vb(j)),epsilon)
    // I:corridor_radius: parameter defining the alignment search paths around the main diagonal
    // O:Gr: gradient vector. SHOULD BE ALLOCATED BEFORE INVOCATION. 
    // O:return value, void

    double *lexp3 = (double *)calloc(lr+2, sizeof(double));
    double *dk = (double *)malloc(dim*sizeof(double));
    double *d2k = (double *)malloc(dim*sizeof(double));
    double *dk3 = (double *)malloc(dim*sizeof(double));
    double *d2k3 = (double *)malloc(dim*sizeof(double));
    double factor =1.0/3.0;    
    double ***matk, ***matkr;
    double **U, **U0, **Ur, **U0r;
    double abs_scalar = fabs(scalar);
    int _d=dim;
    
    matk = teNNCell_mat(dim, va, la, ref, lr, At, Ac, epsilon, corridor_radius, 1);
    matkr = teNNCell_mat(dim, va, la, ref, lr, At, Ac, epsilon, corridor_radius, -1);
    U = matk[0]; U0 = matk[1];
    Ur = matkr[0]; U0r = matkr[1];

    double u, ur, u01, u10, u11, u01r, u10r, u11r, dnu, dact, dr, expo, sumk, act, Dact;
    unsigned int i,j,k, i1, j1, beg, end, dim2=dim*dim;
    
    lexp3[0]=1.0; lexp3[lr+1]=1.0;
    
    for(j=0; j<lr; j++){
        sumk = 0;
        for (k=0; k<dim; k++){
            dk[k] = va[j*dim+k]-ref[j*dim+k];
            d2k[k] = dk[k]*dk[k];
            sumk += (At[j*dim+k])*d2k[k];
	}
	lexp3[j+1] = exp(-sumk/_d); 
    }
    for(j=1; j<=lr; j++){
        j1 = lr-j+1;
        beg = _maxi(1,j-corridor_radius);
        end = _mini(la, j+corridor_radius);
        for (k=0; k<dim; k++){
            dk3[k] = va[(j-1)*dim+k]-ref[(j-1)*dim+k];
            d2k3[k] = dk3[k]*dk3[k];
            }
        for(i=beg; i<=end; i++){
            act = Ac[(i-1)*lr+j-1];
            i1 = la-i+1;
            sumk = 0;
            for (k=0; k<dim; k++){
                dk[k] = va[(i-1)*dim+k]-ref[(j-1)*dim+k];
                d2k[k] = dk[k]*dk[k];
                sumk += (At[(j-1)*dim+k])*d2k[k];
                }  
            
            expo = exp(-sumk/_d); 
            
            u = U[i-1][j] + U[i][j-1] + U[i-1][j-1];  
            ur = Ur[i1][j1-1] + Ur[i1-1][j1] + Ur[i1-1][j1-1]; 
            u = u*ur;
            
            u01 = U0[i-1][j]/2.0; u10 = U0[i][j-1]/2.0; u11=0;
            u01r = U0r[i1-1][j]/2.0; u10r = U0r[i1][j1-1]/2.0; u11r=0;
            if (i==j){ 
               u11 = U0[i-1][j-1];
               u11r = U0r[i1-1][j1-1]; 
               }

            dact = factor*(expo*u + (u01+u10)*(lexp3[j]+lexp3[i])*(u01r+u10r)+u11*lexp3[j]*u11r);//  -1e-10;
            if (!isnan(dact) && isfinite(dact))
                  Gact[((i-1)*lr+j-1)] += dact*scalar;

            for(k=0; k<dim; k++){   
            	   dnu = -factor*act*d2k[k]/_d*expo*u;
            	   dnu -= factor*act*d2k3[k]/_d*(u01+u10+u11)*lexp3[j]*(u01r+u10r+u11r);
            	   dr = 2*factor*act*At[(j-1)*dim+k]*dk[k]/_d*expo*u;
            	   dr += 2*factor*act*At[(j-1)*dim+k]*dk3[k]/_d*(u01+u10+u11)*lexp3[j]*(u01r+u10r+u11r);

                   if (!(isnan(dnu)) && isfinite(dact))
                   	Gatt[(j-1)*dim+k] += dnu*scalar;
                   if (!(isnan(dr)) && isfinite(dact))
                   	Gr[(j-1)*dim+k] +=  dr*scalar;
           } //for k
        } // for i
    }// for j


    free(dk);
    free(d2k);
    free(dk3);
    free(d2k3);
    free(lexp3);

    for (i=0; i<=la; i++){
       free(U[i]); free(U0[i]);
       free(Ur[i]); free(U0r[i]);
       }
     
    free(U); free(Ur);
    free(U0); free(U0r);
    free(matk);
    free(matkr);

}
 
/*
 Evaluate the KDTW barycenter of a set of time series
*/
double *barycenter(int lab, const unsigned int NX, const unsigned L, const unsigned dim, const double *X, double *IB, double *At, double *Ac, 
	double epsilon, int corridor_radius, double eta, unsigned int niter) {
    // I:lab: label attached to each time series of set X
    // I:NX: cardinal of set X
    // I:L: length of the time series
    // I:dim: dimension of multivariate time series va and vb
    // I:X: set of time series matrix of size NX x L x dim (presented as a 1D vector)
    // I/O:IB: multivariate initial estimate of the barycenter time series
    // I:At: Attention matrix
    // I:Ac: Activation matrix
    // I:epsilon: parameter of the local kernel (exp(-nu*delta(va(i),vb(j)),epsilon)
    // I:corridor_radius: parameter defining the alignment search paths around the main diagonal
    // I:eta: relaxation parameter
    // O:return value, loss function.

int LDIM = L*dim;
int LL = L*L;
double *Gr = (double *)malloc(LDIM*sizeof(double));
double *B = (double *)malloc(LDIM*sizeof(double));
double *lloss = (double *)malloc(niter*sizeof(double));
int *I = (int *)malloc(NX*sizeof(int));
double *va;
double nrm, loss, d;
double lossmax = -1e300;
int degree=2;

cpy_1Darray(B, IB, LDIM*sizeof(double));

for(int n=0; n<NX; n++)
   I[n]=n;

for(int it=0; it<niter; it++){
   shuffle(I,NX,it==0);
   reset_1DDarray(Gr, LDIM, 0.0);
   loss = 0;   
   for(int n=0; n<_min(NX,100); n++){
      va = (double *)(X+I[n]*LDIM);
      teNNCell_grad_re(dim, va, L, B, L, At, Ac,  epsilon, corridor_radius, Gr);
      loss += teNNCell(dim, va, L, B, L, At, Ac, epsilon, corridor_radius);
   }
   lloss[it] = loss;
   if (loss<lossmax){
     eta /=1.1;
     }
   else{
      lossmax = loss;
      cpy_1Darray(IB,B, LDIM*sizeof(double));
   }
      
   fprintf(stderr,"lab:%d it:%d, NX:%d, loss barycenter:%.6e\r   ", lab, it, NX, loss);

   nrm = Norm(Gr,L,dim,degree);
   add_1DDarray(B,Gr, LDIM, eta/nrm);
   if (eta<1e-15)
      break;
}
fprintf(stderr,"\n");
free(Gr);
free(B);
free(I);
return(lloss);

}

int eval_errors(int *pred, const int *y, int sz){
int err = 0;
for(int i=0; i<sz; i++){
  if (pred[i] != y[i]){
    err++;
    }
  }
return err;
}

/*
 predict probability function
*/
double *predict_proba(const unsigned int NX, unsigned int L, unsigned int dim, const double *X, unsigned int NR, int *yR, double *R, double *At, double *Ac, double epsilon, int corridor_radius, const unsigned int NC, int* yc){
    // I:NX: cardinal of set X
    // I:L: length of the time series
    // I:dim: dimension of multivariate time series va and vb
    // I:X: set of time series matrix of size NX x L x dim (presented as a 1D vector)
    // I:NR: number of reference time series
    // I:yR: labels associated to the reference time series
    // I:R: referrence time series matrix
    // I:At: Attention matrix
    // I:Ac: Activation matrix
    // I:epsilon: parameter of the local kernel (exp(-nu*delta(va(i),vb(j)),epsilon)
    // I:corridor_radius: parameter defining the alignment search paths around the main diagonal
    // I:NC: number of distinct categories
    // I:yc: list of categories
    // O:return value: array of probabilities.

double *out_probas = (double *)malloc(NX*NC*sizeof(double));
double *Lsim = (double *)malloc(NR*sizeof(double));
double *vx, s, sumSim;
double *r, *at, *ac;
int best_ref, *bestR, err;
int LDIM=L*dim, LL=L*L;
err=0;
for(int n=0; n<NX; n++){
  vx = (double *)(X+n*LDIM);
  sumSim = 0;
  for(int nr=0; nr<NR; nr++){
    s = teNNCell(dim, vx, L, (double *)(R+nr*LDIM), L, (double *)(At+nr*LDIM), (double *)(Ac+nr*LL), epsilon, corridor_radius);
    if (isnan(s))
      s=0.0;
    Lsim[nr] = s + 1e-292;
    sumSim += s;
    }

  bestR = getBestRef(NC, yc, NR, yR, Lsim, &best_ref);
  for(int c=0; c<NC; c++){
     *(out_probas+n*NC+c) = Lsim[c]/sumSim;
     }
  free(bestR);
  }
  return out_probas;
}

/*
 predict labels
*/
int *predict(const unsigned int NX, unsigned int L, unsigned int dim, const double *X, unsigned int NR,  int *yR, double *R, double *At, double *Ac, double epsilon, 
	int corridor_radius, const unsigned int NC, int* yc){
    // I:NX: cardinal of set X
    // I:L: length of the time series
    // I:dim: dimension of multivariate time series va and vb
    // I:X: set of time series matrix of size NX x L x dim (presented as a 1D vector)
    // I:NR: number of reference time series
    // I:yR: labels associated to the reference time series
    // I:R: reference time series matrix
    // I:At: Attention matrix
    // I:Ac: Activation matrix
    // I:epsilon: parameter of the local kernel (exp(-nu*delta(va(i),vb(j)),epsilon)
    // I:corridor_radius: parameter defining the alignment search paths around the main diagonal
    // I:NC: number of distinct categories
    // I:yc: list of categories
    // O:return value: array of predicted labels.

int *ypred = (int *)malloc(NX*sizeof(int));
double *Lsim = (double *)malloc(NR*sizeof(double));
double *vx, s, sumSim;
double *r, *at, *ac;
int best_ref, *bestR, err;
int LDIM=L*dim, LL=L*L;
err=0;
for(int n=0; n<NX; n++){
  vx = (double *)(X+n*LDIM);
  sumSim = 0;
  for(int nr=0; nr<NR; nr++){
    s = teNNCell(dim, vx, L, (double *)(R+nr*LDIM), L, (double *)(At+nr*LDIM), (double *)(Ac+nr*LL), epsilon, corridor_radius);
    if (isnan(s))
      s=0.0;
    Lsim[nr] = s+1e-292;
    }
  
  bestR = getBestRef(NC, yc, NR, yR, Lsim, &best_ref);
  ypred[n] = yR[best_ref];
  free(bestR);
  }
  return ypred;
}
 
/*
 fit function
*/                 
double *fit(const unsigned int NX, const unsigned int L, const unsigned dim, const double *X, const int *y, double epsilon, int corridor_radius, 
	const unsigned int NC, int *yc, const unsigned int NR, int *yR,
	double *R, double *At, double *Ac, double *R_lm, double *At_lm, double *Ac_lm, double *R_ml, double *At_ml, double *Ac_ml,
	double lambda_At, double lambda_Ac, double eta, unsigned int nepoch, double batch_size, int probe, int OAT, int OAC, int ORF,
	const int NV, double *Xvalid, int *yvalid, int verbose) {	
    // I:NX: cardinal of set X (training set)
    // I:L: length of the time series
    // I:dim: dimension of multivariate time series va and vb
    // I:X: set of time series matrix of size NX x L x dim (presented as a 1D vector)
    // I:y: set of labels associated to time series in X (y_train)
    // I:epsilon: parameter of the local kernel (exp(-nu*delta(va(i),vb(j)),epsilon)
    // I:corridor_radius: parameter defining the alignment search paths around the main diagonal
    // I:NC: number of distinct categories
    // I:yc: list of categories
    // I:NR: number of reference time series
    // I:yR: labels associated to the reference time series
    // I/O:R: reference time series matrix
    // I/O:At: Attention matrix
    // I/O:Ac: Activation matrix
    // I/O:R_lm: reference time series matrix (last minimum error reached)
    // I/O:At_lm: Attention matrix (last minimum error reached)
    // I/O:Ac_lm: Activation matrix (last minimum error reached)
    // I/O:R_ml: reference time series matrix (last minimum loss reached)
    // I/O:At_ml: Attention matrix (last minimum loss reached)
    // I/O:Ac_ml: Activation matrix (last minimum loss reached)
    // I:eta: relaxation parameter
    // I:nepoch: number of epoch
    // I:batch_size: size of the bacthes used for stochastic gradient descent
    // I:probe: partial display of training status
    // I:OAT: optimize the attention matrices if equal to 1, do not optimize these matrices if different from 1
    // I:OAC: optimize the activation matrices if equal to 1, do not optimize these matrices if different from 1
    // I:OR: optimize the reference time series if equal to 1, do not optimize these references if different from 1
    // I:NV: cardinal of validation set Xvalid
    // I:Xvalid: set of time series matrix of size NV x L x dim (presented as a 1D vector)
    // I:yvalid: set of labels associated to time series in Xvalid
    // I:verbose: display some log info if verbose is not set to 0
    // O:return value: array of predicted labels.
int LDIM=L*dim, LL=L*L;	
double *lloss=(double *)malloc((nepoch+1)*sizeof(double));
double *R_p = (double *)malloc(NR*LDIM*sizeof(double));
double *At_p = (double *)malloc(NR*LDIM*sizeof(double));
double *Ac_p = (double *)malloc(NR*LL*sizeof(double));
double *GR0 = (double *)malloc(NR*LDIM*sizeof(double));
double *GAt0 = (double *)malloc(NR*LDIM*sizeof(double));
double *GAc0 = (double *)malloc(NR*LL*sizeof(double));
double *Lsim = (double *)malloc(NR*sizeof(double));
double *vx;

int *I = (int *)malloc(NX*sizeof(int));
int *bestR;
double binf_r=BINF, bsup_r=BSUP, epsiMax=1e300, epsi;
double sz, nrm, nrm_gr, nrm_gAt, nrm_gAc, _epsip=1e-300;
int nchunk;
int n0, progress=0, _progress=1;
int *pred_cur, *pred_ml, *pred_lm, *pred_valid;

if(verbose)
   fprintf(stdout, "NC:%d, NR:%d, NX:%d, dim:%d, L:%d, epsilon:%.2e, corridor_radius:%d, lambda_At:%.2e, lambda_Ac:%.2e,  eta:%.2e, nepoch:%d, batch_size:%lf, probe:%d\n", 
	NC,NR,NX, dim, L, epsilon, corridor_radius, lambda_At, lambda_Ac, eta, nepoch, batch_size, probe);

_NEPOCH = nepoch; 

signal(SIGINT, intHandler);


if (batch_size <= 0){
     nchunk =1;
     batch_size = NX;
}
else if (batch_size < 1){
     sz = round(NX*batch_size);
     nchunk = (int)ceil(NX/sz);
     batch_size = (int)sz+1;
}
else{
     nchunk = (int)ceil(NX/batch_size);
     if (nchunk == 0)
          nchunk = 1;
     }
eta /= nchunk;

if(verbose)
  fprintf(stdout, "NX:%d, BATCH SIZE: %d; NCHUNK: %d => CHUNK SIZE: %d eta: %lf\n",NX, (int)batch_size, nchunk, (int)ceil(NX/nchunk), eta);

double s, w, sumSim, loss, lossp=0, minloss=1e300, minloss_lm=1e300, loss_cce, cumul_sumSim, eps;
int ncat = NR, c, best_ref, closest_neg_ref, best_pos_ref, err, errp, minerr, err_lm, err_lmt, err_ml, err_cur, err_valid;

errp = NX;
minerr = NX;
err_lm = 0;
err_lmt = 0;
lloss[0] = nepoch;

eps = 1e-10;  
 
for(int epoch=1; epoch<=nepoch; epoch++){
    cumul_sumSim = 0;
    
    shuffle(I,NX,(epoch==1));
    
    err = 0;
    nrm_gr=0; nrm_gAt=0; nrm_gAc=0;
    loss = 0;
    loss_cce = 0;

    n0 = 0;

       cpy_1Darray(R_p, R, NR*LDIM*SZDBL);
       cpy_1Darray(At_p, At, NR*LDIM*SZDBL);
       cpy_1Darray(Ac_p, Ac, NR*LL*SZDBL);

      
      for(int nck=0; nck<nchunk; nck++){
          reset_1DDarray(GR0, NR*LDIM, 0.0);
          reset_1DDarray(GAt0, NR*LDIM, 0.0);
          reset_1DDarray(GAc0, NR*LL, 0.0);

          for(int n=nck*batch_size; n<_min(NX,(nck+1)*batch_size); n++){
            vx = (double *)(X+I[n]*LDIM);
            
            for(int nr=0; nr<NR; nr++){
              s = teNNCell(dim, vx, L, (double *)(R+nr*LDIM), L, (double *)(At+nr*LDIM), (double *)(Ac+nr*LL), epsilon, corridor_radius);
              if (isnan(s)){
                 s = 1e-300;
                 }
              else if (!isfinite(s) || s >1e300)
                 s = 1e300;

              Lsim[nr] = s+eps;
              }

            bestR = getBestRef(NC, yc, NR, yR, Lsim, &best_ref);
            closest_neg_ref = getBestNegRef(y[I[n]], NR, yR, Lsim);
	    best_pos_ref = getBestPosRef(y[I[n]], NR, yR, Lsim);
	    
	    sumSim = 1e-300;
            for(int nc=0; nc<NC; nc++){
               c = bestR[nc];
               sumSim += Lsim[c];
	       }
            cumul_sumSim += sumSim;
            n0++;
            
            if(yR[best_pos_ref] != y[I[n]])
               err++;

            for(int nc=0; nc<NC; nc++){
              c = bestR[nc];
                 if(yR[c] != y[I[n]]){
                      teNNCell_grads(dim, vx, L, (double *)(R+c*LDIM), L, (double *)(At+c*LDIM), (double *)(Ac+c*LL), epsilon, corridor_radius, 
               	  				(double *)(GR0+c*LDIM), (double *)(GAt0+c*LDIM), (double *)(GAc0+c*LL), -1.0/sumSim);
                 }
                 else{
                   loss_cce -= log(Lsim[c]/sumSim+1e-300);
                   teNNCell_grads(dim, vx, L, (double *)(R+c*LDIM), L, (double *)(At+c*LDIM), (double *)(Ac+c*LL), epsilon, corridor_radius, 
               	   				(double *)(GR0+c*LDIM), (double *)(GAt0+c*LDIM), (double *)(GAc0+c*LL), (1.0/Lsim[c] -1.0/sumSim));		
                 }
           } //for c
         free(bestR);
         } //for n (batch)

     double nrm, nrmR, nrmAt, nrmAc;
     
     loss = loss_cce;
     for(int nr=0; nr<NR; nr++){
          nrm = 0;
          if(ORF){
            nan2zero_bounded_1DDarray(GR0+nr*LDIM, LDIM);
            nrmR = norm2_1DDarray(GR0+nr*LDIM, LDIM);
            nrm += nrmR;
            }
          nan2zero_bounded_1DDarray(GAt0+nr*LDIM, LDIM);
          nrmAt = norm2_1DDarray(GAt0+nr*LDIM, LDIM);
          nrm += nrmAt;
          nan2zero_bounded_1DDarray(GAc0+nr*LL, LL); 
          nrmAc = norm2_1DDarray(GAc0+nr*LL, LL);
          nrm += nrmAc;

        if(ORF){
           adapt_1DDarray((double *)(R+nr*LDIM), (double *)(GR0+nr*LDIM), LDIM, eta, 0.0, binf_r, bsup_r, &nrm);
           nrm_gr += nrmR;
           }
        if(OAT){
          loss += adapt_1DDarray((double *)(At+nr*LDIM), (double *)(GAt0+nr*LDIM), LDIM, eta, lambda_At, 0.0, BSUP, &nrm)/nchunk;
          nrm_gAt += nrmAt;
          }
        if(OAC){
          loss += adapt_1DDarray((double *)(Ac+nr*LL), (double *)(GAc0+nr*LL), LL,  eta, lambda_Ac, 0.0, 1.0, &nrm)/nchunk;
          nrm_gAc += nrmAc;
          }
        }
    }// for nchunk
    eps = cumul_sumSim/NX/NC*1e-8;

    pred_cur = predict(NX, L, dim, X, NR, yR, R, At, Ac, epsilon, corridor_radius, NC, yc);
    err = eval_errors(pred_cur, y, NX);

    if ((err<=minerr && loss<=lossp) || loss<minloss || (err <= minerr && loss <= minloss)){
        _progress = 1;
        if (loss<minloss || err<minerr)
            progress = 1;  
        if (loss<lossp && eta < 1.)
            eta *= 1.01;
        }
    else{
       if (lossp<loss){
          _progress = 0;
          eta /= 1.01;
          }
        else
          _progress = 1;
       }
       
    lloss[epoch] = loss;
    if (minloss>=loss && epoch>=1){
       minloss = loss;
       cpy_1Darray(R_ml, R, NR*LDIM*SZDBL);
       cpy_1Darray(At_ml, At, NR*LDIM*SZDBL);
       cpy_1Darray(Ac_ml, Ac, NR*LL*SZDBL);
       err_ml = err;
    }

   if ((minerr>=err && loss<=minloss_lm)){
       minloss_lm = loss;
       cpy_1Darray(R_lm, R, NR*LDIM*SZDBL);
       cpy_1Darray(At_lm, At, NR*LDIM*SZDBL);
       cpy_1Darray(Ac_lm, Ac, NR*LL*SZDBL);
       err_lmt = err;
    }

    errp = err;
    if (minerr>err)
       minerr = err;
 
    lossp = loss; 
    
    if(epoch%probe == 0 && epoch>2){
       if(progress == 1){
          progress = 0;
          }

       pred_ml =  predict(NX, L, dim, X, NR, yR, R_ml, At_ml, Ac_ml, epsilon, corridor_radius, NC, yc);
       pred_lm =  predict(NX, L, dim, X, NR, yR, R_lm, At_lm, Ac_lm, epsilon, corridor_radius, NC, yc);
       if(verbose)
       	 fprintf(stdout,"epch:%d sumSim:%.4e minErr:%d eta:%.2e loss:%.4e minloss:%.4e SNrm:[R%.2e At%.2e Ac%.2e] mxAt:%.2e mnAc:%.2e Spty[At:%.2e Ac:%.2e] Train cur:%d lm:%d ml:%d ", 
                epoch, sumSim, minerr, eta, loss, minloss,
                nrm_gr, nrm_gAt, nrm_gAc, 
                max_1DDarray(At,NR*LDIM), min_1DDarray(Ac,NR*LL),
                eval_sparsity_1Darray(At, NR*LDIM, 1.e-10), eval_sparsity_1Darray(Ac, NR*LL, 1.e-10),
                err, eval_errors(pred_lm, y, NX), eval_errors(pred_ml, y, NX));

       if (verbose && err_lmt != eval_errors(pred_lm, y, NX))
           fprintf(stdout,"!!!!!!!!!! ERR last min %d\n", err_lmt);
       free(pred_cur); free(pred_ml); free(pred_lm); 
       if(NV >0 && verbose){ 
          pred_valid = predict(NV, L, dim, Xvalid, NR, yR, R, At, Ac, epsilon, corridor_radius, NC, yc);
          err_cur = eval_errors(pred_valid, yvalid, NV);     
	  free(pred_valid);
          pred_valid = predict(NV, L, dim, Xvalid, NR, yR, R_lm, At_lm, Ac_lm, epsilon, corridor_radius, NC, yc);
          err_lm = eval_errors(pred_valid, yvalid, NV);       
          free(pred_valid);
          pred_valid = predict(NV, L, dim, Xvalid, NR, yR, R_ml, At_ml, Ac_ml, epsilon, corridor_radius, NC, yc);
          err_ml = eval_errors(pred_valid, yvalid, NV);     
          free(pred_valid);   
          
          fprintf(stdout,"| Valid curr:%d AC:%.3lf%% |", err_cur, 100.0*(1.0 - (double)err_cur/(double)NV));
          fprintf(stdout,"| Valid lm:%d ACC:%.3lf%% |", err_lm, 100.0*(1.0 - (double)err_lm/(double)NV));
          fprintf(stdout,"| Valid ml:%d ACC:%.3lf%%\n", err_ml, 100.0*(1.0 - (double)err_ml/(double)NV));
          }
       fflush(stdout); 
    }

    if (eta < 1e-10 || loss<=1e-8){
        keepRunning=0;
        if(verbose)  
            fprintf(stdout,"OUT WITH MINLOSS=%.2e eta=%.2e %d\n", minloss, eta, NX);
        }
    if(keepRunning==0){
       if(NV >0){
          pred_valid = predict(NV, L, dim, Xvalid, NR, yR, R_ml, At_ml, Ac_ml, epsilon, corridor_radius, NC, yc);
          err_cur = eval_errors(pred_valid, yvalid, NV);        
          free(pred_valid);
          pred_valid = predict(NV, L, dim, Xvalid, NR, yR, R_lm, At_lm, Ac_lm, epsilon, corridor_radius, NC, yc);
          err_lm = eval_errors(pred_valid, yvalid, NV);        
          free(pred_valid);
          pred_valid = predict(NV, L, dim, Xvalid, NR, yR, R_ml, At_ml, Ac_ml, epsilon, corridor_radius, NC, yc);
          err_ml = eval_errors(pred_valid, yvalid, NV);        
          free(pred_valid);
          if(verbose){
              fprintf(stdout,"Valid running:%d ACC:%.3lf%%\n", err_cur, 100.0*(1.0 - (double)err_cur/(double)NV));
              fprintf(stdout,"|| Valid lm:%d ACC:%.3lf%% |", err_lm, 100.0*(1.0 - (double)err_lm/(double)NV));
              fprintf(stdout,"| Valid ml:%d ACC:%.3lf%%\n", err_ml, 100.0*(1.0 - (double)err_ml/(double)NV));
          }
         }
       lloss[0] = epoch;
       break;
       }

}

if(verbose){
   fprintf(stdout," sparsity At(lm):%lf / sparsity Ac(lm):%.2e\n", eval_sparsity_1Darray(At_lm, NR*LDIM, 1.e-10), eval_sparsity_1Darray(Ac_lm, NR*LL, 1.e-10));
   fprintf(stdout," sparsity At(ml):%lf / sparsity Ac(ml):%.2e\n", eval_sparsity_1Darray(At_ml, NR*LDIM, 1.e-10), eval_sparsity_1Darray(Ac_ml, NR*LL, 1.e-10));
   fprintf(stdout," sparsity At(rn):%lf / sparsity Ac(rn):%.2e\n", eval_sparsity_1Darray(At, NR*LDIM, 1.e-10), eval_sparsity_1Darray(Ac, NR*LL, 1.e-10));
}

free(Lsim); free(I); 
free(GR0); free(GAt0); free(GAc0);
free(R_p); free(At_p); free(Ac_p);

if(_FLOG!=NULL)
   fclose(_FLOG);
_FLOG=NULL;

return lloss;
}




