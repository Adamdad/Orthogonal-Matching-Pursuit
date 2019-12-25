import numpy as np
import random
import scipy

def define_A(M,N):
    '''Define the measurement matrix
    :args
    M: number of row
    N: number of column
    :returns
    A: M x N matrix with normalized columns drawn from N(0,1)
    '''
    A = []
    for i in range(N):
        col = np.random.randn(M)
        col /= norm2(col)
        
        A.append(col)
    A = np.array(A).T
    return A

def define_x(s,N):
    '''Define the real signal
    :args
    s: random support of cardinality
    N: size of x
    :return 
    x: real signal
    '''
    # Uniformly sample the index

    index = random.sample(range(N),s)
    x = np.zeros(N)
    for idx in index:
        # For each index, draw a number from [-10,-1]||[1,10]
        r = np.random.rand()
        if r >0.5:
            x[idx]=(r-0.5)*18+1
        else:
            x[idx]=(r-0.5)*18-1
    return x.T,index

def OMP(A,y,N,stop=np.infty,r_thresh=0.01):
    '''Orthogonal Matching pursuit algorithm
    :args
    A: measurement matrix
    y: 
    '''
    r = y
    x_pre = np.zeros(N)
    Lambdas = []
    i = 0
    # Control stop interation with norm thresh or sparsity
    while norm2(r)>r_thresh and i<stop:
       
        # Compute the score of each atoms
        scores = A.T.dot(r)
        
        # Select the atom with the max score
        Lambda = np.argmax(abs(scores))
        # print(Lambda)
        Lambdas.append(Lambda)
        
        # All selected atoms form a basis
        basis = A[:,Lambdas]

        # Least square solution for y=Ax
        x_pre[Lambdas] = np.linalg.inv(np.dot(basis.T,basis)).dot(basis.T).dot(y)
        
        # Compute the residual
        r = y - A.dot(x_pre)
        
        i += 1
    return x_pre.T,Lambdas

def norm2(x):
    return np.sqrt(np.sum([i**2 for i in x]))

def norm1(x):
    return np.sum([abs(i) for i in x])

def Normalized_Error(x,x_pre):
    return norm2(x-x_pre)/norm2(x)
        
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def dct2(a):
    return scipy.fftpack.dct( scipy.fftpack.dct( a, axis=0, norm='ortho' ), axis=1, norm='ortho' )

def idct2(a):
    return scipy.fftpack.idct( scipy.fftpack.idct( a, axis=0 , norm='ortho'), axis=1 , norm='ortho')

