import numpy as np
from boundary_maps import boundary_operators 


def matrix_rank(A):
    ''''Computes the rank of a matrix'''
    if A.size == 0:
        return 0
    else:
        return np.linalg.matrix_rank(A,tol = 1e-10)

def betti_number(f,g):#kerf/img
    '''Computes the betti number of a given boundary operator'''
    f = np.array(f)
    g = np.array(g)
    rk_f = matrix_rank(f)
    
    dim_kerf = f.shape[1]-rk_f
    #print("f",f.size,f.shape,dim_kerf)
    rk_g = matrix_rank(g)
    #print("g",g.size,rk_g)
    return dim_kerf-rk_g



def mat_multiply(bnd_op, dim, q):
    '''Computes the matrix multiplication of the boundary operator'''
    # Get the initial matrix f which is an array
    f = bnd_op.get(dim-q+1)

    # Perform the matrix multiplications
    for i in range(1, q):
        next_matrix = bnd_op.get(dim+i-q+1)

        # Perform the multiplication
        f = f @ next_matrix

    return f

def calculate_betti_laplacian(bnd_op,N,q):
    '''Computes betti and laplacian information for the given boundary operator,
    Inputs: boundary operator(as a class onject), N(Mayer degree), q(Mayer order <N)'''
    #ker d^q/imd^(n-q) for q<n

    dim = bnd_op.max_dim
  
    betti=[]
    gamma_min=[]
    gamma_max = []
    gamma_mean = []
    gamma_std = []

    for i in range(dim+1):
        
        f = mat_multiply(bnd_op,i,q)
        #f = d^q
        g = mat_multiply(bnd_op,i+N-q,N-q)
        # g = d^(n-q)
        

    
        f = np.matrix(f)
        g = np.matrix(g)
      
        lap = f.H@f+g@g.H
    
        eigvals = np.linalg.eigvalsh(lap)
   
        betti.append(np.count_nonzero(eigvals.real <=1e-6)) 

        positive_eigvals = eigvals.real[eigvals.real>1e-6]
        
    return betti,gamma_min,gamma_max,gamma_mean,gamma_std

def betti_laplacian(X,N,max_dim=2):
    '''Computes betti and laplacian information for the given complex=X for each Mayer order q<N'''
    Bettis=[]

    bnd_op = boundary_operators(X,N)
    for q in range(1,N):
        betti, gmin,gmax,gmean,gstd = calculate_betti_laplacian(bnd_op,N,q)
        Bettis.append(betti)

    return Bettis 

def calculate2(path_complex, distances,N):
    '''Computes betti numbers of pathcomplex for each distance in the list of distances'''
     B = {}

     pre_complex_num = 0
     path = path_complex
     #B stores betti numbers for each distance so len(B)=len(distances)
     for i,distance in enumerate(distances):
         
         cur_complex = path.get_complex(distance)
         if len(cur_complex) == pre_complex_num:
             B[distance] = B[distances[i-1]]
         else:
            
             B[distance] = betti_laplacian(cur_complex,N)   
         pre_complex_num = len(cur_complex)
     for i in range(N-1):
         b0=[]
         b1=[]
         betti = []

         for distance in distances:
             betti = B[distance][i]
             b0.append(betti[0] if len(betti)>0 else 0)
             b1.append(betti[1] if len(betti)>1 else 0)
     return b0 , b1

