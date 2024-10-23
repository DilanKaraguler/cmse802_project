import numpy as np



def matrix_rank(A):
    if A.size == 0:
        return 0
    else:
        return np.linalg.matrix_rank(A,tol = 1e-10)

def betti_number(f,g):#kerf/img
    f = np.array(f)
    g = np.array(g)
    rk_f = matrix_rank(f)
    
    dim_kerf = f.shape[1]-rk_f
    #print("f",f.size,f.shape,dim_kerf)
    rk_g = matrix_rank(g)
    #print("g",g.size,rk_g)
    return dim_kerf-rk_g



def mat_multiply(bnd_op, dim, q):
    # Get the initial matrix f which is an array
    f = bnd_op.get(dim-q+1)

    # Print the initial shape of f
    #print(f"Initial f shape: {f.shape}")

    # Perform the matrix multiplications
    for i in range(1, q):
        next_matrix = bnd_op.get(dim+i-q+1)
        # Print the shape of the next matrix before multiplication
        #print(f"Multiplying f of shape {f.shape} with next_matrix of shape {next_matrix.shape}")

        # Perform the multiplication
        f = f @ next_matrix

        # Print the shape of f after multiplication
        #print(f"Resulting f shape: {f.shape}")

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
    #gamma_zero = []
    
    for i in range(dim+1):#only consider b0,b1,l0,l1 if max_dim =1
        f = mat_multiply(bnd_op,i,q)
        #f = d^q
        g = mat_multiply(bnd_op,i+N-q,N-q)
        # g = d^(n-q)
        
        #betti.append(betti_number(f, g))
    
        f = np.matrix(f)
        g = np.matrix(g)
      
        lap = f.H@f+g@g.H
    
        eigvals = np.linalg.eigvalsh(lap)
        #gamma_zero.append(np.count_nonzero(eigvals.real <=1e-6)
        betti.append(np.count_nonzero(eigvals.real <=1e-6)) 
        #gamme_zero can be added too.
        positive_eigvals = eigvals.real[eigvals.real>1e-6]
        gamma_min.append(np.min(positive_eigvals) if positive_eigvals.size>0 else 0)
        gamma_max.append(np.max(positive_eigvals) if positive_eigvals.size>0 else 0)
        gamma_mean.append(np.mean(positive_eigvals)if positive_eigvals.size>0 else 0)
        gamma_std.append(np.std(positive_eigvals) if positive_eigvals.size>0 else 0)
        

    return betti,gamma_min,gamma_max,gamma_mean,gamma_std

def betti_laplacian(X,N,max_dim=2):
    '''Computes betti and laplacian information for the given complex=X for each Mayer order q<N'''
    Bettis=[]
    Gmin=[]
    Gmax=[]
    Gmean=[]
    Gstd=[]
    bnd_op = boundary_operators(X,N)
    for q in range(1,N):
        betti, gmin,gmax,gmean,gstd = calculate_betti_laplacian(bnd_op,N,q)
        Bettis.append(betti)
        Gmin.append(gmin)
        Gmax.append(gmax)
        Gmean.append(gmean)
        Gstd.append(gstd)
    return Bettis,Gmin,Gmax,Gmean,Gstd

