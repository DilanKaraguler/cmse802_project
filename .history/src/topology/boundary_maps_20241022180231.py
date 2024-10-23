import numpy as np
import cmath

def croot(k,N):
    '''Computes the k-th root of unity for N'''
    if N<=0:
        return None
    return cmath.exp((2 * cmath.pi * 1j * k ) / N)
    
#the followings needed to compute boundary matrices for path complexes
#nested, get_faces, get_coeff, boundary
def nested(complex):
    '''Computes the non-elementary simplices in the complex which are called nested
    
    imputs: complex (list of simplices)
    outputs: nested_info, all_elementary, to_be_deleted (all lists)
        nested_info = nonelementary simplices
        all_elementary = all elementary simplices
        to_be_deleted = simplices that are used to compute non-elementary simplices and will be deleted later
    '''
    path_complex = complex
    nested_info = []
    all_elementary = path_complex.copy()
    to_be_deleted = []
    for spx in path_complex:
        if type(spx[0]) == list:
            all_elementary.append(spx[0])
            all_elementary.append(spx[1])
            #if these elements were already in the set,
            #then there has to be a mistake. contradict with minimality
            nested_info.append(spx)
            all_elementary.remove(spx)

            faces1 = get_faces(spx[0])
            faces2 = get_faces(spx[1])
            
            faces = faces1 + faces2
            
            for face in faces:
                if face not in all_elementary:
                    
                    all_elementary.append(face)
                    to_be_deleted.append(face)
                else:
                    continue    
    # using all elementary simplexes to be able to compute boundary matrices
    #saving the nested info to modify the result boundary matrices
    return nested_info , all_elementary, to_be_deleted
    
def get_faces(lst):
    '''Computes the list of faces of given simplex
    input: lst (list)
    output: list of faces of lst (list of lists)'''
    return [lst[:i] + lst[i+1:] for i in range(len(lst))]


def get_coeff(simplex, faces,N):
    '''Computes the coefficient of faces in the boundary of a simplex
    inputs: simplex, faces, N (list,list,number)
    N is the Mayer degree
    output: coefficient of the given face in the boundary of a given simplex (number)'''
    if simplex in faces:
        idxs = [i for i in range(len(faces)) if faces[i] == simplex]
        return sum([croot(idx,N) for idx in idxs])
    else:
        return 0

def boundary(complex,N):
    '''Boundary matrix of a complex
    inputs: complex (list of simplices), N (Mayer degree)
    output: boundary matrix (list of numpy arrays)'''
    
    path_complex = complex
    nested_info , all_elementary , to_be_deleted =  nested(path_complex)
   
    maxdim = len(max(all_elementary, key=len))
    simplices = [sorted([spx for spx in all_elementary if len(spx)==i]) for i in range(1,maxdim+1)]
    
    #it will sort and group according to their dimension
    bnd = [np.zeros((0,len(simplices[0])))]
    #added zero boundary map
    for spx_k, spx_kp1 in zip(simplices, simplices[1:]):
        # zip([1,2,3],[a,b,c]) = {(1,a),(2,b),(3,c)}
        mtx = []
        #mtx is placeholder for boundary matrix at the spesific dimension
        for sigma in spx_kp1:
            faces = get_faces(sigma)
            mtx.append([get_coeff(spx, faces,N) for spx in spx_k])
        bnd.append(np.array(mtx).T)
    
    for ekstra in to_be_deleted:
            
            nested_dim = len(ekstra)-1
            index_eks = simplices[nested_dim].index(ekstra)
            
            bnd[nested_dim]=np.delete(bnd[nested_dim],index_eks, axis=1)
            bnd[nested_dim+1]=np.delete(bnd[nested_dim+1],index_eks, axis=0)
            simplices[nested_dim].remove(ekstra)
    
    for nested_pair in nested_info:
            spx0 = nested_pair[0]
            spx1 = nested_pair[1]
            nested_dim = len(spx0)-1
            idx1 = simplices[nested_dim].index(spx0)
            idx2 = simplices[nested_dim].index(spx1)
            bnd[nested_dim][:, idx1] -= bnd[nested_dim][:, idx2]
            bnd[nested_dim]=np.delete(bnd[nested_dim],idx2, axis=1)
            
            if len(bnd) - 1 >= nested_dim:
                #idx2 = simplices[nested_dim].index(nested_pair[1])
                bnd[nested_dim+1]=np.delete(bnd[nested_dim+1],idx2, axis=0)
            simplices[nested_dim].remove(spx1)

    return bnd


class boundary_operators():
    '''Creates the boundary operator as a class for the given path complex and a Mayer degree
    
    inputs: complex (list of simplices), N (Mayer degree)
    outputs: boundary operator (class object)
    
    .get(index) returns the boundary matrix of the given index
    '''

    def __init__(self,complex,N):
        self.boundaries = {}
        bnd = boundary(complex,N)
        self.max_dim = len(bnd)
        for i in range(self.max_dim):
            self.boundaries[i] = bnd[i]
    def get(self,index):
        if index in self.boundaries:
            return self.boundaries[index]
        elif index == self.max_dim:
            return np.zeros((self.boundaries[self.max_dim-1].shape[1],0))
        else:
            return np.zeros((0,0))
    
    #indexes in boundary_operators match with the given index bnd.op.get(n)=B_n