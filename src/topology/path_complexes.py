import numpy as np
import gudhi
import cmath
import Path_Mayer_Homology_Laplacian
from Path_Mayer_Homology_Laplacian import get_faces
# The difference betweeen alpha and Rips,
# Alpha is a subcomplex of Delunay triangulation so there are no crossing where Rips contains crossings too. 

class path_complex_alpha():
    # Inputs are pointcloud and distance 
    # Output is Alpha complex for given pointcloud at the given distance
    #Rips is not computing higher cells
    def __init__(self, pointcloud):
        self.simplex_list = []
        self.filtration_list = []
        comp = gudhi.AlphaComplex(points=pointcloud)
        simplex_tree = comp.create_simplex_tree()
        for filtered_value in simplex_tree.get_filtration():
            self.simplex_list.append(filtered_value[0])
            self.filtration_list.append(filtered_value[1])
        
    def get_complex(self, filtration_distance):
        # Compute the standard Rips complex
        fil_list = np.array(self.filtration_list)
        index = np.array(fil_list) <= (filtration_distance / 2) ** 2
        
        simplices = np.array(self.simplex_list, dtype=object)[index]
        simplices= simplices.tolist()
        
        # Additional logic to check for specified sequences of edges and add 2-cells
        additional_cells = []
        possibles =[]

        for i, simplex in enumerate(simplices):
            if len(simplex) == 2:
                #faces = get_faces(simplex)
                # Check if the simplex is an edge [i, j]
                # Look for sequences [i, j], [j, k], [i, l], [l, k] in the remaining simplices
                for j in range(i+1, len(simplices)):
                    next_simplex = simplices[j]
                    #print(simplex,next_simplex)
                    if len(next_simplex) == 2:
                        if {simplex[1]} == {next_simplex[0]} and ([simplex[0],next_simplex[1]] not in simplices):
                            possibles.append([simplex,next_simplex])
                            #possibles is collection pair of edges whose initial and end is matching like [i,j],[j,k]
        
        for i, pair1 in enumerate(possibles):
             for pair2 in possibles[i+1:]:
                  if pair1[0][0] == pair2[0][0] and pair1[1][1] == pair2[1][1]:
                        #add as [[0, 1, 4], [0, 2, 4]] so it will give direction to the cell like first-second
                        i=pair1[0][0]
                        j=pair1[0][1]
                        k=pair1[1][1]
                        l=pair2[0][1]
                        additional_cells.append([[i, j, k],[i, l, k]])
        
        #print(additional_cells)
        for cell in additional_cells:  
            simplices.append(cell)
        return  simplices



class path_complex_rips():

    def __init__(self, pointcloud,max_dim=5):
        self.simplex_list = []
        self.filtration_list = []
        comp = gudhi.RipsComplex(points=pointcloud)
        simplex_tree = comp.create_simplex_tree(max_dimension=max_dim)
        for filtered_value in simplex_tree.get_filtration():
            self.simplex_list.append(filtered_value[0])
            self.filtration_list.append(filtered_value[1])
        
    
    def get_complex(self, filtration_distance):
        # Compute the standard Rips complex
        fil_list = np.array(self.filtration_list)
        #index = np.array(fil_list) <= (filtration_distance / 2) ** 2
        index = np.array(fil_list) <=  filtration_distance 

        #### Check whether the distance should be as it is or this half squared version
        
        simplices = np.array(self.simplex_list, dtype=object)[index]
        simplices= simplices.tolist()
        
        # Additional logic to check for specified sequences of edges and add 2-cells
        additional_cells = []
        possibles =[]
        for i, simplex in enumerate(simplices):
            if len(simplex) == 2:
                # Check if the simplex is an edge [i, j]
                # Look for sequences [i, j], [j, k], [i, l], [l, k] in the remaining simplices
                for j in range(i+1, len(simplices)):
                    next_simplex = simplices[j]
                    #print(simplex,next_simplex)
                    if len(next_simplex) == 2:
                        if {simplex[1]} == {next_simplex[0]} and ([simplex[0],next_simplex[1]] not in simplices):
                            possibles.append([simplex,next_simplex])
                            #possibles is collection pair of edges whose initial and end is matching like [i,j],[j,k]
        
        for i, pair1 in enumerate(possibles):
             for pair2 in possibles[i+1:]:
                  if pair1[0][0] == pair2[0][0] and pair1[1][1] == pair2[1][1]:
                        #add as [[0, 1, 4], [0, 2, 4]] so it will give direction to the cell like first-second
                        i=pair1[0][0]
                        j=pair1[0][1]
                        k=pair1[1][1]
                        l=pair2[0][1]
                        additional_cells.append([[i, j, k],[i, l, k]])
        
        #print(additional_cells)
        for cell in additional_cells:  
            simplices.append(cell)
        return  simplices
    


class PathComplexAlpha:
    # Inputs are pointcloud and distance 
    # Output is Rips complex for given pointcloud at the given distance
    # Rips is not computing higher cells
    def __init__(self, pointcloud, atom_types):
        self.simplex_list = []
        self.filtration_list = []
        self.pointcloud = pointcloud
        self.atom_types = atom_types
        
        # Dictionary of electronegativities for each atom type
        self.electronegativities = {
            'H': 2.20,
            'C': 2.55,
            'N': 3.04,
            'O': 3.44,
            # Add more elements as needed
        }
        
        comp = gudhi.AlphaComplex(points=pointcloud)
        simplex_tree = comp.create_simplex_tree()
        for filtered_value in simplex_tree.get_filtration():
            self.simplex_list.append(filtered_value[0])
            self.filtration_list.append(filtered_value[1])
        
    def get_complex(self, filtration_distance):
        # Compute the standard Rips complex
        fil_list = np.array(self.filtration_list)
        index = np.array(fil_list) <= (filtration_distance / 2) ** 2
        
        simplices = np.array(self.simplex_list, dtype=object)[index]
        simplices = simplices.tolist()
        
        # Add direction to the edges based on electronegativities
        directed_simplices = []
        for simplex in simplices:
            if len(simplex) == 2:
                i, j = simplex
                atom_i = self.atom_types[i]
                atom_j = self.atom_types[j]
                if self.electronegativities[atom_i] < self.electronegativities[atom_j]:
                    directed_simplices.append([i, j])
                else:
                    directed_simplices.append([j, i])
            else:
                directed_simplices.append(simplex)
        
        # Additional logic to check for specified sequences of edges and add 2-cells
        additional_cells = []
        possibles = []
        for i, simplex in enumerate(directed_simplices):
            if len(simplex) == 2:
                # Check if the simplex is an edge [i, j]
                # Look for sequences [i, j], [j, k], [i, l], [l, k] in the remaining simplices
                for j in range(i + 1, len(directed_simplices)):
                    next_simplex = directed_simplices[j]
                    if len(next_simplex) == 2:
                        if {simplex[1]} == {next_simplex[0]} and ([simplex[0], next_simplex[1]] not in directed_simplices):
                            possibles.append([simplex, next_simplex])
        
        for i, pair1 in enumerate(possibles):
            for pair2 in possibles[i + 1:]:
                if pair1[0][0] == pair2[0][0] and pair1[1][1] == pair2[1][1]:
                    i = pair1[0][0]
                    j = pair1[0][1]
                    k = pair1[1][1]
                    l = pair2[0][1]
                    additional_cells.append([[i, j, k], [i, l, k]])
        
        for cell in additional_cells:
            directed_simplices.append(cell)
        
        return directed_simplices
