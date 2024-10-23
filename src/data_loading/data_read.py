def read_file(filename):
'''Converts given file (mol2, xyz, xy) to a pointcloud'''
    file_format = filename.split('.')[-1]
    pointcloud=[]
    if file_format == 'mol2':
        from biopandas.mol2 import PandasMol2
        data =PandasMol2().read_mol2(filename).df
        for x,y,z in zip(data["x"],data["y"],data["z"]):
              pointcloud.append(np.array([x,y,z]))
    if file_format == 'xyz':
        with open(filename,'r') as f:
            for item in f.readlines():
                x,y,z = item.split()
                pointcloud.append(np.array([float(x),float(y),float(z)]))
    if file_format == 'xy':
        with open(filename,'r') as f:
            for item in f.readlines():
                x,y = item.split()
                pointcloud.append(np.array([float(x),float(y)]))
    
    return pointcloud

def read_file_mol(filename):
'''Converts given file (mol2) to a pointcloud with atom type information'''
    pointcloud=[]
    atom_types = []
   
    from biopandas.mol2 import PandasMol2
    data =PandasMol2().read_mol2(filename).df
    for x,y,z, atom_name in zip(data["x"],data["y"],data["z"],data['atom_name']):
        pointcloud.append(np.array([x,y,z]))
        atom_types.append(atom_name)

    return pointcloud, atom_types