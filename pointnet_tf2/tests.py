import h5py

file = h5py.File("./data/test_VDS.h5", 'r')
file = h5py.File("./data/stanford_indoor3d/ply_data_all_0.h5", 'r')
print(file.keys())
print(file['data'][0])
print(file['label'][mask, ...])
