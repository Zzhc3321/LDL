import scipy.io as sio
from matplotlib import pyplot as plt
mat_data = sio.loadmat("DataSets/Ensemble/whole_data1.mat")

print(mat_data.keys())
print(mat_data['whole_data']['train_fea'])
print(mat_data['whole_data']['N_train'])


print(sum(sio.loadmat("DataSets/SBU_3DFE.mat")['labels'][0]))


# labels = mat_data["labels"]
# features = mat_data["features"]
# happiness, sadness, surprise,
# fear, anger, and disgust
#
# print(labels[0])
# print(features[0])

