import sys
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF

dataset = int(sys.argv[1])

txt1="Reconstructed dataset view X1-X2"
txt2="Reconstructed dataset view X2-X3"

def euclidean(x, y):
    dist = np.sqrt(np.sum(np.square(np.subtract(x,y)),keepdims=True))
    return dist   
    
X = pd.read_csv('output_files' + '/' + str(dataset) + '.Original_bottom_half_3d.csv', header = None).to_numpy()
Y = pd.read_csv('output_files' + '/' + str(dataset) + '.Reconstructed_bottom_half_3d.csv', header = None).to_numpy()

[R, C] = X.shape

temp1 = np.zeros([R,C])

for i in range(1, R):
    temp1[i, 0] = i
    temp1[i, 1] = euclidean(X[i,:], Y[i,:])

#for i in range(0, R-1):
#    temp1[i+1, 2] = euclidean(X[i,:], X[i+1,:])

plt.plot([temp1[:,0], temp1[:,0]], [temp1[:,1], temp1[:,1]], "r.")
#plt.plot([temp1[:,0], temp1[:,0]], [temp1[:,1], temp1[:,1]], "r.", [temp1[:,0], temp1[:,0]], [temp1[:,2]+300, temp1[:,2]+300], "b.")
plt.axis([min(min(temp1[:,0]), min(temp1[:,1]))-7, max(max(temp1[:,0]), max(temp1[:,1]))+7, min(min(temp1[:,0]), min(temp1[:,1]))-7, max(max(temp1[:,0]), max(temp1[:,1]))+7])
plt.xlabel("X1")
plt.ylabel("X2")
#plt.show()
plt.savefig('output_images' + '/' + str(dataset) + '.data_error.png')
plt.clf()
