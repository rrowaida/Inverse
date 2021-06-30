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



if dataset==4:
    X = pd.read_csv('data' + '/' + 'Data_' + str(dataset) + '_128d.csv', header = None).to_numpy()
else:
    X = pd.read_csv('data' + '/' + 'Data_' + str(dataset) + '_3d.csv', header = None).to_numpy()
Y = pd.read_csv('data' + '/' +'Data_' + str(dataset) + '_2d.csv', header = None).to_numpy()


def euclidean(x, y):
    dist = np.sqrt(np.sum(np.square(np.subtract(x,y)),keepdims=True))
    return dist   


def rbf(x, y):
    #return np.exp(-(euclidean(x, y)**2)*(eps**2))
    return np.power(euclidean(x, y),1) 


[N, d1] = X.shape
[N2, d2] = Y.shape

#creating empty lists to hold train and test data
top_half_3d = np.zeros([math.floor(N/2),d1])
top_half_2d = np.zeros([math.floor(N/2),d2])
bottom_half_3d = np.zeros([math.floor(N/2),d1])
bottom_half_2d = np.zeros([math.floor(N/2),d2])

j=0
for i in range (N):
    if i%2 == 0:
        top_half_3d[j,:] = X[i,:]
        top_half_2d[j,:] = Y[i,:]
        j = j + 1
    else:
        bottom_half_3d[j-1,:] = X[i,:]
        bottom_half_2d[j-1,:] = Y[i,:]

#saving original test data for error comparison
name1 = 'output_files' + '/' + str(dataset) + '.Original_bottom_half_3d.csv'
np.savetxt(name1, bottom_half_3d, delimiter = ',')


X = top_half_3d
Y = top_half_2d


n = math.floor(N/2)
D = d1


K = np.zeros([n,n])
for i in range(n):
    for j in range(n):
        K[i,j] = rbf(Y[i,:], Y[j,:])
#print(K)
A = np.dot(np.linalg.inv(K),X)
print(A.shape)


Y_new = bottom_half_2d
X_new = np.zeros([n,d1])


K_new = np.zeros([n,n])

for i in range(n):
    for j in range(n):
        K_new[i,j] = rbf(Y_new[i,:], Y[j,:])

for i in range(n):
    for j in range(d1):
          X_new[i,j] = np.dot(np.transpose(A[:,j]),np.transpose(K_new[i,:]))
          #print(X_new[i,j])


X_new_df = pd.DataFrame(X_new)

# Saving reconstructed test higher dimension data for error comparion
name2 = 'output_files' + '/' + str(dataset) + '.Reconstructed_bottom_half_3d.csv'
np.savetxt(name2, X_new_df, delimiter = ',')

# plot and save original test data from X1-X2
plt.plot([bottom_half_3d[:,0], bottom_half_3d[:,0]], [bottom_half_3d[:,1], bottom_half_3d[:,1]], "r.")
plt.axis([min(min(bottom_half_3d[:,0]), min(bottom_half_3d[:,1]))-3, max(max(bottom_half_3d[:,0]), max(bottom_half_3d[:,1]))+3, min(min(bottom_half_3d[:,0]), min(bottom_half_3d[:,1]))-3, max(max(bottom_half_3d[:,0]), max(bottom_half_3d[:,1]))+3])
plt.xlabel("X1")
plt.ylabel("X2")
#plt.show()
plt.savefig('output_images' + '/' + str(dataset) + '.before_x1-x2.png')
plt.clf()

# plot and save original test data from X2-X3
plt.plot([bottom_half_3d[:,1], bottom_half_3d[:,1]], [bottom_half_3d[:,2], bottom_half_3d[:,2]], "g.")
plt.axis([min(min(bottom_half_3d[:,1]), min(bottom_half_3d[:,2]))-3, max(max(bottom_half_3d[:,1]), max(bottom_half_3d[:,2]))+3, min(min(bottom_half_3d[:,1]), min(bottom_half_3d[:,2]))-3, max(max(bottom_half_3d[:,1]), max(bottom_half_3d[:,2]))+3])
plt.xlabel("X2")
plt.ylabel("X3")
#plt.show()
plt.savefig('output_images' + '/' + str(dataset) + '.before_x2-x3.png')
plt.clf()

# plot and save reconstructed higher dimension test data from X1-X2
plt.plot([X_new[:,0], X_new[:,0]], [X_new[:,1], X_new[:,1]], "r.")
plt.axis([min(min(X_new[:,0]), min(X_new[:,1]))-3, max(max(X_new[:,0]), max(X_new[:,1]))+3, min(min(X_new[:,0]), min(X_new[:,1]))-3, max(max(X_new[:,0]), max(X_new[:,1]))+3])
plt.xlabel("X1")
plt.ylabel("X2")
plt.figtext(0.5, 0.9, txt1, wrap=True, horizontalalignment='center', fontweight="bold", fontsize=12, color="green")
#plt.show()
plt.savefig('output_images' + '/' + str(dataset) + '.after_x1-x2.png')
plt.clf()

# plot and save reconstructed higher dimension test data from X1-X2
plt.plot([X_new[:,1], X_new[:,1]], [X_new[:,2], X_new[:,2]], "g.")
plt.axis([min(min(X_new[:,1]), min(X_new[:,2]))-3, max(max(X_new[:,1]), max(X_new[:,2]))+3, min(min(X_new[:,1]), min(X_new[:,2]))-3, max(max(X_new[:,1]), max(X_new[:,2]))+3])
plt.xlabel("X2")
plt.ylabel("X3")
plt.figtext(0.5, 0.9, txt2, wrap=True, horizontalalignment='center', fontweight="bold", fontsize=12, color="green")
#plt.show()

# saving reconstructed higher dimension test data points
plt.savefig('output_images' + '/' + str(dataset) + '.after_x2-x3.png')


