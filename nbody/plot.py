import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
tmp = np.genfromtxt('f0.dat')
data = np.zeros((tmp.shape[0], tmp.shape[1], 4))
data[:,:,0] = tmp 
data[:,:,1] = np.genfromtxt('f100.dat')
data[:,:,2] = np.genfromtxt('f200.dat')
data[:,:,3] = np.genfromtxt('f1000.dat')
fig = plt.figure()
for k in range(1,5):
	ax = fig.add_subplot(2,2,k,projection='3d')
	ax.scatter(data[:,0,k-1],data[:,1,k-1],data[:,2,k-1],marker = 'o')
	ax.set_xticks([])
	ax.set_yticks([])
	ax.set_zticks([])
plt.show()
