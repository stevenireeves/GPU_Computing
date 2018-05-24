import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
temp = np.genfromtxt('IC.dat')
data = np.zeros((temp.shape[0], temp.shape[1],4))
data[:,:,0] = temp
data[:,:,1] = np.genfromtxt('f100.dat')
data[:,:,2] = np.genfromtxt('f250.dat')
data[:,:,3] = np.genfromtxt('f500.dat')
fig = plt.figure()

for k in range(0,4):
	ax = fig.add_subplot(2,2,k+1,projection='3d')
	ax.scatter(data[:,0,k],data[:,1,k],data[:,2,k],s=data[:,3,k]*10,marker='o')
	ax.set_xticks([])
	ax.set_yticks([])
	ax.set_zticks([])
plt.show()
