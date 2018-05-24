import matplotlib.pyplot as plt
import numpy as np
data0 = np.genfromtxt('sol0.dat')
data1 = np.genfromtxt('sol65536.dat')
data2 = np.genfromtxt('sol131072.dat')
data3 = np.genfromtxt('sol196608.dat')
plt.plot(data0[:,0],data0[:,1],'r',label="t=0.0")
plt.plot(data1[:,0],data1[:,1],'g',label="t=0.5")
plt.plot(data2[:,0],data2[:,1],'b',label="t=1.0")
plt.plot(data3[:,0],data3[:,1],'k',label="t=1.5")
plt.xlabel('x')
plt.ylabel('heat distribution at time t')
plt.legend()
plt.show()
