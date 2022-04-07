import matplotlib.pyplot as plt
import numpy as np
data0 = np.genfromtxt('cdf.dat')
plt.plot(data0[:,0],data0[:,2])
plt.xlabel('x')
plt.ylabel('CDF')
plt.legend()
plt.show()
