import matplotlib.pyplot as plt
import numpy as np
data = np.genfromtxt('histo.dat')
bins = np.linspace(0,255,num=256)
plt.plot(bins,data[:,0],'r',label="r")
plt.plot(bins,data[:,1],'g',label="g")
plt.plot(bins,data[:,2],'b',label="b")
plt.xlabel('Bins')
plt.ylabel('Color Distributions')
plt.xlim(0,255)
plt.ylim(0,400000)
plt.legend()
plt.show()
