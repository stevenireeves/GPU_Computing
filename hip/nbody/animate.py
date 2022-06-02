import os

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import mpl_toolkits.mplot3d.axes3d as p3
import numpy as np
direc = os.listdir('data/')
direc.sort()


temp = np.genfromtxt('data/f00000.dat')
data = np.zeros((temp.shape[0], temp.shape[1], len(direc)))
data[:,:,0] = temp
for i in range(1, len(direc)):
    data[:, :, i] = np.genfromtxt('data/' + direc[i]) 

def update_graph(iterator):
    graph._offsets3d = (data[:,0,iterator], data[:,1,iterator], data[:,2, iterator])

fig = plt.figure()
fig.set_size_inches(5, 5)
ax = fig.add_subplot(111, projection='3d')
graph = ax.scatter(temp[:,0],temp[:,1],temp[:,2],s=temp[:,3]*10,marker='o')
ani = animation.FuncAnimation(fig, update_graph, len(direc)-1, interval=50, blit=False, repeat=True)
#plt.show()
dpi = 300
Writer = animation.writers['ffmpeg']
writer = Writer(fps=60, metadata=dict(artist='Me'), bitrate=1800, extra_args=['-vcodec', 'libx264'])
ani.save('N-Body.mp4', writer=writer, dpi=dpi)
