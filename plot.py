import matplotlib.pyplot as plt 
import numpy as np 
import os
plt.rcParams["font.family"] = "Times New Roman"
Ns=[20,50,100]
# os.mkdir('plot/Noise2/')
for N in Ns:
    # image = np.load('Noise2_N{}_sigma0.001.npy'.format(N))
    # name = 'N{}_ESR_rate'.format(N)
    name = 'Noise2_N{}_sigma0.001'.format(N)
    image = np.load('{}.npy'.format(name))

    plt.figure()
    print(image.shape)
    plt.imshow(image)
    ytick = range(N//2,N,3)
    xtick = range(10,100,5)
    plt.xticks(np.arange(len(xtick)),xtick)
    plt.yticks(np.arange(len(ytick)),ytick)
    plt.xlabel('M',fontsize=15)
    plt.ylabel('s max',fontsize=15)
    plt.colorbar()
    plt.savefig('plot/Noise2/{}.png'.format(name))