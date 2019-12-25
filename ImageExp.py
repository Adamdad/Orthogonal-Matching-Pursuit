from OMP import *
import numpy as np
import os
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy import fftpack
from skimage.transform import resize
from skimage.metrics import peak_signal_noise_ratio as PSNR
# ! Problem 3, image compression

class ImageExperiment:
    def __init__(self,M=range(5,100,5)):
        self.sigma = 0.2
        self.M = M
        self.zigzag=np.array(open('Zig-Zag Pattern.txt').read().split()).astype(np.int).reshape(8,8)
        self.expname = "marilynmonroe210-2"
        self.imsize = (400, 320)
        self.norm_error =[]
        self.psnrs = []
        
    
    def dct8x8(self,img):
        '''Do 8x8 DCT on image (in-place)'''
        dct = np.zeros_like(img)
        for i in range(0,img.shape[0],8):
            for j in range(0,img.shape[1],8):
                dct[i:(i+8),j:(j+8)] = dct2(img[i:(i+8),j:(j+8)])
        return dct
    
    def idct8x8(self,dct):
        '''8x8 DCT to image'''
        dct_img = np.zeros_like(dct)

        for i in range(0,dct.shape[0],8):
            for j in range(0,dct.shape[1],8):
                dct_img[i:(i+8),j:(j+8)] = idct2(dct[i:(i+8),j:(j+8)])
        return dct_img
    
    def block2zigzag(self,block):
        vector = np.zeros(64)
        vector[self.zigzag] = block
        return vector

    def zigzag2block(self,vector):
        block = np.zeros((8,8))
        block = vector[self.zigzag]
        return block
    
    def dct2zigzag(self,dct):
        h,w = dct.shape
        block_h,block_w = (h//8),(w//8)
        zigzag_vector = np.zeros((block_h*block_w,64))

        for i in range(0,h,8):
            for j in range(0,w,8):
                zigzag_vector[(i//8)*block_w+j//8] = self.block2zigzag(dct[i:(i+8),j:(j+8)])
        return zigzag_vector
    
    def zigzag2dct(self,zigzag_vector,imsize):
        h,w = imsize
        block_h,block_w = (h//8),(w//8)
        dct = np.zeros((h,w))
        for i in range(0,h,8):
            for j in range(0,w,8):
                dct[i:(i+8),j:(j+8)] = self.zigzag2block(zigzag_vector[(i//8)*block_w+j//8])
        return dct
    
    def run_image(self):
        foldername = "experiment/exp_{}/".format(self.expname)
        if os.path.exists(foldername)==False:
            os.mkdir(foldername)
        img = plt.imread('images/menglu.jpg')/255    
        
        img = rgb2gray(img)
        img = resize(img, self.imsize)
        
        # Do 8x8 DCT on image and vectorized with zigzag pattern
        dct = self.dct8x8(img)
        zigzag_vector = self.dct2zigzag(dct)
        zigzag_shape = zigzag_vector.shape
        # OMP with dct zigzag vectors
        N = 64
        
        for M in self.M:
            pre_zigzag_vector = np.zeros(zigzag_shape)
            A = define_A(M,N)
            for i in range(zigzag_shape[0]):
                x = zigzag_vector[i].T
                y = A.dot(x)
                x_pre,Lambdas = OMP(A,y,N)
                pre_zigzag_vector[i] = x_pre.T
                
            
            pre_dct = self.zigzag2dct(pre_zigzag_vector,self.imsize)
            pre_img = self.idct8x8(pre_dct)
            norm_e =Normalized_Error(img,pre_img)
            self.norm_error.append(norm_e)
            psnr = PSNR(img,pre_img)
            self.psnrs.append(psnr)
            print("M={} Normalized Error={} PSNR={}".format(M,norm_e,psnr))
            plt.imsave(os.path.join(foldername,"preimageM={}.png".format(M)),pre_img,cmap="gray")
        
    def run_image_noise(self):
        foldername = "experiment/exp_{}/".format(self.expname)
        if os.path.exists(foldername)==False:
            os.mkdir(foldername)
        img = plt.imread('images/menglu.jpg')/255    
        
        img = rgb2gray(img)
        img = resize(img, self.imsize)
        
        # Do 8x8 DCT on image and vectorized with zigzag pattern
        dct = self.dct8x8(img)
        zigzag_vector = self.dct2zigzag(dct)
        zigzag_shape = zigzag_vector.shape
        # OMP with dct zigzag vectors
        N = 64
        
        for M in self.M:
            pre_zigzag_vector = np.zeros(zigzag_shape)
            A = define_A(M,N)
            for i in range(zigzag_shape[0]):
                x = zigzag_vector[i].T
                n = np.random.normal(0,self.sigma,M).T
                y = A.dot(x) + n
                x_pre,Lambdas = OMP(A,y,N,r_thresh=norm2(n))
                pre_zigzag_vector[i] = x_pre.T
                
            
            pre_dct = self.zigzag2dct(pre_zigzag_vector,self.imsize)
            pre_img = self.idct8x8(pre_dct)
            norm_e =Normalized_Error(img,pre_img)
            self.norm_error.append(norm_e)
            psnr = PSNR(img,pre_img)
            self.psnrs.append(psnr)
            print("Noise M={} Normalized sigma={} Error={} PSNR={}".format(M,self.sigma,norm_e,psnr))
            
            plt.imsave(os.path.join(foldername,"preimageM={}.png".format(M)),pre_img,cmap="gray")

    def plot(self):
        plotpath = "plot/"
        plt.figure()
        plt.plot(self.M,self.psnrs)
        plt.xlabel("M")
        plt.ylabel("PSNR")
        plt.savefig(os.path.join(plotpath,"{}PSNR.png".format(self.expname)))
        
        plt.figure()
        plt.plot(self.M,self.norm_error)
        plt.xlabel("M")
        plt.ylabel("Normalized Error")
        plt.savefig(os.path.join(plotpath,"{}NormalizedError.png".format(self.expname)))
    
    
              
   
if __name__ == "__main__":
    exp = ImageExperiment()
    # exp.run_image()
    exp.run_image_noise()
    exp.plot()


    