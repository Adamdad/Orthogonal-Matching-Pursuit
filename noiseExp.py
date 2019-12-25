from baseExp import Experiment
from OMP import *
import numpy as np
from tqdm import tqdm

# ! Problem 2, OMP with noise

class NoiseExperiment(Experiment):
    def __init__(self,sigma=0.001,M=range(10,100,5),N=[20,50,100]):
        '''
        :args
        sigma: the std of the noise
        
        '''
        super(NoiseExperiment,self).__init__
        self.sigma = sigma
        self.M = M
        self.N = N
        
        
    def define_noise_all(self,M,N,s):
        A = define_A(M,N)
        x,index = define_x(s,N)
        y = A.dot(x)
        n = np.random.normal(0, self.sigma, M)
        y += n.T
        
        return A,x,y,index,norm2(n)
    
    def run_exp_Normtrans1(self,times = 2000):
        '''Noise experiment 1: known sparsity'''
        for N in self.N:
            s_max_range = range(N//2,N,3)
            len_M = len(self.M)
            len_s_max = len(s_max_range)
            succes_map = np.zeros((len_s_max,len_M))
            print(succes_map.shape)
            for M_id in range(len_M):
                M = self.M[M_id]
                for s_max_id in range(len_s_max):
                    s_max = s_max_range[s_max_id]
                    t = 0
                    for e_id in tqdm(range(times)):
                        s = np.random.randint(low=1,high = s_max)
                        A,x,y,index,norm = self.define_noise_all(M,N,s)
                        x_pre,Lambdas = OMP(A,y,N,stop=s)
                        temp_e = Normalized_Error(x,x_pre)

                        if temp_e<0.001:
                            t+=1
                    t = t/times
                    succes_map[s_max_id,M_id] = t
                    print("Normalized_Error_propability for M={} and N={} {}".format(M,N,t))
            
            np.save("Noise1_N{}_sigma{}.npy".format(N,self.sigma),succes_map)
            print("N={} finished".format(N))
    
    def run_exp_Normtrans2(self,times = 2000):
        '''Noise experiment 2: unknown sparsity, known noise norm'''
        for N in self.N:
            s_max_range = range(N//2,N,3)
            len_M = len(self.M)
            len_s_max = len(s_max_range)
            succes_map = np.zeros((len_s_max,len_M))
            print(succes_map.shape)
            for M_id in range(len_M):
                M = self.M[M_id]
                for s_max_id in range(len_s_max):
                    s_max = s_max_range[s_max_id]
                    t = 0
                    for e_id in tqdm(range(times)):
                        s = np.random.randint(low=1,high = s_max)
                        A,x,y,index,norm = self.define_noise_all(M,N,s)
                        x_pre,Lambdas = OMP(A,y,N,r_thresh=norm)
                        temp_e = Normalized_Error(x,x_pre)

                        if temp_e<0.001:
                            t+=1
                    t = t/times
                    succes_map[s_max_id,M_id] = t
                    print("Normalized_Error_propability for M={} and N={} {}".format(M,N,t))
            
            np.save("Noise2_N{}_sigma{}.npy".format(N,self.sigma),succes_map)
            print("N={} finished".format(N))   
            
        

if __name__ == "__main__":
    exp = NoiseExperiment()
    exp.run_exp_Normtrans2()