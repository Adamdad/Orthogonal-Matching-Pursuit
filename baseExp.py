from OMP import *
import numpy as np
from tqdm import tqdm
import time
# ! Problem 1, OMP with no noise

class Experiment:
    def __init__(self,M=range(10,100,5),N=[20,50,100]):
        self.M = M
        self.N = N
        
    def define_all(self,M,N,s):
        A = define_A(M,N)
        x,index = define_x(s,N)
        y = A.dot(x)
        return A,x,y,index
        
    def run_exp_NormError(self,times = 2000):
        
        for N in self.N:
            s_max_range = range(N//2,N,3)
            len_M = len(self.M)
            len_s_max = len(s_max_range)
            transition_map = np.zeros((len_s_max,len_M))
            print(transition_map.shape)
            for M_id in range(len_M):
                M = self.M[M_id]
                for s_max_id in range(len_s_max):
                    s_max = s_max_range[s_max_id]
                    e = 0
                    for e_id in tqdm(range(times)):
                        s = np.random.randint(low=1,high = s_max)
                        A,x,y,index = self.define_all(M,N,s)
                        x_pre,Lambdas = OMP(A,y,N)
                        e += Normalized_Error(x,x_pre)
                    e /= times
                    transition_map[s_max_id,M_id]=e
                    print("Normalized_Error for M={} and N={} {}".format(M,N,e))
            
            np.save("N{}.npy".format(N),transition_map)
            print("N={} finished".format(N))     
              
    def run_exp_ESR(self,times = 2000):
            
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
                        s = np.random.randint(low=1,high=s_max)
                        A,x,y,true_index = self.define_all(M,N,s)
                        x_pre,pred_index = OMP(A,y,N,r_thresh=0.001)

                        if set(true_index)==set(pred_index):
                            t+=1
                    t=t/times
                    succes_map[s_max_id,M_id] = t
                    print("ESR rate for M={} and N={} {}".format(M,N,t))
            
            np.save("N{}_ESR_rate.npy".format(N),succes_map)
            print("N={} finished".format(N))        
    
    def run_exp_Normtrans(self,times = 2000):
            
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
                        A,x,y,index = self.define_all(M,N,s)
                        x_pre,Lambdas = OMP(A,y,N)
                        e = Normalized_Error(x,x_pre)
                        if e < 0.001:
                            t += 1
                    t /= times
                    succes_map[s_max_id,M_id] = t
                    print("Normalized_Error_propability for M={} and N={} {}".format(M,N,t))
            
            np.save("N{}_Normalized_Error_propability_trans_plot.npy".format(N),succes_map)
            print("N={} finished".format(N)) 

if __name__ == "__main__":
    exp = Experiment()
    exp.run_exp_ESR()


    