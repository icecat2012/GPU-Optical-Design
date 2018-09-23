import re
import time
import numpy as np
import matplotlib.pyplot as plt
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

class lights:
    def __init__(self, size=1, beams_cnt=1, beams=None):
        #eg lights(np.array([[0.484, 100, 0,0, 1,0], [0.485, 50, 0,0, 0,1]]))
        self.size = size
        self.beams_cnt = beams_cnt
        self.beams = beams #[[wavelength, energy, start_x, start_y, direction_x, direction_y], ...]
    def interpolation_angle(self, table, index, rand_num, i, k):
        if table[k,2]==0:
            angle = table[k,2]
        elif table[k-1,1]==table[k,1]:
            angle = (table[k,2]-table[k-1,2])*(rand_num[i,0]-index[k-1])/(index[k]-index[k-1])+table[k-1,2]
        else:
            angle = table[k,2]
        return angle
        
    def read_source_file(self, filename):
        table = list()
        for line in open(filename, 'r', encoding='UTF-8'):
            if line[0]>='0' and line[0]<='9':
                line = re.sub(r'\s+', ' ', line)
                tmp = line.split()
                table.append(tmp)
        table = np.array(table).astype(np.float32)
        delete_list = list()
        for i in range(table.shape[0]):
            if table[i, 4]==0:
                delete_list.append(i)
        
        return np.delete(table, delete_list, 0)

    def random_beams(self, filename):
        #random x, y
        rand_num = np.random.uniform(-self.size/2, self.size/2, self.beams_cnt)
        self.beams = np.zeros((self.beams_cnt, 6), np.float32)
        rand_num = np.array(rand_num).reshape((self.beams_cnt,))
        self.beams[:, 2] = rand_num
        self.beams[:, 3] = np.zeros((self.beams_cnt,))
        
        #load light source table
        table = self.read_source_file(filename)
        index = [0]
        for v in table:
            index.append(index[-1]+v[4])
        
        #random polar angle
        rand_num = np.random.uniform(0, index[-1], self.beams_cnt*2)
        rand_num = rand_num.reshape((self.beams_cnt,2))
        for i in range(self.beams_cnt):
            for k in range(len(index)-1):
                if rand_num[i,0]>=index[k] and rand_num[i,0]<index[k+1]:
                    angle = self.interpolation_angle(table, index, rand_num, i, k)
                    rad = np.pi*(90-angle)/180
                    self.beams[i, 0] = table[k, 1]
                    self.beams[i, 1] = 1000
                    self.beams[i, 4] = (rand_num[i, 1]-(index[-1]/2))/np.absolute(rand_num[i, 1]-(index[-1]/2)) * np.cos(rad)
                    self.beams[i, 5] = np.sin(rad)
                    break
        
    def wavelength_distribution(self, bins=50):
        #eg lights(...).wavelength_distribution(10)
        plt.figure(figsize=(16,16))
        n, bins, patches = plt.hist(self.beams[:,0].T, bins, density=False, facecolor='g')
        plt.xlabel('wavelength')
        plt.ylabel('count')
        plt.title('wavelength_distribution')
        plt.grid(True)
        plt.show()
        return n, bins, patches
    
    def energy_distribution(self, bins=50):
        #eg lights(...).energy_distribution(10)
        plt.figure(figsize=(16,16))
        n, bins, patches = plt.hist(self.beams[:,1].T, bins, density=False, facecolor='g')
        plt.xlabel('energy')
        plt.ylabel('count')
        plt.title('energy_distribution')
        plt.grid(True)
        plt.show()
        return n, bins, patches
    
    def light_distribution(self, sample_ratio=1.0):
        #eg lights(...).energy_distribution(0.7)
        # sample data
        rows = self.beams.shape[0]
        samples = self.beams[np.random.choice(rows, int(rows*sample_ratio), replace=False), :]
        
        # plot data
        plt.figure(figsize=(16,16))
        plt.quiver(samples[:,2], samples[:,3], samples[:,4], samples[:,5], label='sample ratio {}'.format(sample_ratio), width=0.005, headwidth=3, headaxislength=4)
        plt.title('light_distribution')
        plt.legend(loc='best')
        plt.axis('equal')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.grid(True)
        plt.show()
    
    def cuda_normalize_beam(self):
        code = """
        __global__ void normalize_beam(float *in, int total_beam)
        {
            int idx = threadIdx.x + blockIdx.x*blockDim.x;
            if(idx >= total_beam) return;

            //normalize
            float d = sqrt(in[idx*6+4]*in[idx*6+4]+in[idx*6+5]*in[idx*6+5]);
            in[idx*6+4] = in[idx*6+4]/d;
            in[idx*6+5] = in[idx*6+5]/d;
        }"""
        return code
    
    def GPU_normalize_beam(self, beams):
        #eg a = lights(...); a.GPU_normalize_beam(a.beams)
        beams = beams.astype(np.float32)
        beams_cnt = beams.shape[0]
        
        # GPU in/output prepare
        gpu_in = cuda.mem_alloc(beams.nbytes)
        total_beams = np.int32(beams_cnt)
        mod = SourceModule(self.cuda_normalize_beam())
        func = mod.get_function("normalize_beam")
        grid = (int((beams_cnt+1023)/1024), 1) 
        block = (1024, 1, 1)
        func.prepare("Pi") 
        
        # calculate normalize beams' direction vector
        cuda.memcpy_htod(gpu_in, beams)
        func.prepared_call(grid, block, gpu_in, total_beams)
        cuda.memcpy_dtoh(beams, gpu_in)
        
        return beams
    
    def normalize_beam(self):
        #eg lights(...).normalize_beam()
        self.beams = self.GPU_normalize_beam(self.beams)
        
    def add_beams_xyz(self, x, y, z):
        for i in range(self.beams.shape[0]):
            self.beams[i][2] = self.beams[i][2] + x
            self.beams[i][3] = self.beams[i][3] + y
            
    def add_beams_energy(self, eng):
        for i in range(self.beams.shape[0]):
            self.beams[i][1] = self.beams[i][1] + eng
    
    def set_beams_wavelength(self, length):
        for i in range(self.beams.shape[0]):
            self.beams[i][0] = length