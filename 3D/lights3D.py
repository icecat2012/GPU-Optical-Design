import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

class lights:
    def __init__(self, beams=None):
        #eg lights(np.array([[0.484, 100, 0,0,0, 1,0,0], [0.485, 50, 0,0,0, 0,1,0]]))
        self.beams = beams #[[wavelength, energy, start_x, start_y, start_z, direction_x, direction_y, direction_z], ...]
        
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
    
    def light_distribution(self, sample_ratio=1.0, az=45, el=45):
        #eg lights(...).energy_distribution(0.7, 0, 90)
        # sample data
        rows = self.beams.shape[0]
        samples = self.beams[np.random.choice(rows, int(rows*sample_ratio), replace=False), :]
        xmin = min(np.min(samples[:,2]), np.min(samples[:,2]+samples[:,5]))
        xmax = max(np.max(samples[:,2]), np.max(samples[:,2]+samples[:,5]))
        ymin = min(np.min(samples[:,3]), np.min(samples[:,3]+samples[:,6]))
        ymax = max(np.max(samples[:,3]), np.max(samples[:,3]+samples[:,6]))
        zmin = min(np.min(samples[:,4]), np.min(samples[:,4]+samples[:,7]))
        zmax = max(np.max(samples[:,4]), np.max(samples[:,4]+samples[:,7]))
        
        # plot data
        fig = plt.figure(figsize=(16,16))
        ax = fig.add_subplot(111, projection='3d')
        ax.quiver(samples[:,2], samples[:,3], samples[:,4], samples[:,5], samples[:,6], samples[:,7], arrow_length_ratio = 0.02, label='sample ratio {}'.format(sample_ratio), color='blue')
        plt.title('light_distribution')
        plt.legend(loc='best')
        plt.axis('equal')
        ax.view_init(azim=az, elev=el)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_xlim(xmin-0.1, xmax+0.1)
        ax.set_ylim(ymin-0.1, ymax+0.1)
        ax.set_zlim(zmin-0.1, zmax+0.1)
        plt.show()
    
    def cuda_normalize_beam(self):
        code = """
        __global__ void normalize_beam(float *in, int total_beam)
        {
            int idx = threadIdx.x + blockIdx.x*blockDim.x;
            if(idx >= total_beam) return;

            //normalize
            float d = sqrt(in[idx*8+5]*in[idx*8+5]+in[idx*8+6]*in[idx*8+6]+in[idx*8+7]*in[idx*8+7]);
            in[idx*8+5] = in[idx*8+5]/d;
            in[idx*8+6] = in[idx*8+6]/d;
            in[idx*8+7] = in[idx*8+7]/d;
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
            self.beams[i][4] = self.beams[i][4] + z
            
    def add_beams_energy(self, eng):
        for i in range(self.beams.shape[0]):
            self.beams[i][1] = self.beams[i][1] + eng
    
    def set_beams_wavelength(self, length):
        for i in range(self.beams.shape[0]):
            self.beams[i][0] = length
    
    def random_beams(self, count, position_var, direction_var, energy_var):
        now = 0
        self.beams = zeros((count,8))
        while now < count:
            break
        pass