import numpy as np
import matplotlib.pyplot as plt
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

class lens:
    def __init__(self, size, air_n=1, glass_n=1.4860):
        #eg lens(size=90)
        self.size = size 
        self.outer_lens = np.zeros(size+1)
        self.inner_lens = np.zeros(size+1)
        
        self.outer_surface = None
        self.inner_surface = None
        self.connet_surface = None
        
        self.refractive_index_out = air_n
        self.refractive_index_self = glass_n
        
    def get_surfaces(self):
        return np.concatenate((self.outer_surface, np.concatenate((self.inner_surface, self.connet_surface), axis=0)), axis=0)
        
    def sample_convex_lens(self):
        for i in range(self.outer_lens.shape[0]):
            self.outer_lens[i] = 2
        for i in range(self.inner_lens.shape[0]):
            self.inner_lens[i] = 1
                    
    def show_lens(self):
        xs, ys = list(), list()
        gap = 90.0/self.size
        for i in range(self.inner_lens.shape[0]):
            xs.append(self.inner_lens[i]*np.cos(np.pi*gap*i/180))
            ys.append(self.inner_lens[i]*np.sin(np.pi*gap*i/180))
        xs = xs[::-1]
        ys = ys[::-1]
        for i in range(self.outer_lens.shape[0]):
            xs.append(self.outer_lens[i]*np.cos(np.pi*gap*i/180))
            ys.append(self.outer_lens[i]*np.sin(np.pi*gap*i/180))
        xs = xs + [-1*i for i in xs[::-1]]
        ys = ys + ys[::-1]
        plt.figure()
        plt.plot(xs, ys,'b-')
        plt.grid(True)
        plt.show()
    
    def cuda_surface(self):
        code = """
        inline __host__ __device__ float2 findC(float a, float b, float gap, float alpha)
        {
            float x = (a*cospif(gap)-b)*cospif(alpha)-a*sinpif(gap)*sinpif(alpha);
            float y = a*cospif(alpha)*sinpif(gap)+(a*cospif(gap)-b)*sinpif(alpha);
            return make_float2(x, y);
        }
        __global__ void node2surface(float *in, float *out, int size, float gap, float quadrant, int surf_inv)
        {
            int idx = threadIdx.x + blockIdx.x*blockDim.x;
            if(idx >= size) return;
            
            //surface
            float alpha = gap*idx, now_x = quadrant*in[idx]*cospif(alpha/180), now_y = in[idx]*sinpif(alpha/180);
            
            float2 ans = findC(in[idx+1], in[idx], gap/180, alpha/180);
            out[idx*3*3+0] = quadrant*surf_inv*ans.y;
            out[idx*3*3+1] = -surf_inv*ans.x;
            out[idx*3*3+2] = -surf_inv*(quadrant*ans.y*now_x - ans.x*now_y);

            //two boundary
            out[idx*3*3+3*1+0] = quadrant*ans.x;
            out[idx*3*3+3*1+1] = ans.y;
            out[idx*3*3+3*1+2] = -(quadrant*ans.x*now_x + ans.y*now_y);
            
            now_x = quadrant*in[idx+1]*cospif((alpha+gap)/180), now_y = in[idx+1]*sinpif((alpha+gap)/180);
            out[idx*3*3+3*2+0] = -quadrant*ans.x;
            out[idx*3*3+3*2+1] = -ans.y;
            out[idx*3*3+3*2+2] = (quadrant*ans.x*now_x + ans.y*now_y);
            
        }"""
        return code
    
    def GPU_surface(self, nodes, surf_inv=1):
        #eg a=lens(900); a.GPU_surface(a.outer_lens)
        nodes = nodes.astype(np.float32)
        surfaces_cnt = nodes.shape[0]-1
        surfaces1 = np.zeros((surfaces_cnt,9), dtype=np.float32)
        surfaces2 = np.zeros((surfaces_cnt,9), dtype=np.float32)
        
        # GPU in/output prepare
        gpu_in = cuda.mem_alloc(nodes.nbytes)
        gpu_out = cuda.mem_alloc(surfaces1.nbytes)
        size = np.int32(surfaces_cnt)
        gap = np.float32(90.0/size)
        mod = SourceModule(self.cuda_surface())
        func = mod.get_function("node2surface")
        grid = (int((surfaces_cnt+1023)/1024), 1) 
        block = (1024, 1, 1)
        func.prepare("PPiffi") 
        
        # calculate first Quadrant
        cuda.memcpy_htod(gpu_in, nodes)
        func.prepared_call(grid, block, gpu_in, gpu_out, size, gap, np.float32(1), np.int32(surf_inv))
        cuda.memcpy_dtoh(surfaces1, gpu_out)
        
        # calculate second Quadrant
        cuda.memcpy_htod(gpu_in, nodes)
        func.prepared_call(grid, block, gpu_in, gpu_out, size, gap, np.float32(-1), np.int32(surf_inv))
        cuda.memcpy_dtoh(surfaces2, gpu_out)
        
        return np.concatenate((surfaces1, surfaces2), axis=0)
    
    def build_surface(self):
        # lens((4,4)).build_surface()
        self.outer_surface = self.GPU_surface(self.outer_lens, 1)
        self.inner_surface = self.GPU_surface(self.inner_lens, -1)
        self.connet_surface = np.array([[0,-1,0, 1,0,-self.inner_lens[0], -1,0,-self.outer_lens[0]],
                                        [0,-1,0, 1,0,self.outer_lens[0], -1,0,self.inner_lens[0]]])
    
    def cuda_normalize_surface(self):
        code = """
        __global__ void normalize_surface(float *in, int total_surface)
        {
            int idx = threadIdx.x + blockIdx.x*blockDim.x;
            if(idx >= total_surface) return;

            //normalize
            float d = sqrt(in[idx*3+0]*in[idx*3+0]+in[idx*3+1]*in[idx*3+1]);
            in[idx*3+0] = in[idx*3+0]/d;
            in[idx*3+1] = in[idx*3+1]/d;
            in[idx*3+2] = in[idx*3+2]/d;
        }"""
        return code
    
    def GPU_normalize_surface(self, surfaces):
        #eg a=lens(900); a.GPU_normalize_surface(a.outer_surface)
        surfaces = surfaces.astype(np.float32)
        surfaces_cnt = surfaces.shape[0]*3
        
        # GPU in/output prepare
        gpu_in = cuda.mem_alloc(surfaces.nbytes)
        total_surface = np.int32(surfaces_cnt)
        mod = SourceModule(self.cuda_normalize_surface())
        func = mod.get_function("normalize_surface")
        grid = (int((surfaces_cnt+1023)/1024), 1) 
        block = (1024, 1, 1)
        func.prepare("Pi") 
        
        # calculate normalize surfaces
        cuda.memcpy_htod(gpu_in, surfaces)
        func.prepared_call(grid, block, gpu_in, total_surface)
        cuda.memcpy_dtoh(surfaces, gpu_in)
        
        return surfaces
    
    def normalize_surface(self):
        # lens(900).sample_convex_lens().build_surface().normalize_surface()
        self.outer_surface = self.GPU_normalize_surface(self.outer_surface)
        self.inner_surface = self.GPU_normalize_surface(self.inner_surface)
        self.connet_surface = self.GPU_normalize_surface(self.connet_surface)