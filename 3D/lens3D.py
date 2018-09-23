import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib import cm
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

class lens:
    def __init__(self, size, air_n=1, glass_n=1.4860):
        #eg lens(size=(10,10))
        self.size = size 
        self.upper_lens = np.zeros(size)
        self.lower_lens = np.zeros(size)
        
        self.upper_surface = None
        self.lower_surface = None
        
        self.refractive_index_out = air_n
        self.refractive_index_self = glass_n
        
    def get_surfaces(self):
        surfaces = list()
        for u_sur, l_sur in zip(self.upper_surface, self.lower_surface):
            if np.dot(u_sur[:3], l_sur[:3])==-1.0 and np.array_equal(u_sur[4:], l_sur[4:]):
                continue
            surfaces.append(u_sur)
            surfaces.append(l_sur)
        
        return np.array(surfaces)
        
    def sample_convex_lens(self):
        #eg lens(size=(10,10)).sample_convex_lens()
        center1 = self.size[0]/2 - 0.5
        center2 = self.size[1]/2 - 0.5
        (hight, wide) = self.size
        
        #build upper_lens
        for i in range(hight):
            for j in range(wide):
                if i%2==1:
                    z = ((center1+center2)/2)**2-(i-center1)**2-(j+0.5-center2)**2
                else:
                    z = ((center1+center2)/2)**2-(i-center1)**2-(j-center2)**2
                
                if z <= 0:
                    self.upper_lens[i,j] = 0
                else:
                    self.upper_lens[i,j] = np.sqrt(z)

        #build lower_lens
        for i in range(hight):
            for j in range(wide):
                if i%2==1:
                    z = ((center1+center2)/2)**2-(i-center1)**2-(j+0.5-center2)**2
                else:
                    z = ((center1+center2)/2)**2-(i-center1)**2-(j-center2)**2
                
                if z <= 0:
                    self.lower_lens[i,j] = 0
                else:
                    self.lower_lens[i,j] = -np.sqrt(z)
                    
    def show_lens(self, img='surface',az=45, el=45):
        #eg lens(size=(10,10)).sample_convex_lens().show_lens('wireframe', 0, 0)
        center1 = (self.size[0]+2)/2 + 0.5
        center2 = (self.size[1]+2)/2 + 0.5
        (hight, wide) = self.size
        
        # calculate X Y Z for plot
        X,Y = np.meshgrid(np.arange((-center2+1),(center2)), np.arange((-center1+1),(center1)))
        Z1 = np.zeros((hight+2,wide+2))
        Z2 = np.zeros((hight+2,wide+2))
        Z1[1:-1,1:-1] = self.upper_lens
        Z2[1:-1,1:-1] = self.lower_lens
        
        # X odd row shift 0.5
        for i in range(X.shape[0]):
            if i%2==0:
                X[i] += 0.5
        
        # plot
        fig = plt.figure(figsize=(16,16))
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        if img == 'surface':
            ax.plot_trisurf(X.flatten(), Y.flatten(), Z2.flatten(), linewidth=0.5, antialiased=False, cmap=cm.Blues_r)
            ax.plot_trisurf(X.flatten(), Y.flatten(), Z1.flatten(), linewidth=0.5, antialiased=False, cmap=cm.Reds)
        elif img == 'wireframe':
            ax.plot_trisurf(X.flatten(), Y.flatten(), Z2.flatten(), linewidth=0.5, antialiased=False, edgecolor='Black', alpha=0)
            ax.plot_trisurf(X.flatten(), Y.flatten(), Z1.flatten(), linewidth=0.5, antialiased=False, edgecolor='Black', alpha=0)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.view_init(azim=az, elev=el)
        plt.title(img)
        plt.axis('equal')
        plt.show()
        
    def draw_triangle_boundary(self, surfaces):
        #eg a = lens((4,4)).build_surface(); a.draw_triangle_boundary(a.lower_surface)
        plt.figure(figsize=(8,8))
        for idx in range(surfaces.shape[0]):
            x1 = (surfaces[idx,7]*surfaces[idx,9]-surfaces[idx,5]*surfaces[idx,11])/(surfaces[idx,5]*surfaces[idx,8]-surfaces[idx,4]*surfaces[idx,9])
            y1 = -(surfaces[idx,4]*x1+surfaces[idx,7])/surfaces[idx,5]
            x2 = (surfaces[idx,7]*surfaces[idx,13]-surfaces[idx,5]*surfaces[idx,15])/(surfaces[idx,5]*surfaces[idx,12]-surfaces[idx,4]*surfaces[idx,13])
            y2 = -(surfaces[idx,4]*x2+surfaces[idx,7])/surfaces[idx,5]
            x3 = (surfaces[idx,11]*surfaces[idx,13]-surfaces[idx,9]*surfaces[idx,15])/(surfaces[idx,9]*surfaces[idx,12]-surfaces[idx,8]*surfaces[idx,13])
            y3 = -(surfaces[idx,8]*x3+surfaces[idx,11])/surfaces[idx,9]
            plt.plot([x1,x2,x3,x1],[y1,y2,y3,y1],label='{}'.format(idx))
        plt.legend(loc='best')
        plt.axis('equal')
        plt.show()
    
    def cuda_surface(self):
        code = """
        __device__ inline float3 cross(float ax, float ay, float az, float bx, float by, float bz)
        {
            return make_float3(ay*bz-az*by, az*bx-ax*bz, ax*by-ay*bx);
        }
        __device__ inline float2 xy_rot180(int row, int col, int bound_inv, int idx, int bias)
        {
            int y = ((col + bound_inv*__float2int_rn(idx/col)+bias)%col);
            float x = ((row + bound_inv*(idx%col)+bias)%row+(y%2)*0.5);
            return make_float2(x, y);
        }
        __global__ void node2surface(float *in, float *out, int row, int col, int bound_inv, int surf_inv)
        {
            int idx = threadIdx.x + blockIdx.x*blockDim.x;
            if(idx >= (row-1)*col) return;
            int bias = ((bound_inv+1)/2)-1;

            //surface
            float2 xy = xy_rot180(row, col, bound_inv, idx, bias);
            float3 ans = cross(1., 0., in[idx+1]-in[idx], 0.5, 1., in[idx+col]-in[idx]);
            out[idx*4*4+0] = surf_inv*ans.x;
            out[idx*4*4+1] = surf_inv*ans.y;
            out[idx*4*4+2] = surf_inv*ans.z;
            out[idx*4*4+3] = -surf_inv*(ans.x*xy.x + ans.y*xy.y + ans.z*in[idx]);

            //three boundary
            ans = cross(0.5, 1., in[idx+col]-in[idx], 0., 0., 1.);
            out[idx*4*4+4*1+0] = bound_inv*ans.x;
            out[idx*4*4+4*1+1] = bound_inv*ans.y;
            out[idx*4*4+4*1+2] = bound_inv*ans.z;
            out[idx*4*4+4*1+3] = -(bound_inv*ans.x*xy.x + bound_inv*ans.y*xy.y + bound_inv*ans.z*in[idx]);

            ans = make_float3(0., 1., 0.); //ans = cross(0., 0., 1., 1., 0., in[idx+1]-in[idx]);
            out[idx*4*4+4*2+0] = bound_inv*ans.x;
            out[idx*4*4+4*2+1] = bound_inv*ans.y;
            out[idx*4*4+4*2+2] = bound_inv*ans.z;
            out[idx*4*4+4*2+3] = -(bound_inv*ans.x*xy.x + bound_inv*ans.y*xy.y + bound_inv*ans.z*in[idx]);

            xy = xy_rot180(row, col, bound_inv, idx+1, bias);
            ans = cross(0., 0., 1., -0.5, 1., in[idx+col]-in[idx+1]);
            out[idx*4*4+4*3+0] = bound_inv*ans.x;
            out[idx*4*4+4*3+1] = bound_inv*ans.y;
            out[idx*4*4+4*3+2] = bound_inv*ans.z;
            out[idx*4*4+4*3+3] = -(bound_inv*ans.x*xy.x + bound_inv*ans.y*xy.y + bound_inv*ans.z*in[idx+1]);
        }"""
        return code
    
    def GPU_surface(self, nodes, surf_inv=1):
        #eg a=lens((4,4)); a.GPU_surface(a.upper_lens)
        nodes = nodes.astype(np.float32)
        surfaces_cnt = (nodes.shape[0]-1)*(nodes.shape[1])
        surfaces1 = np.zeros((surfaces_cnt,16), dtype=np.float32)
        surfaces2 = np.zeros((surfaces_cnt,16), dtype=np.float32)
        
        # GPU in/output prepare
        gpu_in = cuda.mem_alloc(nodes.nbytes)
        gpu_out = cuda.mem_alloc(surfaces1.nbytes)
        row = np.int32(nodes.shape[0])
        col = np.int32(nodes.shape[1])
        mod = SourceModule(self.cuda_surface())
        func = mod.get_function("node2surface")
        grid = (int((surfaces_cnt+1023)/1024), 1) 
        block = (1024, 1, 1)
        func.prepare("PPiiii") 
        
        # calculate regular triangles
        cuda.memcpy_htod(gpu_in, nodes)
        func.prepared_call(grid, block, gpu_in, gpu_out, row, col, np.int32(1), np.int32(surf_inv))
        cuda.memcpy_dtoh(surfaces1, gpu_out)
        # calculate upside-down triangle
        cuda.memcpy_htod(gpu_in, np.ascontiguousarray(np.rot90(nodes,2), dtype=np.float32))
        func.prepared_call(grid, block, gpu_in, gpu_out, row, col, np.int32(-1), np.int32(surf_inv))
        cuda.memcpy_dtoh(surfaces2, gpu_out)
        
        # delete redundant triangles
        delete_rows = list()
        for i in range(col-1, surfaces_cnt, col):
            delete_rows.append(i)
        surfaces1 = np.delete(surfaces1, delete_rows, 0)
        surfaces2 = np.delete(surfaces2, delete_rows, 0)
        return np.concatenate((surfaces1, surfaces2), axis=0)
    
    def build_surface(self):
        # lens((4,4)).build_surface()
        self.upper_surface = self.GPU_surface(self.upper_lens, 1)
        self.lower_surface = self.GPU_surface(self.lower_lens, -1)
    
    def cuda_normalize_surface(self):
        code = """
        __global__ void normalize_surface(float *in, int total_surface)
        {
            int idx = threadIdx.x + blockIdx.x*blockDim.x;
            if(idx >= total_surface) return;

            //normalize
            float d = sqrt(in[idx*4+0]*in[idx*4+0]+in[idx*4+1]*in[idx*4+1]+in[idx*4+2]*in[idx*4+2]);
            in[idx*4+0] = in[idx*4+0]/d;
            in[idx*4+1] = in[idx*4+1]/d;
            in[idx*4+2] = in[idx*4+2]/d;
            in[idx*4+3] = in[idx*4+3]/d;
        }"""
        return code
    
    def GPU_normalize_surface(self, surfaces):
        #eg a=lens((4,4)); a.GPU_normalize_surface(a.upper_surface)
        surfaces = surfaces.astype(np.float32)
        surfaces_cnt = surfaces.shape[0]*4
        
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
        # lens((4,4)).sample_convex_lens().build_surface().normalize_surface()
        self.upper_surface = self.GPU_normalize_surface(self.upper_surface)
        self.lower_surface = self.GPU_normalize_surface(self.lower_surface)