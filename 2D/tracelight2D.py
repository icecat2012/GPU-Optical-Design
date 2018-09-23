import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import matplotlib.pyplot as plt

class trace:
    def filter_out_len_beam(self, beams, out_len_beams):
        in_len_beams = list()
        all_zero = np.zeros(beams.shape[1])
        two_zero = np.zeros(2)
        for i in range(int(beams.shape[0]/2)):
            if np.array_equal(beams[i*2+1], all_zero):
                out_len_beams.append(beams[i*2])
            else:
                if not np.array_equal(beams[i*2][4:], two_zero):
                    in_len_beams.append(beams[i*2])
                if not np.array_equal(beams[i*2+1][4:], two_zero):
                    in_len_beams.append(beams[i*2+1])
        return np.array(in_len_beams), out_len_beams
    
    def cuda_reflect_refract(self):
        code = """
        inline __host__ __device__ float2 operator-(float2 a)
        {
            return make_float2(-a.x, -a.y);
        }
        inline __host__ __device__ float2 operator+(float2 a, float2 b)
        {
            return make_float2(a.x + b.x, a.y + b.y);
        }
        inline __host__ __device__ float2 operator*(float b, float2 a)
        {
            return make_float2(a.x * b, a.y * b);
        }
        inline __host__ __device__ float dot(float2 &a, float2 &b)
        {
            return a.x * b.x + a.y * b.y;
        }
        __device__ inline float2 reflect(float2 v, float2 n)
        {
            float d = dot(v, n);
            v.x = v.x - 2*d*n.x;
            v.y = v.y - 2*d*n.y;
            return v;
        }
        __device__ inline float2 refract(float2 v, float2 n, float r1=1.0, float r2=1.5)
        {
            float eta, cosi = dot(v, n), k;
            if(cosi < 0)
            {
                cosi = -cosi;
                eta = r1 / r2;
            }
            else
            {
                n = -n;
                eta = r2 / r1;
            }
            k = 1 - eta * eta * (1 - cosi * cosi);
            return k<0 ? make_float2(0, 0) : (eta * v + (eta * cosi - sqrt(k)) * n);
        }
        __device__ inline float fresnel(float2 v, float2 n, float r1=1.0, float r2=1.5) 
        { 
            float cosi = dot(v, n); 
            if(cosi > 0)
            {
                float tmp=r1;
                r1=r2;
                r2=tmp;
            }
            float eta = r1 / r2;
            float sint = eta * sqrt(max(0.f, 1 - cosi * cosi)); 

            float kr = 1;
            if(sint < 1) 
            {
                float cost = sqrt(max(0.f, 1 - sint * sint)); 
                cosi = fabsf(cosi); 
                float Rs = ((r2 * cosi) - (r1 * cost)) / ((r2 * cosi) + (r1 * cost)); 
                float Rp = ((r1 * cosi) - (r2 * cost)) / ((r1 * cosi) + (r2 * cost)); 
                kr = (Rs * Rs + Rp * Rp) / 2; 
            }
            return kr;
            // As a consequence of the conservation of energy, transmittance is given by:
            // kt = 1 - kr;
        }
        __global__ void reflect_refract(float *in_surfaces, float *in_beams, float *out_beams, int surfaces_cnt, int beams_cnt, float air_n, float glass_n)
        {
            int idx = threadIdx.x + blockIdx.x*blockDim.x;
            if(idx >= beams_cnt) return;
            int i;
            float2 reflect_v, refract_v;
            float bias=1e-5f, record_distance=9999999999, x, y, wavelength, energy;
            float d, tmp_x, tmp_y;
            for(i=0; i<surfaces_cnt; ++i)
            {
                d = -(in_beams[idx*6+2]*in_surfaces[i*9+0] + in_beams[idx*6+3]*in_surfaces[i*9+1] + in_surfaces[i*9+2]) / (in_beams[idx*6+4]*in_surfaces[i*9+0] + in_beams[idx*6+5]*in_surfaces[i*9+1]+bias);
                tmp_x = in_beams[idx*6+4]*d + in_beams[idx*6+2];
                tmp_y = in_beams[idx*6+5]*d + in_beams[idx*6+3];
                if( d > 0 \
                && (in_surfaces[i*9+3]*tmp_x + in_surfaces[i*9+4]*tmp_y + in_surfaces[i*9+5])>0 \
                && (in_surfaces[i*9+6]*tmp_x + in_surfaces[i*9+7]*tmp_y + in_surfaces[i*9+8])>0 \
                && d*d*(in_beams[idx*6+4]*in_beams[idx*6+4]+in_beams[idx*6+5]*in_beams[idx*6+5]) < record_distance)
                {
                    //hit the surface
                    record_distance = d*d*(in_beams[idx*6+4]*in_beams[idx*6+4] + in_beams[idx*6+5]*in_beams[idx*6+5]);
                    x = tmp_x; y = tmp_y;
                    wavelength = in_beams[idx*6+0];
                    energy = in_beams[idx*6+1];

                    float2 v, N;
                    float kr;
                    v = make_float2(in_beams[idx*6+4], in_beams[idx*6+5]);
                    N = make_float2(in_surfaces[i*9+0], in_surfaces[i*9+1]);

                    reflect_v = reflect(v, N);
                    refract_v = refract(v, N, air_n, glass_n);
                    kr = fresnel(v, N, air_n, glass_n);
                    out_beams[idx*2*6] = wavelength;
                    out_beams[idx*2*6+1] = energy * kr;
                    out_beams[idx*2*6+2] = x+reflect_v.x*bias;
                    out_beams[idx*2*6+3] = y+reflect_v.y*bias;
                    out_beams[idx*2*6+4] = reflect_v.x;
                    out_beams[idx*2*6+5] = reflect_v.y;
                    out_beams[idx*2*6+6] = wavelength;
                    out_beams[idx*2*6+7] = energy * (1.0-kr);
                    out_beams[idx*2*6+8] = x+refract_v.x*bias;
                    out_beams[idx*2*6+9] = y+refract_v.y*bias;
                    out_beams[idx*2*6+10] = refract_v.x;
                    out_beams[idx*2*6+11] = refract_v.y;
                }
            }
        }"""
        return code

    def GPU_reflect_refract(self, surfaces, beams, air_n=1.0, glass_n=1.48, max_reflect=10):
        # GPU in/output prepare
        out_len_beams = list()
        code = self.cuda_reflect_refract()
        mod = SourceModule(code)
        func = mod.get_function("reflect_refract")
        func.prepare("PPPiiff")

        #surface input prepare
        surfaces = surfaces.astype(np.float32)
        surfaces_cnt = surfaces.shape[0]
        gpu_in_s = cuda.mem_alloc(surfaces.nbytes)
        cuda.memcpy_htod(gpu_in_s, surfaces)

        while beams.shape[0] > 0 and max_reflect>0:
            #input prepare
            beams = beams.astype(np.float32)
            beams_cnt = beams.shape[0]
            out_beams = np.zeros((beams_cnt*2,6), dtype=np.float32)
            out_beams[::2] = beams
            grid = (int((beams_cnt+1023)/1024), 1) 
            block = (1024, 1, 1)

            #gpu input prepare
            gpu_in_b = cuda.mem_alloc(beams.nbytes)
            gpu_out = cuda.mem_alloc(out_beams.nbytes)
            cuda.memcpy_htod(gpu_in_b, beams)
            cuda.memcpy_htod(gpu_out, out_beams)

            # calculate
            func.prepared_call(grid, block, gpu_in_s, gpu_in_b, gpu_out, np.int32(surfaces_cnt), np.int32(beams_cnt), np.float32(air_n), np.float32(glass_n))
            cuda.memcpy_dtoh(out_beams, gpu_out)
            
            beams, out_len_beams = self.filter_out_len_beam(out_beams, out_len_beams)
            #print('------------------')
            #print(out_beams)
            #print('##')
            #print(beams)
            #print('@@')
            #print(np.array(out_len_beams))
            #print('************************')

            gpu_in_b.free()
            gpu_out.free()

            max_reflect = max_reflect - 1

        return beams, np.array(out_len_beams)
        #return np.concatenate((beams, out_len_beams), axis=0)

    
    
    def cuda_beam_surface_intersect(self):
        code = """
        __global__ void beam_surface_intersect(float *in_surface, float *in_beams, int beams_cnt)
        {
            int idx = threadIdx.x + blockIdx.x*blockDim.x;
            if(idx >= beams_cnt) return;
            float d, d_is_pos, bias=1e-5f;
            d = -(in_beams[idx*6+2]*in_surface[0] + in_beams[idx*6+3]*in_surface[1] + in_surface[2]) / (in_beams[idx*6+4]*in_surface[0] + in_beams[idx*6+5]*in_surface[1]+bias);
            
            d_is_pos = d>0? 1 : 0;
            in_beams[idx*6] = d_is_pos*in_beams[idx*6];
            in_beams[idx*6+1] = d_is_pos*in_beams[idx*6+1];
            in_beams[idx*6+2] = d_is_pos*(in_beams[idx*6+4]*d + in_beams[idx*8+2]);
            in_beams[idx*6+3] = d_is_pos*(in_beams[idx*6+5]*d + in_beams[idx*8+3]);
            in_beams[idx*6+4] = d_is_pos*in_beams[idx*6+4];
            in_beams[idx*6+5] = d_is_pos*in_beams[idx*6+5];
        }"""
        return code
    
    def GPU_beam_surface_intersect(self, surface, beams):
        mod = SourceModule(self.cuda_beam_surface_intersect())
        func = mod.get_function("beam_surface_intersect")
        func.prepare("PPi")

        #input prepare
        surface = surface.astype(np.float32)
        gpu_in_s = cuda.mem_alloc(surface.nbytes)
        cuda.memcpy_htod(gpu_in_s, surface)
        beams = beams.astype(np.float32)
        beams_cnt = beams.shape[0]
        gpu_in_b = cuda.mem_alloc(beams.nbytes)
        cuda.memcpy_htod(gpu_in_b, beams)
        grid = (int((beams_cnt+1023)/1024), 1) 
        block = (1024, 1, 1)
        
        # calculate
        func.prepared_call(grid, block, gpu_in_s, gpu_in_b, np.int32(beams_cnt))
        cuda.memcpy_dtoh(beams, gpu_in_b)
        
        #filter zero vectors
        output = list()
        all_zero = np.zeros(beams.shape[1])
        for i in range(beams.shape[0]):
            if np.array_equal(beams[i], all_zero):
                continue
            output.append(beams[i])

        return np.array(output)
    
    def uniformity(self, surface, beams, width, bins):
        hits = self.GPU_beam_surface_intersect(surface, beams)
        gap = width/bins
        mu = width/2
        hist = np.zeros(bins)
        for v in hits:
            if v[2]>width/2 or v[2]< -width/2:
                continue
            hist[int((v[2] + mu)/gap)] += v[1]
            
        return np.mean(hist)/np.max(hist), hist