import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

class trace:
    def filter_out_len_beam(self, beams, out_len_beams):
        in_len_beams = list()
        all_zero = np.zeros(beams.shape[1])
        three_zero = np.zeros(3)
        for i in range(int(beams.shape[0]/2)):
            if np.array_equal(beams[i*2+1], all_zero):
                out_len_beams.append(beams[i*2])
            else:
                if not np.array_equal(beams[i*2][5:], three_zero):
                    in_len_beams.append(beams[i*2])
                if not np.array_equal(beams[i*2+1][5:], three_zero):
                    in_len_beams.append(beams[i*2+1])
        return np.array(in_len_beams), out_len_beams
    
    def cuda_reflect_refract(self):
        code = """
        inline __host__ __device__ float3 operator-(float3 a)
        {
            return make_float3(-a.x, -a.y, -a.z);
        }
        inline __host__ __device__ float3 operator+(float3 a, float3 b)
        {
            return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
        }
        inline __host__ __device__ float3 operator*(float b, float3 a)
        {
            return make_float3(a.x * b, a.y * b, a.z * b);
        }
        inline __host__ __device__ float dot(float3 &a, float3 &b)
        {
            return a.x * b.x + a.y * b.y + a.z * b.z;
        }
        __device__ inline float3 reflect(float3 v, float3 n)
        {
            float d = dot(v, n);
            v.x = v.x - 2*d*n.x;
            v.y = v.y - 2*d*n.y;
            v.z = v.z - 2*d*n.z;
            return v;
        }
        __device__ inline float3 refract(float3 v, float3 n, float r1=1.0, float r2=1.5)
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
            return k<0 ? make_float3(0, 0, 0) : (eta * v + (eta * cosi - sqrt(k)) * n);
        }
        __device__ inline float fresnel(float3 v, float3 n, float r1=1.0, float r2=1.5) 
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
        __device__ inline int under_surf(float px, float py, float pz, float pc, float x, float y, float z)
        {
            return (px*x+py*y+pz*z+pc) <= 0? 1 : 0;
        }
        __global__ void reflect_refract(float *in_surfaces, float *in_beams, float *out_beams, int surfaces_cnt, int beams_cnt, float air_n, float glass_n)
        {
            int idx = threadIdx.x + blockIdx.x*blockDim.x;
            if(idx >= beams_cnt) return;
            int i;
            float3 reflect_v, refract_v;
            float bias=1e-6f, record_distance=9999999999, x, y, z, wavelength, energy;
            float d, tmp_x, tmp_y, tmp_z;
            for(i=0; i<surfaces_cnt; ++i)
            {
                d = -(in_beams[idx*8+2]*in_surfaces[i*16+0]+in_beams[idx*8+3]*in_surfaces[i*16+1]+in_beams[idx*8+4]*in_surfaces[i*16+2]+in_surfaces[i*16+3])    \
                    /(in_beams[idx*8+5]*in_surfaces[i*16+0]+in_beams[idx*8+6]*in_surfaces[i*16+1]+in_beams[idx*8+7]*in_surfaces[i*16+2]+bias);
                tmp_x = in_beams[idx*8+5]*d + in_beams[idx*8+2];
                tmp_y = in_beams[idx*8+6]*d + in_beams[idx*8+3];
                tmp_z = in_beams[idx*8+7]*d + in_beams[idx*8+4];
                if( d > 0 && \
                (under_surf(in_surfaces[i*16+4], in_surfaces[i*16+5], in_surfaces[i*16+6], in_surfaces[i*16+7], tmp_x, tmp_y, tmp_z) \
                | under_surf(in_surfaces[i*16+8], in_surfaces[i*16+9], in_surfaces[i*16+10], in_surfaces[i*16+11], tmp_x, tmp_y, tmp_z) \
                | under_surf(in_surfaces[i*16+12], in_surfaces[i*16+13], in_surfaces[i*16+14], in_surfaces[i*16+15], tmp_x, tmp_y, tmp_z))==0 \
                && d*d*(in_beams[idx*8+5]*in_beams[idx*8+5]+in_beams[idx*8+6]*in_beams[idx*8+6]+in_beams[idx*8+7]*in_beams[idx*8+7]) < record_distance)
                {
                    //hit the surface
                    record_distance = d*d*(in_beams[idx*8+5]*in_beams[idx*8+5]+in_beams[idx*8+6]*in_beams[idx*8+6]+in_beams[idx*8+7]*in_beams[idx*8+7]);
                    x = tmp_x; y = tmp_y; z = tmp_z;
                    wavelength = in_beams[idx*8+0];
                    energy = in_beams[idx*8+1];

                    float3 v, N;
                    float kr;
                    v = make_float3(in_beams[idx*8+5], in_beams[idx*8+6], in_beams[idx*8+7]);
                    N = make_float3(in_surfaces[i*16+0], in_surfaces[i*16+1], in_surfaces[i*16+2]);

                    reflect_v = reflect(v, N);
                    refract_v = refract(v, N, air_n, glass_n);
                    kr = fresnel(v, N, air_n, glass_n);
                    out_beams[idx*2*8] = wavelength;
                    out_beams[idx*2*8+1] = energy * kr;
                    out_beams[idx*2*8+2] = x+reflect_v.x*bias;
                    out_beams[idx*2*8+3] = y+reflect_v.y*bias;
                    out_beams[idx*2*8+4] = z+reflect_v.z*bias;
                    out_beams[idx*2*8+5] = reflect_v.x;
                    out_beams[idx*2*8+6] = reflect_v.y;
                    out_beams[idx*2*8+7] = reflect_v.z;
                    out_beams[idx*2*8+8] = wavelength;
                    out_beams[idx*2*8+9] = energy * (1.0-kr);
                    out_beams[idx*2*8+10] = x+refract_v.x*bias;
                    out_beams[idx*2*8+11] = y+refract_v.y*bias;
                    out_beams[idx*2*8+12] = z+refract_v.z*bias;
                    out_beams[idx*2*8+13] = refract_v.x;
                    out_beams[idx*2*8+14] = refract_v.y;
                    out_beams[idx*2*8+15] = refract_v.z;
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
            out_beams = np.zeros((beams_cnt*2,8), dtype=np.float32)
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
            print('------------------')
            print(out_beams)
            print('##')
            print(beams)
            print('@@')
            print(np.array(out_len_beams))
            print('************************')

            gpu_in_b.free()
            gpu_out.free()

            max_reflect = max_reflect - 1

        return beams, np.array(out_len_beams)
        #return np.concatenate((beams, out_len_beams), axis=0)

    
    
    def cuda_beam_surface_intersect(self):
        code = """
        __global__ void beam_surface_intersect(float *in_surface, float *in_beams, float *out_beams, int beams_cnt)
        {
            int idx = threadIdx.x + blockIdx.x*blockDim.x;
            if(idx >= beams_cnt) return;
            float d, d_is_pos, bias=1e-6f;
            d = -(in_beams[idx*8+2]*in_surface[0] + in_beams[idx*8+3]*in_surface[1] + in_beams[idx*8+4]*in_surface[2] + in_surface[3]) / (in_beams[idx*8+5]*in_surface[0] + in_beams[idx*8+6]*in_surface[1] + in_beams[idx*8+7]*in_surface[2] + bias);
            
            d_is_pos = d>0? 1 : 0;
            out_beams[idx*8] = d_is_pos*in_beams[idx*8];
            out_beams[idx*8+1] = d_is_pos*in_beams[idx*8+1];
            out_beams[idx*8+2] = d_is_pos*(in_beams[idx*8+5]*d + in_beams[idx*8+2]);
            out_beams[idx*8+3] = d_is_pos*(in_beams[idx*8+6]*d + in_beams[idx*8+3]);
            out_beams[idx*8+4] = d_is_pos*(in_beams[idx*8+7]*d + in_beams[idx*8+4]);
            out_beams[idx*8+5] = d_is_pos*in_beams[idx*8+5];
            out_beams[idx*8+6] = d_is_pos*in_beams[idx*8+6];
            out_beams[idx*8+7] = d_is_pos*in_beams[idx*8+7];
        }"""
        return code
    
    def GPU_beam_surface_intersect(self, surface, beams):
        mod = SourceModule(self.cuda_beam_surface_intersect())
        func = mod.get_function("beam_surface_intersect")
        func.prepare("PPPi")

        #input prepare
        surface = surface.astype(np.float32)
        gpu_in_s = cuda.mem_alloc(surface.nbytes)
        cuda.memcpy_htod(gpu_in_s, surface)
        beams = beams.astype(np.float32)
        beams_cnt = beams.shape[0]
        out_beams = np.zeros((beams_cnt,8), dtype=np.float32)
        out_beams = beams
        gpu_in_b = cuda.mem_alloc(beams.nbytes)
        gpu_out = cuda.mem_alloc(out_beams.nbytes)
        cuda.memcpy_htod(gpu_in_b, beams)
        cuda.memcpy_htod(gpu_out, out_beams)
        grid = (int((beams_cnt+1023)/1024), 1) 
        block = (1024, 1, 1)
        
        # calculate
        func.prepared_call(grid, block, gpu_in_s, gpu_in_b, gpu_out, np.int32(beams_cnt))
        cuda.memcpy_dtoh(out_beams, gpu_out)
        
        #filter zero vectors
        output = list()
        all_zero = np.zeros(beams.shape[1])
        for i in range(out_beams.shape[0]):
            if np.array_equal(out_beams[i], all_zero):
                continue
            output.append(out_beams[i])

        return np.array(output)
    
    def uniformity(self, surface, beams, shape, width):
        hits = self.GPU_beam_surface_intersect(surface, beams)
        return hits