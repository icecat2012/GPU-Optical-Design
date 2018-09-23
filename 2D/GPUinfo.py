import pycuda.driver as cuda
import pycuda.autoinit

def GPUinfo():
    cuda.Device.count()
    dev = cuda.Device(0)
    dev.get_attributes()
    #print(dev.MAX_THREADS_PER_BLOCK)
    #print(dev.MAX_BLOCK_DIM_X)
    #print(dev.MAX_BLOCK_DIM_Y)
    #print(dev.MAX_BLOCK_DIM_Z)