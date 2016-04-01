import long_period

import scipy.ndimage
import scipy.misc
import pycuda.autoinit
import pycuda.driver as cuda
import numpy as np

from pycuda.compiler import SourceModule

resolution = 1024

blockSize = 16
gridSize = (resolution - 1) // blockSize + 1

header = long_period.generateHeader(256, 8, factorCount = 2)

f = open('long_period_hash.h', 'w')
f.write(header)
f.close()

body = """
extern "C"
__global__ void test_greyscale(float3* result, unsigned int resolution) {
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int i = y * resolution + x;
    if (x >= resolution) return;
    if (y >= resolution) return;
    unsigned int c = long_period_hash(x, y, 0);
    result[i].x = c;
    result[i].y = c;
    result[i].z = c;
}

extern "C"
__global__ void test_color(float3* result, unsigned int resolution) {
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int i = y * resolution + x;
    if (x >= resolution) return;
    if (y >= resolution) return;
    result[i].x = long_period_hash(x, y, 0);
    result[i].y = long_period_hash(x, y, 1);
    result[i].z = long_period_hash(x, y, 2);
}
"""

source = header + body

print(source)

mod = SourceModule(source, no_extern_c=True)

resultGreyscale = np.zeros((resolution, resolution, 3), dtype=np.float32)
resultColor = np.zeros((resolution, resolution, 3), dtype=np.float32)

mod.get_function('test_greyscale')(
    cuda.Out(resultGreyscale),
    np.int32(resolution),
    block = (blockSize, blockSize, 1),
    grid = (gridSize, gridSize, 1),
    )

mod.get_function('test_color')(
    cuda.Out(resultColor),
    np.int32(resolution),
    block = (blockSize, blockSize, 1),
    grid = (gridSize, gridSize, 1),
    )

scipy.misc.imsave('test_greyscale.png', resultGreyscale)
scipy.misc.imsave('test_color.png', resultColor)
