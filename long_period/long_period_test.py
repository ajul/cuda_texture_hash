import long_period

import scipy.ndimage
import scipy.misc
import pycuda.autoinit
import pycuda.driver as cuda
import numpy as np

from pycuda.compiler import SourceModule

resolution = 512

header = long_period.generateHeader(256, 4)

body = """
extern "C"
__global__ void test_greyscale(float4* result, unsigned int resolution) {
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int i = y * resolution + x;
    if (i >= resolution * resolution) return;
    unsigned int c = long_period_hash(x, y, 0);
    result[i].x = c;
    result[i].y = c;
    result[i].z = c;
    result[i].w = 255.0f;
}

extern "C"
__global__ void test_color(float4* result, unsigned int resolution) {
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int i = y * resolution + x;
    if (i >= resolution * resolution) return;
    result[i].x = long_period_hash(x, y, 0);
    result[i].y = long_period_hash(x, y, 1);
    result[i].z = long_period_hash(x, y, 2);
    result[i].w = 255.0f;
}
"""

mod = SourceModule(header + body, no_extern_c=True)

resultGreyscale = np.zeros((resolution, resolution, 4), dtype=np.float32)
resultColor = np.zeros((resolution, resolution, 4), dtype=np.float32)

mod.get_function('test_greyscale')(
    cuda.Out(resultGreyscale),
    np.int32(resolution),
    block = (512, 1, 1),
    grid = ((resolution * resolution - 1) // 512 + 1, 1, 1),
    )

mod.get_function('test_color')(
    cuda.Out(resultColor),
    np.int32(resolution),
    block = (512, 1, 1),
    grid = ((resolution * resolution - 1) // 512 + 1, 1, 1),
    )

scipy.misc.imsave('test_greyscale.png', resultGreyscale)
scipy.misc.imsave('test_color.png', resultColor)
