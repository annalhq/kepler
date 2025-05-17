// Tanh kernel using 2D indexing and float4 vectorization.
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__device__ __forceinline__ float4 tanh4(float4 v) {
     return make_float4(tanhf(v.x),
                            tanhf(v.y),
                            tanhf(v.z),
                            tanhf(v.w));
}

// Kernel: each thread loads/stores one float4
// input, output: pointers to linear float arrays of size width*height
// width must be a multiple of 4
__global__ void tanh_kernel_f4(const float* __restrict__ input,
                                      float* __restrict__ output,
                                      int width,
                                      int height)
{
     int vx = blockIdx.x * blockDim.x + threadIdx.x;
     int y  = blockIdx.y * blockDim.y + threadIdx.y;

     int vecWidth = width >> 2;
     if (vx >= vecWidth || y >= height) return;

     int idx = y * vecWidth + vx;
     const float4* in4  = reinterpret_cast<const float4*>(input);
     float4        val  = in4[idx];
     float4        out4 = tanh4(val);
     float4*       o4   = reinterpret_cast<float4*>(output);
     o4[idx] = out4;
}

void launchTanhKernel(const float* input_d,
                           float*       output_d,
                           int          width,
                           int          height,
                           cudaStream_t stream = 0)
{
     dim3 block(16, 16);
     int vx = (width >> 2);
     dim3 grid( (vx + block.x - 1) / block.x,
                  (height + block.y - 1) / block.y );
     tanh_kernel_f4<<<grid, block, 0, stream>>>(input_d, output_d, width, height);
}

int main() {
     const int width = 1024;
     const int height = 768;
     float* input_d;
     float* output_d;

     cudaMalloc(&input_d, width * height * sizeof(float));
     cudaMalloc(&output_d, width * height * sizeof(float));

     launchTanhKernel(input_d, output_d, width, height);

     cudaFree(input_d);
     cudaFree(output_d);

     return 0;
}