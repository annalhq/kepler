#include <cstdint>
#include <torch/extension.h>

__global__ void rgb2gray(const uint8_t *__restrict__ input,
                         uint8_t *__restrict__ output, int width, int height) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height)
    return;

  const int idx = (y * width + x) * 3;
  const uint8_t r = input[idx];
  const uint8_t g = input[idx + 1];
  const uint8_t b = input[idx + 2];

  // Grayscale (luminance)
  const uint8_t gray = static_cast<uint8_t>(0.21f * r + 0.72f * g + 0.07f * b);
  output[y * width + x] = gray;
}

void rgb2gray(const torch::Tensor &input, torch::Tensor &output) {
  TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
  TORCH_CHECK(output.is_cuda(), "Output must be a CUDA tensor");
  TORCH_CHECK(input.is_contiguous(), "Input tensor must be contiguous");
  TORCH_CHECK(output.is_contiguous(), "Output tensor must be contiguous");

  const int height = input.size(0);
  const int width = input.size(1);

  const dim3 blockSize(16, 16);
  const dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                      (height + blockSize.y - 1) / blockSize.y);

  img2gray_kernel<<<gridSize, blockSize>>>(
      input.data_ptr<uint8_t>(), output.data_ptr<uint8_t>(), width, height);
  cudaDeviceSynchronize();
}
}
