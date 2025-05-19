#include <iostream>

// The __global__ keyword marks a function to run on the GPU device
__global__ void kernel(){
     printf("From the GPU\n");
}

// The <<<...>>> syntax specifies grid and block configuration for the kernel
int main()
{
     kernel<<<1,10>>>();
     cudaDeviceReset();
     return 0;
}