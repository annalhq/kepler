#include <iostream>

// The __global__ keyword marks a function to run on the GPU device
__global__ void kernel(){}

// The <<<...>>> syntax specifies grid and block configuration for the kernel
int main()
{
     kernel<<<1,1>>>();
     printf("Hello, World!");
     return 0;
}