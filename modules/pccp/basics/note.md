# Basics

1. The qualifier `__global__` tells the compiler that the function will be called from the CPU and executed on the GPU

2. Triple angle brackets mark a call from the host thread to the code on the device side. A kernel is executed by an array of threads and all threads run the same code. The parameters within the triple angle brackets are the execution confi guration, which specifi es how many threads will execute the kernel. In the following example, it will run 10 GPU threads

     ```c
     #include <iostream>

     __global__ void kernel(){
          printf("From the GPU\n");
     }

     int main()
     {
          kernel<<<1,10>>>();
          cudaDeviceReset();
          return 0;
     }
     ```

3. `cudaDeviceReset()` will explicitly destroy and clean up all resources associated with
the current device in the current process.

4. CUDA program stucture:
     1. allocate gpu memory
     2. copy data from cpu mem to gpu mem
     3. invoke the cuda kernel to perform computation
     4. copy data back from gpu to cpu mem
     5. destroy gpu memories

## Aspects to CUDA programmning

1. **Locality:** It refers to the reuse of data so to reduce memory access latency. 

2. Two types of reference locality:
     - **Temporal locality:** refers to reuse of data/resources within relatively small time durations
     - **Spatial locality:** refers to use of data elements withins relatively close storage locality

3. Modern CPU architectures use large *cache* with good spatial and temporal locality

4. As CPU does not expose how threads are being scheduled on the underlying arch, we must handle low-level cache optimizations

5. CUDA enables concepts of both memory hierarchy and thread hierarhy

6. **Shared memory:** It is exposed by the CUDA programming. It can be thought of as a software-managed cache.

> [!NOTE]
>It provides great speedup by conserving bandwidth to main memory. With shared memory, we can control the locality of our code directly.

7. In ANSI C, for parallel programming we explicitly organize threads using `pthreads` or `OpenMP`. Whereas in CUDA it just write a piece of serial code to be called by only one thread. The GPU takes this kernel and makes it parallel by launching thousands of threads, all performing that computation.

8. The CUDA programming model provides you with a way to organize your threads 
hierarchically.

9. CUDA abstracts away the hardware details and does not require applications to be mapped to traditional graphics APIs

10. Three key abstractions: 
     1. Hierarchy of thread groups
     2. Hierarchy of memory groups
     3. Barrier synchronization

---

**TODO:**
1. GPU Architectures (Fermi and Kepler)

### References: 

1. Chapter 1, Heterogeneous Parallel Computing with CUDA. Professional CUDA C programming by John Cheng, Max Grossman, Ty McKercher