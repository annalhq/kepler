// modified cuda sample from v8.0 docs

#include <prof.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cutil_inline.h>
#include <cuda_runtime.h>

extern"C" void fwtCPU(float *h_Output, float *h_Input, int log2N);
extern"C" void slowWTcpu(float *h_Output, float *h_Input, int log2N);
extern "C" void dyadicConvolutionCPU(
    float *h_Result,
    float *h_Data,
    float *h_Kernel,
    int log2dataN,
    int log2kernelN
);

#include "fastWalshTransform_kernel.cu"

const int log2Kernel = 7;

#ifndef __DEVICE_EMULATION__
    const   int log2Data = 23;
#else
    const   int log2Data = 15;
#endif
const int   dataN = 1 << log2Data;
const int kernelN = 1 << log2Kernel;

const int   DATA_SIZE = dataN   * sizeof(float);
const int KERNEL_SIZE = kernelN * sizeof(float);

const double NOPS = 3.0 * (double)dataN * (double)log2Data / 2.0;

int main(int argc, char *argv[]){
	GpuProfiling::initProf();
    float
        *h_Data,
        *h_Kernel,
        *h_ResultCPU,
        *h_ResultGPU;

    float
        *d_Data,
        *d_Kernel;

    double
        delta, ref, sum_delta2, sum_ref2, L2norm, gpuTime;

    cudaEvent_t start_event, stop_event;
    int i;

    if( cutCheckCmdLineFlag(argc, (const char**)argv, "device") )
        cutilDeviceInit(argc, argv);
    else
        cudaSetDevice( cutGetMaxGflopsDeviceId() );

    cutilSafeCall(cudaEventCreate(&start_event));
    cutilSafeCall(cudaEventCreate(&stop_event));

    printf("Initializing data...\n");
        printf("...allocating CPU memory\n");
        cutilSafeCall(cudaMallocHost((void **)&h_Kernel, KERNEL_SIZE));
        cutilSafeCall(cudaMallocHost((void **)&h_Data, DATA_SIZE));
        h_ResultCPU = (float *)malloc(DATA_SIZE);
        if (h_ResultCPU == NULL) {
            fprintf(stderr, "Failed to allocate h_ResultCPU!\n");
            exit(EXIT_FAILURE);
        }
        cutilSafeCall(cudaMallocHost((void **)&h_ResultGPU, DATA_SIZE));

        printf("...allocating GPU memory\n");
        cutilSafeCall(cudaMalloc((void **)&d_Kernel, DATA_SIZE));
        cutilSafeCall(cudaMalloc((void **)&d_Data,   DATA_SIZE));

        printf("...generating data\n");
        printf("Data length: %i; kernel length: %i\n", dataN, kernelN);
        srand(2007);
        for (i = 0; i < kernelN; i++)
            h_Kernel[i] = (float)rand() / (float)RAND_MAX;

        for (i = 0; i < dataN; i++)
            h_Data[i] = (float)rand() / (float)RAND_MAX;

        cutilSafeCall(cudaMemset(d_Kernel, 0, DATA_SIZE));
        cutilSafeCall(cudaMemcpy(d_Kernel, h_Kernel, KERNEL_SIZE, cudaMemcpyHostToDevice));
        cutilSafeCall(cudaMemcpy(d_Data,   h_Data,     DATA_SIZE, cudaMemcpyHostToDevice));

    printf("Running GPU dyadic convolution using Fast Walsh Transform...\n");
    
    cutilSafeCall(cudaDeviceSynchronize());
    cutilSafeCall(cudaEventRecord(start_event, 0));

    fwtBatchGPU(d_Data, 1, log2Data);
    fwtBatchGPU(d_Kernel, 1, log2Data);
    modulateGPU(d_Data, d_Kernel, dataN);
    fwtBatchGPU(d_Data, 1, log2Data);

    cutilSafeCall(cudaEventRecord(stop_event, 0));
    cutilSafeCall(cudaEventSynchronize(stop_event));
    
    float milliseconds = 0;
    cutilSafeCall(cudaEventElapsedTime(&milliseconds, start_event, stop_event));
    gpuTime = (double)milliseconds;

    printf("GPU time: %f ms; GOP/s: %f\n", gpuTime, NOPS / (gpuTime * 0.001 * 1E+9));

    printf("Reading back GPU results...\n");
    cutilSafeCall(cudaMemcpy(h_ResultGPU, d_Data, DATA_SIZE, cudaMemcpyDeviceToHost));

    printf("Running straightforward CPU dyadic convolution...\n");
    dyadicConvolutionCPU(h_ResultCPU, h_Data, h_Kernel, log2Data, log2Kernel);

    printf("Comparing the results...\n");
        sum_delta2 = 0;
        sum_ref2   = 0;
        for(i = 0; i < dataN; i++){
            delta       = h_ResultCPU[i] - h_ResultGPU[i];
            ref         = h_ResultCPU[i];
            sum_delta2 += delta * delta;
            sum_ref2   += ref * ref;
        }
        L2norm = sqrt(sum_delta2 / sum_ref2);
        printf("L2 norm: %E\n", L2norm);
    printf((L2norm < 1e-6) ? "PASSED\n" : "FAILED\n");
	GpuProfiling::printResults();

    printf("Shutting down...\n");
        cutilSafeCall(cudaEventDestroy(start_event));
        cutilSafeCall(cudaEventDestroy(stop_event));
        cutilSafeCall(cudaFree(d_Data));
        cutilSafeCall(cudaFree(d_Kernel));
        cutilSafeCall(cudaFreeHost(h_ResultGPU));
        free(h_ResultCPU);
        cutilSafeCall(cudaFreeHost(h_Data));
        cutilSafeCall(cudaFreeHost(h_Kernel));

    cutilSafeCall(cudaDeviceReset());
    return 0;
}