#include <stdio.h>
// import more stuff
#include <stdlib.h>
#include <time.h>

__global__ void parallelSum(int* inputArray, int* outputResult, int arraySize) {
    extern __shared__ int sharedMemory[];
    int threadID = threadIdx.x;
    int globalID = blockIdx.x * blockDim.x + threadIdx.x;
    // Load data into shared memory
    sharedMemory[threadID] = (globalID < arraySize) ? inputArray[globalID] : 0;
    __syncthreads();
    // Perform parallel reduction using shared memory
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        int index = 2 * stride * threadID;
        if (index < blockDim.x) {
            sharedMemory[index] += sharedMemory[index + stride];
        }
        __syncthreads();
    }

    // Write the result back to global memory
    if (threadID == 0) {
        outputResult[blockIdx.x] = sharedMemory[0];
    }
}

int main() {
    // TODO : Write the main() function
    
    const int width = 1 << 13;

    printf("Using array size of %d\n", width);
    const int blockSize = 256; 
    const int numBlocks = (width + blockSize - 1) / blockSize;
    const int sharedMemorySize = blockSize * sizeof(int);
    const int iterations = 100; 

    size_t size = width * sizeof(int);

    // some timing stuff
    cudaEvent_t gpu_start, gpu_stop;
    cudaEventCreate(&gpu_start);
    cudaEventCreate(&gpu_stop);

    // host array
    int h_inputArray[width];

    int h_gpuSum = 0;
    int cpuSum = 0;

    // declare some device array
    int *d_inputArray, *d_outputArray, *d_finalResult;

    // TODO : Initialize input and output arrays with random values (host arrays)
    srand(time(NULL));
    for (int i = 0; i < width; i++) {
        h_inputArray[i] = rand() % 10;
    }

    float totalGpuTime = 0.0f;

    for (int i = 0; i < iterations; i++) {
        // allocate memory on device
        cudaMalloc((void**)&d_inputArray, size);
        cudaMalloc((void**)&d_outputArray, numBlocks * sizeof(int));
        cudaMalloc((void**)&d_finalResult, sizeof(int));

        // start timing
        cudaEventRecord(gpu_start, 0);

        // copy input array from host to device
        cudaMemcpy(d_inputArray, h_inputArray, size, cudaMemcpyHostToDevice);

        // perform the first reduction, this will make each block only have 1 final sum
        parallelSum<<<numBlocks, blockSize, sharedMemorySize>>>(d_inputArray, d_outputArray, width);

        // after getting the sum of each blocks, we need to sum all of them again to get the final sum
        if (numBlocks > 1) {
            // since there is only "numOfBlocks" elements left, we can just use fewer blocks to sum these elements 
            // 65526 / 256 = 256 elements left, we can just use 256 threads which is 1 block
            parallelSum<<<1, blockSize, sharedMemorySize>>>(d_outputArray, d_finalResult, numBlocks);
            // copy the final result on gpu back to host
            cudaMemcpy(&h_gpuSum, d_finalResult, sizeof(int), cudaMemcpyDeviceToHost);

        } else {
            // copy the final result on gpu back to host
            cudaMemcpy(&h_gpuSum, d_outputArray, sizeof(int), cudaMemcpyDeviceToHost);
        }

        // stop timing
        cudaEventRecord(gpu_stop, 0);
        cudaEventSynchronize(gpu_stop);

        float iterationTime = 0.0f;
        cudaEventElapsedTime(&iterationTime, gpu_start, gpu_stop);
        totalGpuTime += iterationTime;

        // free gpu mem
        cudaFree(d_inputArray);
        cudaFree(d_outputArray);
        cudaFree(d_finalResult);
    }

    // get avg gpu time
    float avgGpuTime = totalGpuTime / iterations;

    // total cpu time
    float totalCpuTime = 0.0f;

    for (int i = 0; i < iterations; i++) {
        clock_t cpu_start = clock();

        int tempSum = 0;

        // perform normal addition on the cpu
        for (int j = 0; j < width; j++) {
            tempSum += h_inputArray[j];
        }

        clock_t cpu_end = clock();
        float iterationTime = 1000.0f * (cpu_end - cpu_start) / CLOCKS_PER_SEC; // convert to ms by x1000
        totalCpuTime += iterationTime;
    }

    // avg cpu time
    float avgCpuTime = totalCpuTime / iterations;

    // Print out the timings for both cpu and gpu
    printf("GPU with parallel sum takes %fms\n", avgGpuTime);
    printf("CPU with normal sum takes %fms\n", avgCpuTime);

    // perform normal addition on the cpu
    for (int i = 0; i < width; i++) {
        cpuSum += h_inputArray[i];
    }

    // verify the results
    if (cpuSum == h_gpuSum) {
        printf("The parallelSum was running correctly on the GPU as results on both CPU and GPU match\n");
    } else {
        printf("The sum is not similar\n");
    }
    printf("CPU Sum = %d\n", cpuSum);
    printf("GPU Sum = %d\n", h_gpuSum);

    return 0;
}
