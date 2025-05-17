#include <stdio.h>
// import more
#include <stdlib.h> 
#include <time.h>


__global__ void matrixMultiplication ( int * A , int * B , int * C , int width ) {
    // TODO : Implement matrix multiplication kernel
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < width && col < width) {
        int single_value = 0;
        for (int i = 0; i < width; i++) {
            single_value += A[row * width + i] * B[i * width + col]; // sum up all the values
        }
        // assign that single value to C entries
        C[row * width + col] = single_value;
    }
}

// cpu for matrix mul 
void cpuMatrixMul(int* A, int* B, int* C, int width) {
    for (int row = 0; row < width; row++) {
        for (int col = 0; col < width; col++) {
            int single_value = 0;
            for (int i = 0; i < width; i++) {
                single_value += A[row * width + i] * B[i * width + col];
            }            
            C[row * width + col] = single_value;
        }
    }
}

int main () {
    const int width = 128; // Matrix width
    int A [ width * width ] , B [ width * width ] , C [ width * width ], D [width*width]; // Host matrices
    int *d_A, *d_B, *d_C; // device pointers
    int iterations = 100; // iterations for timing
    
    // size for the vectors
    size_t size = width * width * sizeof(int);
    
    printf("Using matrix size = %d with width = %d\n", width*width, width);
    // TODO : Initialize matrices ’A’ and ’B’ with random values
    srand(time(NULL));
    for (int i = 0; i < width * width; i++) {
        // rand between 1 and 10: https://stackoverflow.com/questions/17846212/generate-a-random-number-between-1-and-10-in-c
        A[i] = rand() % 10;
        B[i] = rand() % 10;
    }

    // timing stuff
    cudaEvent_t gpu_start, gpu_stop;
    cudaEventCreate(&gpu_start);
    cudaEventCreate(&gpu_stop);

    float totalGpuTime = 0.0f;

    // define multiple iterations and avg 
    for (int i = 0; i < iterations; i++) {

        // TODO : Allocate device memory for matrices ’A ’, ’B ’, and ’C ’
        // cudaMalloc(void **pointer, size_t nbytes) 
        cudaMalloc((void**)&d_A, size);
        cudaMalloc((void**)&d_B, size);
        cudaMalloc((void**)&d_C, size);

        // start timing
        cudaEventRecord(gpu_start, 0);


        // TODO : Copy matrices 'A' and 'B' from host to device 
        // cudaMemcpy(void *dst, void *src, size_t nbytes, enum cudaMemcpyKind direction);
        cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

        // Define grid and block dimensions (this is full dim so we dont need to grid-stride stuff)
        dim3 dimGrid (( width + 15) / 16 , ( width + 15) / 16 , 1);
        dim3 dimBlock (16 , 16 , 1);

        // Launch the matrix multiplication kernel
        matrixMultiplication<<< dimGrid , dimBlock >>>(d_A , d_B , d_C , width );

        // TODO : Copy the result matrix ’C ’ from device to host
        cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

        cudaEventRecord(gpu_stop, 0);
        cudaEventSynchronize(gpu_stop);

        float iterationTime = 0.0f;
        cudaEventElapsedTime(&iterationTime, gpu_start, gpu_stop);
        totalGpuTime += iterationTime;

        // TODO : What is needed here ? -> cudaFree
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
    }

    // get the avg gpu time
    float avgGpuTime = totalGpuTime / iterations;


    // timings for cpu after 100 iterations
    float totalCpuTime = 0.0f;

    for (int i = 0; i < iterations; i++) {
        clock_t cpu_start = clock();

        cpuMatrixMul(A, B, D, width);

        clock_t cpu_end = clock();
        float iterationTime = 1000.0f * (cpu_end - cpu_start) / CLOCKS_PER_SEC; // convert to ms by x1000
        totalCpuTime += iterationTime;
    }

    // avg cpu time
    float avgCpuTime = totalCpuTime / iterations;

    // Print out the timings for both cpu and gpu
    printf("GPU takes %fms\n", avgGpuTime);
    printf("CPU takes %fms\n", avgCpuTime);

    // TODO : Verify the correctness of the result
    // loop thru each element in each
    int correct = 1;
    for (int i = 0; i < width * width; i++) {
        if (C[i] != D[i]) {
            correct = 0;
            printf("Index mismatch at %d, GPU: %d, CPU: %d\n", i, C[i], D[i]);
            break;
        }
    }

    if (correct == 1) {
        printf("The matrix mul is right as both CPU and GPU matches\n");
    }

    // destroy the cudaEvent stuff
    cudaEventDestroy(gpu_start);
    cudaEventDestroy(gpu_stop);

    return 0;
}

