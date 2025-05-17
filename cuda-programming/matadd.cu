#include <stdio.h>
// import more stuff
#include <stdlib.h>
#include <time.h>

// CUDA kernel for matrix addition
__global__ void matrixAddition ( int * A , int * B , int * C , int width , int height ) {
    int row = blockIdx . y * blockDim . y + threadIdx . y ;
    int col = blockIdx . x * blockDim . x + threadIdx . x ;
    if ( row < height && col < width ) {
        C [ row * width + col ] = A [ row * width + col ] +
        B [ row * width + col ];
        }
}

// cpu for matrix add to compare and timing later
void cpuMatrixAdd(int *A, int *B, int *C, int width, int height) {
    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
            C[row * width + col] = A [ row * width + col ] +
            B [ row * width + col ];
        }
    }
}

int main () {
    const int width = 1; // Matrix width
    const int height = 1; // Matrix height
    size_t size = width * height * sizeof(int); // matrix size in bytes for malloc

    // int A[width * height] , B[width * height] , C[width * height], D[width * height];

    int *A = (int*)malloc(size);
    int *B = (int*)malloc(size);
    int *C = (int*)malloc(size);
    int *D = (int*)malloc(size);

    int iterations = 1; // iterations to get steady time
	

    printf("Using array of size %d, with width = %d and height = %d\n", width*height, width, height);
    // timing stuff 
    cudaEvent_t gpu_start, gpu_stop;
    cudaEventCreate(&gpu_start);
    cudaEventCreate(&gpu_stop);

    // TODO : Initialize matrices ’A ’ and ’B ’ with random values (host matrices)
    srand(time(NULL));
    for (int i = 0; i < width * height; i++) {
        // rand between 1 and 10: https://stackoverflow.com/questions/17846212/generate-a-random-number-between-1-and-10-in-c
        A[i] = rand() % 10;
        B[i] = rand() % 10;
    }

    // TODO : Declare pointers for device matrices
    int *d_A, *d_B, *d_C;

    // total gpu time
    float totalGpuTime = 0.0f;

    for (int i = 0; i < iterations; i++) {
        // TODO : Allocate device memory for matrices ’A ’, ’B ’, and ’C ’
        cudaMalloc((void**)&d_A, size);
        cudaMalloc((void**)&d_B, size);
        cudaMalloc((void**)&d_C, size);

        // start the timing
        cudaEventRecord(gpu_start, 0);

        // TODO : Copy matrices ’A ’ and ’B ’ from host to device
        cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

        // Define grid and block dimensions
        dim3 dimGrid (( width + 15) / 16 , ( height + 15) / 16 , 1);
        dim3 dimBlock (16 , 16 , 1);
        
        // Launch the matrix addition kernel
        matrixAddition <<<dimGrid,dimBlock>>> (d_A , d_B , d_C , width , height );
        
        // TODO : Copy the result matrix ’C ’ from device to host
        cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

        // end timing here
        cudaEventRecord(gpu_stop, 0);
        cudaEventSynchronize(gpu_stop);

        float iterationTime = 0.0f;
        cudaEventElapsedTime(&iterationTime, gpu_start, gpu_stop);
        totalGpuTime += iterationTime;

        // TODO : Free allocated memory
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

        cpuMatrixAdd(A, B, D, width, height);

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
    cpuMatrixAdd(A, B, D, width, height);
    int correct = 1;
    for (int i = 0; i < width * height; i++) {
        // printf("%d, %d\n", C[i], D[i]);
        if (C[i] != D[i]) {
            correct = 0;
            printf("Result does not match at index %d, GPU: %d and CPU: %d", i, C[i], D[i]);
            break;
        }
    }
    if (correct == 1) {
        printf("The vector add is right as both CPU and GPU matches\n");
    }
    
    // destroy the cudaEvent stuff
    cudaEventDestroy(gpu_start);
    cudaEventDestroy(gpu_stop);
    return 0;   
}
