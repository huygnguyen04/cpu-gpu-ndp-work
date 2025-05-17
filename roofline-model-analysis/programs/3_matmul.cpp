#include <iostream>
#include <omp.h> // For OpenMP

const int msize = 512;

int main() {
    int i, j, k;
    int a[msize][msize];
    int b[msize][msize];
    int t[msize][msize];
    int c[msize][msize] = {0}; // Initialize to zero

    // Initialize matrices a and b
    for (i = 0; i < msize; i++) {
        for (j = 0; j < msize; j++) {
            a[i][j] = 2;
            b[i][j] = 4;
        }
    }
    
    //Transpose matrix b
    for (i = 0; i < msize; i++) {
        for (j = 0; j < msize; j++) {
            t[i][j] = b[j][i];
        }
    }
    #define CHUNK_SIZE 16
    int ichunk, jchunk, ci, cj;
    for (ichunk = 0; ichunk < msize; ichunk += CHUNK_SIZE) {
        for (jchunk = 0; jchunk < msize; jchunk += CHUNK_SIZE) {
            for (i = 0; i < CHUNK_SIZE; i++) {
                ci = ichunk + i;
                for (j = 0; j < CHUNK_SIZE; j++) {
                    cj = jchunk + j;
                    int sum = 0;
                    #pragma omp simd reduction(+:sum)
                    for (k = 0; k < msize; k++) {
                        sum += a[ci][k] * t[cj][k];
                    }
                    c[ci][cj] = sum;
                }
            }
        }
    }
    
    for(i = 0; i < msize; i++){
        for(j = 0; j < msize; j++){
            //std::cout << c[i][j] << std::endl;
            if(c[i][j]==0){
                std::cout << "Something went wrong!!!";
            }
        }
    }

    // Output result
    std::cout << "Done multiplication of n*n and n*n matrices where n is " << msize << std::endl;

    return 0;
}

