#include <iostream>

const int msize = 512;

int main() {
    int i, j, k;
    int a[msize][msize];
    int b[msize][msize];
    int c[msize][msize] = {0}; // Initialize to zero

    // Initialize matrices a and b
    for (i = 0; i < msize; i++) {
        for (j = 0; j < msize; j++) {
            a[i][j] = 2;
            b[i][j] = 4;
        }
    }

    for (i = 0; i < msize; i++) {
        for (j = 0; j < msize; j++) {
            for (k = 0; k < msize; k++) {
                c[i][j] += a[i][k] * b[k][j];
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

