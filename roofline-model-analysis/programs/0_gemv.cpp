#include <iostream>

const int msize = 512; // original was 512

int main() {
    int i, j;
    int a[msize][msize];
    int b[msize];
    int c[msize] = {0}; // Initialize to zero

    // Initialize matrices a and b
    for (i = 0; i < msize; i++) {
        b[i] = 4;
        for (j = 0; j < msize; j++) {
            a[i][j] = 2;
        }
    }
    int iter;
    for (iter=0; iter<1000; iter++){
        for (i = 0; i < msize; i++) {
            for (j = 0; j < msize; j++) {
                c[i] += a[i][j] * b[j];
            }
        }
    }
 
    for(i = 0; i < msize; i++){
        if(c[i]==0){
            std::cout << "Something went wrong!!!";
        }
    }

    // Output result
    std::cout << "Done multiplication of n*n matrix with n*1 vector where n is " << msize << std::endl;

    return 0;
}

