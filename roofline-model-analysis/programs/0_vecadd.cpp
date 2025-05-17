#include <iostream>

const int msize = 512*512;

int main() {
    int i, j, k;
    int a[msize];
    int b[msize];
    int c[msize] = {0}; // Initialize to zero

    int iter;
    for (iter = 0; iter < 1000; iter++){
        // Initialize matrices a and b
        for (i = 0; i < msize; i++) {
            a[i] = 2;
            b[i] = 4;
        }

    
        for (i = 0; i < msize; i++) {
            c[i] += a[i] + b[i];
        }
    }
 
    for(i = 0; i < msize; i++){
        if(c[i]==0){
            std::cout << "Something went wrong!!!";
        }
    }

    // Output result
    std::cout << "Done addition of two n*1 vectors where n is " << msize << std::endl;

    return 0;
}
