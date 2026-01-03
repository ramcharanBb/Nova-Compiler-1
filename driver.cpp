#include <iostream>
#include <vector>
#include <cstdint>
#include <cstdlib>

struct MemRefDescriptor2 {
    float *allocated;
    float *aligned;
    int64_t offset;
    int64_t sizes[2];
    int64_t strides[2];
};

extern "C" MemRefDescriptor2 main1();

void printTensorRec(float* data, int64_t* sizes, int64_t* strides, int64_t rank, int64_t curDim, int64_t offset) {
    if (curDim == rank) {
        std::cout << data[offset] << " ";
        return;
    }

    std::cout << "[ ";
    for (int64_t i = 0; i < std::min((int64_t)5, sizes[curDim]); ++i) {
        printTensorRec(data, sizes, strides, rank, curDim + 1, offset + i * strides[curDim]);
    }
    if (sizes[curDim] > 5) std::cout << "... ";
    std::cout << "]";
    if (curDim < rank - 1) std::cout << "\n";
}

int main() {
    std::cout << "Calling main1 (Rank 2)..." << std::endl;
    MemRefDescriptor2 res = main1();
    
    float* aligned = res.aligned;
    int64_t offset = res.offset;
    int64_t* sizes = res.sizes;
    int64_t* strides = res.strides;

    std::cout << "main1 returned!" << std::endl;
    std::cout << "Base Ptr: " << aligned << std::endl;
    std::cout << "Shape: (" << sizes[0] << "x" << sizes[1] << ")" << std::endl;

    if (aligned && sizes[0] > 0 && sizes[1] > 0) {
        std::cout << "Data (Sample):\n";
        printTensorRec(aligned, sizes, strides, 2, 0, offset);
        std::cout << std::endl;
    }

    return 0;
}
