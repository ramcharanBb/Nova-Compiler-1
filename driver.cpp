#include <iostream>
#include <vector>
#include <cstdint>
#include <cstdlib>
#include <cassert>

struct GenericMemRefDescriptor {
    uint64_t padding[128]; // 1024 bytes, enough for rank ~60
};

extern "C" GenericMemRefDescriptor main1();

// Helper to print tensor recursively
void printTensorRec(float* data, int64_t* sizes, int64_t* strides, int64_t rank, int64_t curDim, int64_t offset) {
    if (curDim == rank) {
        std::cout << data[offset] << " ";
        return;
    }

    std::cout << "[ ";
    for (int64_t i = 0; i < sizes[curDim]; ++i) {
        printTensorRec(data, sizes, strides, rank, curDim + 1, offset + i * strides[curDim]);
    }
    std::cout << "]";
    if (curDim < rank - 1) std::cout << "\n"; // Newline for outer dimensions
}

int main(int argc, char** argv) {
    int rank = 2; // Default to 2
    if (argc > 1) {
        rank = std::atoi(argv[1]);
    }

    std::cout << "Calling main1 (assuming Rank " << rank << ")..." << std::endl;
    GenericMemRefDescriptor raw = main1();
    
    // Parse the raw buffer based on Standard MemRef Layout:
    // ptr allocated, ptr aligned, i64 offset, [rank x i64] sizes, [rank x i64] strides
    
    int64_t* buffer = (int64_t*)&raw;
    float* allocated = (float*)buffer[0];
    float* aligned = (float*)buffer[1];
    int64_t offset = buffer[2];
    
    int64_t* sizes = &buffer[3];
    int64_t* strides = &buffer[3 + rank]; // Strides follow sizes immediately

    std::cout << "main1 returned!" << std::endl;
    std::cout << "Base Ptr: " << aligned << std::endl;
    
    // Debug info
    std::cout << "Shape: (";
    for(int i=0; i<rank; ++i) std::cout << sizes[i] << (i<rank-1 ? "x" : "");
    std::cout << ")" << std::endl;

    if (aligned) {
        // Since we are dealing with dense tensors in this test, strides might be identity.
        // We use the strides from the descriptor to be correct.
        
        // Print
        std::cout << "Data:\n";
        printTensorRec(aligned, sizes, strides, rank, 0, offset);
        std::cout << std::endl;
    }

    return 0;
}
