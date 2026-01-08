#include <iostream>
#include <vector>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <cmath>
#include <cuda_runtime.h>

struct MemRefDescriptor {
    float *allocated;
    float *aligned;
    int64_t offset;
    int64_t sizes[2];
    int64_t strides[2];
};

extern "C" {
    void _mlir_ciface_test_cpu_only(MemRefDescriptor* res, MemRefDescriptor* in);
    void _mlir_ciface_test_gpu_only(MemRefDescriptor* res, MemRefDescriptor* in0, MemRefDescriptor* in1);
    void _mlir_ciface_test_mixed_h2d(MemRefDescriptor* res, MemRefDescriptor* in);
    void _mlir_ciface_test_mixed_d2h(MemRefDescriptor* res, MemRefDescriptor* in);
    void _mlir_ciface_test_stress_chain(MemRefDescriptor* res, MemRefDescriptor* in);
}

void init_memref(MemRefDescriptor& desc, int64_t rows, int64_t cols, float val, bool on_gpu = false) {
    desc.sizes[0] = rows; desc.sizes[1] = cols;
    desc.strides[0] = cols; desc.strides[1] = 1;
    desc.offset = 0;
    int64_t size = rows * cols * sizeof(float);
    if (on_gpu) {
        cudaMalloc(&desc.allocated, size);
        std::vector<float> tmp(rows * cols, val);
        cudaMemcpy(desc.allocated, tmp.data(), size, cudaMemcpyHostToDevice);
    } else {
        desc.allocated = (float*)malloc(size);
        for(int i=0; i<rows*cols; ++i) desc.allocated[i] = val;
    }
    desc.aligned = desc.allocated;
}

void free_memref(MemRefDescriptor& desc, bool on_gpu = false) {
    if (on_gpu) cudaFree(desc.allocated);
    else free(desc.allocated);
}

void print_sample(const std::string& name, const MemRefDescriptor& desc, bool on_gpu = false) {
    int64_t rows = desc.sizes[0];
    int64_t cols = desc.sizes[1];
    std::vector<float> host_data(rows * cols);
    if (on_gpu) {
        cudaMemcpy(host_data.data(), desc.aligned, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);
    } else {
        for(int i=0; i<rows*cols; ++i) host_data[i] = desc.aligned[i];
    }
    std::cout << name << " [0,0]: " << host_data[0] << std::endl;
}

int main() {
    std::cout << "Starting Robustness Tests..." << std::endl;


    // 1. CPU Only
    {
        MemRefDescriptor in, res;
        init_memref(in, 8, 8, 1.0f, false);
        _mlir_ciface_test_cpu_only(&res, &in);
        print_sample("CPU Only (Transpose)", res, false);
        free_memref(in, false);
    }

    // 2. GPU Only
    {
        MemRefDescriptor in0, in1, res;
        init_memref(in0, 16, 16, 1.0f, true);
        init_memref(in1, 16, 16, 2.0f, true);
        _mlir_ciface_test_gpu_only(&res, &in0, &in1);
        print_sample("GPU Only (Add+Matmul)", res, true);
        free_memref(in0, true);
        free_memref(in1, true);
    }

    // 3. Mixed H2D
    {
        MemRefDescriptor in, res;
        init_memref(in, 32, 32, 1.0f, false);
        _mlir_ciface_test_mixed_h2d(&res, &in);
        print_sample("Mixed H2D (Host->Device Add)", res, true);
        free_memref(in, false);
    }

    // 4. Mixed D2H
    {
        MemRefDescriptor in, res;
        init_memref(in, 32, 32, 1.0f, true);
        _mlir_ciface_test_mixed_d2h(&res, &in);
        print_sample("Mixed D2H (Device->Host Transpose)", res, false);
        free_memref(in, true);
    }

    // 5. Stress Chain
    {
        MemRefDescriptor in, res;
        init_memref(in, 64, 64, 0.5f, true);
        _mlir_ciface_test_stress_chain(&res, &in);
        print_sample("Stress Chain (Add+Matmul+Gelu+Softmax)", res, true);
        free_memref(in, true);
    }

    std::cout << "Robustness Tests Completed." << std::endl;
    return 0;
}
